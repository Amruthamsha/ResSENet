"""Training and evaluation loop for the FER2013 benchmark."""

import copy
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import DEFAULT_CONFIG


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=DEFAULT_CONFIG.grad_clip):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


def train_model(
    model,
    train_loader,
    val_loader,
    class_weights_tensor,
    device,
    name: str = "model",
    config: DEFAULT_CONFIG.__class__ = DEFAULT_CONFIG,
):
    """Full training loop with early stopping and ReduceLROnPlateau scheduling."""
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=config.lr_patience)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_ctr = 0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, config.grad_clip)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_val_acc = vl_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch <= 5:
            elapsed = time.time() - t0
            print(
                f"[{name}] Ep {epoch:3d}/{config.epochs} | "
                f"TR {tr_acc:.1f}% | VL {vl_acc:.1f}% | "
                f"VL_loss {vl_loss:.4f} | {elapsed:.0f}s"
            )

        if patience_ctr >= config.patience:
            print(f"[{name}] Early stop at epoch {epoch}. Best val acc: {best_val_acc:.1f}%")
            break

    model.load_state_dict(best_state)
    print(f"[{name}] Training done. Best val acc: {best_val_acc:.1f}%  ({time.time()-t0:.0f}s)")
    return model, history
