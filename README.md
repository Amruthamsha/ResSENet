# ResSENet: Residual Squeeze-and-Excitation Networks for Robust Facial Emotion Recognition

Publication-grade benchmark implementation on FER2013. The core architectural claim is
where Squeeze-and-Excitation is placed inside a residual block: **SE embedded inside the
residual branch** (this paper) vs. the conventional **post-hoc placement** (SE applied after
the residual addition), evaluated as an iso-parametric control condition.

- **Eq. 1** (proposed): X̂ = X + SE(F(X)) — SE gates the residual branch before the identity add
- **Eq. 2**: s = σ(W₂·δ(W₁·GAP(X))), X̂ = X·s — squeeze-excite gating, reduction ratio r=16

## Paper

Published open access in *Array* 31 (2026) 101163 (Elsevier), under CC BY 4.0 license.

📄 [Read the paper (PDF)]([paper/ressenet_fer_array_2026.pdf](https://pdf.sciencedirectassets.com/320039/1-s2.0-S2590005626X20021/1-s2.0-S2590005626004868/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIC%2FOzBsk6G9j4gVp5PwX8xo0jmttKyxo1RtazudbiBrqAiBm3XCg6zt1WfCZf8SPaRLL0uJhPKuZFUAhHz6VVdHlpiq7BQir%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMSmuKF3I7qt%2BenDeQKo8FG344wSaVojY9tKf%2BKz4T9F0NAmru7dhgH7l047NbyHUfndDdDyiGu7qv5%2BlfEx%2FXE7XyaoQ8%2BiWDZjESAxRAJGDH7Iv4KcdNh4QqjIDDrWOUdPzdAK8EhwvoCGUEa2d6%2BQEwyHz5qH8hkZBoRvHqLhbda2gnil19BWS2zpVTobLlkh7Xt7O%2B5m8qoNZyRFROA%2F5%2FOQ6hboIfjYNydhF7vOs7VqMQJLQa0eMJa6N1KXWcnVHUeda%2FmKTtwZOAR8t3S0gdXGknZFuK2uOohlPnNUaKudO%2BtJAtCUolKaDhiyrlb7vUzv%2FSUwEEDttAepWr1rcmgafzt%2FzZpdbcjaVgqVVyNiytXN6AugNzvp5foZMJVsZB5alWAGZdWqV6RoC%2Fw7DxfwPEy2QHuzD6OoN2PwM88VZICRBvy1%2FQ30HDZL%2FDbrZIaWWXjKhVjVCTq%2Fghsn2dxuF7lMtkuddp5S1qExIpHL0IBrWY8moK9DHRUd%2BEt4tcNX6oPgCmMsoyzc42YVU5uP2Ogoj4ku%2BoTCZr3zda8Oa8n%2BpT%2FhWKhaaAZ9cfiSr9K7ey5EjlPeWFS4yx1mWu2axGS5M%2B0U%2Fkx7W%2FGwjvAha5cA1odJr3Pwf%2F6jPwkEU42LXGXpltAZwQVhFJHIP4twE5NKZjO1m85jBoDaiaPT9KLamK5kGVKhxoy68mvzsNDzG3poF47Pzw3uxUYE3%2FDtCOMyxxDFgt2jPQ4nhnqjCD8wf4zFglLgohtQYZfq5%2F06zWaX9MIN0IW1dDoFco82AFYrQr%2BIF3B9U48m2unymIjJbDJJ%2BYHQqsc%2FThNsxDnp6O8FuS3dn7PUGWcLYj%2FhzBG0je4WSjTDiCp27VmPrgdbhCSPOcwNM99jClnKLUBjqyAQwGkA7RVYlMV7NRJzDbPO0J4YXM31i5CTGf0v1P2UW8TqJkHoqOKdXy6JWHj28peJXzPMcKdAd4RTRYmXlnZ0CKpHXbKI%2FoYkro0jFsfTuDB4vsZ6PbF1QnSjRPGr00I7X24gqYpIW%2B6907c2qT4Ho%2Bs%2FHftgDnzWouCSBXcC7i46jcNTRxWVtjZKKpz6xlUYkBPrQxjeugU%2BQ%2FGdG61aBh6aToUXvDPDcbOLihu6%2BK9Mo%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260821T183020Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYWWWPSDOY%2F20260821%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=7ed43dffb5b85bda2bf2f997243710208eae7f12cb174cc4e30802f192b6c535&hash=0b4c2d8ab299a64d4e7d3a8a68cbbfd1605080e3b6d10c65ed47f78ec0c06d51&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2590005626004868&tid=spdf-a01c0109-4aac-4072-af98-dc32eff707af&sid=b525f28055db444b088a924-29b9531945c2gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f0005570204075b5453&rr=a2ebb5562db8f470&cc=in))
🔗 DOI: [10.1016/j.array.2026.101163](https://doi.org/10.1016/j.array.2026.101163)

If you use this code, please cite:
> Raju, A. P., Sudhakar, K., & Rubini, P. (2026). Investigating squeeze-and-excitation placement in residual networks for facial emotion recognition. *Array*, 31, 101163. https://doi.org/10.1016/j.array.2026.101163

## What this reproduces

- **Table I** — comparative evaluation: BaselineCNN, ResNet-18, VGGNet-13, SE-ResNet
  (post-hoc control), ResSENet (proposed), ResNet-50, EfficientNet-B0, ResSENet+ (pretrained stem)
- **Table II** — ablation study (4 variants: with/without residual, with/without SE)
- Computational complexity (FLOPs, param count, GPU/CPU latency, throughput)
- Training-dynamics comparison: ResSENet vs. post-hoc SE-ResNet (isolates the placement claim)
- Per-class confusion matrix, GradCAM interpretability, accuracy-vs-parameter trade-off plot

## Repo structure

```
ressenet-fer/
├── src/
│   ├── config.py     # Config dataclass — all hyperparameters in one place
│   ├── models.py      # SEModule, ResSEBlock, ResSENet, SEPostHocBlock/SEResNetPostHoc
│   │                    # (iso-parametric control), BaselineCNN, pretrained-baseline getters
│   ├── dataset.py      # FER2013Dataset, augmentation pipeline, loader construction
│   └── train.py        # train_one_epoch / evaluate / train_model (early stopping + LR scheduling)
├── notebooks/
│   └── ResSENet_FER2013_benchmark.ipynb   # Full benchmark: Table I/II, ablations, GradCAM, figures
├── requirements.txt
└── README.md
```

`src/` holds the architecture and training code as an importable package; the notebook is the
full reviewer-response benchmark suite (data download, all 8 models trained end-to-end,
statistical/ablation analysis, publication figures) and imports its model/training logic from
`src/` rather than redefining it inline.

## ⚠️ Flagged discrepancy — verify before submission

Running `ResSENet(num_classes=7)` gives **~11.3M parameters**, not the **~4.2M** figure stated
in the original notebook's own sanity-check comment (and, if it appears there too, the
manuscript). The architecture is a 4-stage network with channel progression 64→128→256→512 —
structurally similar in scale to ResNet-18 (~11.7M params at ImageNet resolution), which is
consistent with the 11.3M measured here. This wasn't something I could resolve without the
manuscript text: either the "~4.2M" figure is a stale comment left over from an earlier,
narrower version of the architecture, or the actual channel/depth configuration in the paper
differs from what's implemented. **Check this against the submitted/draft manuscript before
treating Table I's parameter-count column as correct** — sending a paper with a 2.7x param-count
error to reviewers (or profs) would undercut the "iso-parametric control" framing that Table I
and the ablation study depend on.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/ressenet-fer.git
cd ressenet-fer
pip install -r requirements.txt
```

Get `fer2013.csv` from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) — either via
the Kaggle API (`kaggle.json` in the repo root) or manual download.

## Usage

Open `notebooks/ResSENet_FER2013_benchmark.ipynb` in Colab (~90 min on a T4 for the full
8-model benchmark; set `QUICK_RUN=True` in Cell 1 for a 5-epoch smoke test, ~8 min).

As a library:
```python
from src import Config, ResSENet, SEResNetPostHoc, load_fer2013, build_loaders, train_model

cfg = Config(quick_run=True)
train_df, val_df, test_df = load_fer2013("fer2013.csv")
train_loader, val_loader, test_loader, class_weights = build_loaders(train_df, val_df, test_df, cfg)

model = ResSENet(num_classes=cfg.num_classes)
model, history = train_model(model, train_loader, val_loader, class_weights, device="cuda", config=cfg)
```

## Status

Full benchmark suite implemented and validated: 8 architectures, ablation study, GradCAM,
computational complexity metrics, and the iso-parametric post-hoc-SE control condition
requested in review. Parameter-count discrepancy above needs resolving before submission.

## License

MIT
