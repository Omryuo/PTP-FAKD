# PTP-FAKD: Progressive Token Pruning with Feature-Aware Distillation for ViTs

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FIECCT68664.2026.11541604-blue)](https://doi.org/10.1109/IECCT68664.2026.11541604)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-omryuo.github.io%2FPTP--FAKD-brightgreen)](https://omryuo.github.io/PTP-FAKD/)

**🔗 Live project page (architecture overview + demo): [omryuo.github.io/PTP-FAKD](https://omryuo.github.io/PTP-FAKD/)**

Official code, trained weights, and supporting materials for the paper **"Progressive Token Pruning with Feature-Aware Distillation for ViTs"**, published in the *2026 International Conference on Recent Advancement in Electrical, Computer and Communication Technologies (IECCT)*, Bangalore, India (Paper ID 1523).

> **Headline result:** PTP-FAKD reaches **89.60% top-1 accuracy on CIFAR-100** at **50% token density** using ViT-B/16 — **surpassing the dense teacher by 2.57 percentage points** while pruning half the tokens.

---

## Overview

Vision Transformers are accurate but expensive: attention scales with the number of tokens, and most tokens contribute little to the final prediction. PTP-FAKD attacks this on two fronts at once:

1. **Progressive Token Pruning (PTP)** — tokens are pruned gradually across transformer stages rather than all at once, so the model retains the information it needs early and sheds redundancy deeper in the network.
2. **Feature-Aware Knowledge Distillation (FAKD)** — a dense teacher transfers not just output logits but intermediate feature structure to the pruned student, recovering (and here, exceeding) the accuracy lost to pruning.

The combination lets the pruned student outperform its own dense teacher — the central finding of the paper.

---

## Key Results

| Model / Configuration | Dataset | Token Density | Top-1 Acc. |
|---|---|---:|---:|
| Dense ViT-B/16 (teacher) | CIFAR-100 | 100% | 87.03% |
| Sparse ViT (no KD) | CIFAR-100 | 50% | 78.29% |
| Progressive pruning, no KD (ablation) | CIFAR-100 | 50% | 77.28% |
| Naive feature KD (ablation) | CIFAR-100 | 50% | 84.91% |
| **PTP-FAKD (full method)** | CIFAR-100 | 50% | **89.60%** |

> The dense teacher accuracy and ablation numbers are recorded in the `*_result.json` files in `Model using C100/`. Fill the table from those so the README matches your paper exactly.

---

## Repository Structure

```
PTP-FAKD/
├── Model using C100/                 # Main experiments (CIFAR-100)
│   ├── cifar100-dense-sparse-vit.ipynb
│   ├── cifar-100-sparsevit-kd(0.77).ipynb
│   ├── cifar-100-token-ratio-ablation-sweep.ipynb
│   ├── sparse-kd-cifar-100.ipynb
│   ├── naive_sparse_featkd_result.json
│   └── progressive_nokd_result.json
├── Model using C10/                  # Preliminary experiments (CIFAR-10)
│   ├── dense-sparse-no-kd.ipynb
│   └── sparsevit-kd-using-saved-dense-teacher.ipynb
├── Conference materials/             # Published paper, camera-ready, presentation
├── Project report.pdf                # Full project report
├── index.html                        # Live project page (served via GitHub Pages)
└── README.md
```

Trained weights (`.pt`) are **not** stored in the repository — they are published under [Releases](../../releases) (see below).

---

## Pretrained Weights

The trained checkpoints are the experimental proof behind every number in the paper. They are attached to the [**v1.0 release**](../../releases/tag/v1.0).

### CIFAR-100 (main experiments)

| File | Role |
|---|---|
| `dense_cifar100_best.pt` | Dense ViT-B/16 teacher (baseline) |
| `sparse_cifar100_best.pt` | Sparse student, no distillation |
| `progressive_nokd_best.pt` | Progressive pruning, no KD (ablation) |
| `naive_sparse_featkd_best.pt` | Single-stage feature KD (ablation) |
| `sparse_kd_cifar100_final.pt` | **Full PTP-FAKD method — headline 89.60% result** |

### CIFAR-10 (preliminary experiments)

| File | Role |
|---|---|
| `dense_vit.pt` | Dense teacher |
| `sparse_vit.pt` | Sparse student, no KD |
| `best_sparse_kd.pt` | Full PTP-FAKD method |

> Verify these filename-to-role mappings against your training notebooks before publishing — they're inferred from filenames and should match how each checkpoint was saved.

### Loading a checkpoint

```python
import torch

ckpt = torch.load("sparse_kd_cifar100_final.pt", map_location="cpu")
# If the checkpoint stores a state_dict:
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
```

---

## Reproducing the Experiments

The experiments run as self-contained Jupyter notebooks (developed on Kaggle/Colab GPU).

```bash
# 1. Create and activate an environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt   # add this file if not present (see note below)

# 3. Launch notebooks
jupyter notebook
```

Run order:
1. `cifar100-dense-sparse-vit.ipynb` — trains the dense teacher and the sparse baseline.
2. `sparse-kd-cifar-100.ipynb` — runs the full PTP-FAKD distillation (produces the headline result).
3. `cifar-100-token-ratio-ablation-sweep.ipynb` — token-density sweep and ablations.

> **Note:** this repo currently has no `requirements.txt`. Generate one from your working environment with `pip freeze > requirements.txt` so the setup above works for anyone cloning it.

---

## Citation

If you use this work, please cite the published paper:

```bibtex
@inproceedings{tm2026ptpfakd,
  author    = {O. T. M and S. S. Mokashi and B. Harsha and S. H. K and P. Jain},
  title     = {Progressive Token Pruning with Feature-Aware Distillation for {ViTs}},
  booktitle = {2026 International Conference on Recent Advancement in Electrical,
               Computer and Communication Technologies (IECCT)},
  address   = {Bangalore, India},
  year      = {2026},
  pages     = {1--5},
  doi       = {10.1109/IECCT68664.2026.11541604}
}
```

**Plain-text (IEEE):** O. T. M, S. S. Mokashi, B. Harsha, S. H. K and P. Jain, "Progressive Token Pruning with Feature-Aware Distillation for ViTs," *2026 International Conference on Recent Advancement in Electrical, Computer and Communication Technologies (IECCT)*, Bangalore, India, 2026, pp. 1–5, doi: 10.1109/IECCT68664.2026.11541604.

---

## Author

**Omswaroop T M** — Cybersecurity & ML researcher
GitHub: [@Omryuo](https://github.com/Omryuo)

---

## License

Released under the MIT License — see `LICENSE` for details.
