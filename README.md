# Tiny Federated Wireless Foundation Model — Backbone

**Phase 0 of a research project at the IPT Lab, SEECS NUST** — building a tiny, federated, privacy-preserving wireless foundation model: a Vision Transformer trained on radio spectrograms, fine-tuned across devices via federated learning without sharing raw data.

Builds on the lab's paper **"Tiny Federated Wireless Foundation Models for Resource-Constrained Devices"** (IEEE Internet of Things Journal, DOI: [10.1109/JIOT.2025.3591169](https://doi.org/10.1109/JIOT.2025.3591169)), which itself builds on ["Building 6G Radio Foundation Models with Transformer Architectures"](https://arxiv.org/abs/2411.09996) (ViT + Masked Spectrogram Modeling).

## The pipeline

```
raw IQ signal ──▶ Sionna channel ──▶ STFT ──▶ log-magnitude ──▶ 224×224 ──▶ ViT-tiny
                  (fading + noise)            spectrogram
                                                    │
              many non-IID clients each train locally ──▶ FedAvg
                                                    │
                        metrics ──▶ W&B (accuracy · comm cost · size · MACs · ε · ASR)
```

Two research tracks share this backbone:

| Track | Owner | Focus |
|---|---|---|
| **Efficiency** | Taha | Pruning · quantization · knowledge distillation |
| **Security & Privacy** | Rohaan | DP-SGD · secure aggregation · poisoning / membership-inference defenses |

## Build status

| Stage | What | Status |
|---|---|---|
| A | Environment, accounts, shared repo | Done |
| B | ViT-tiny on CIFAR-10 (learning milestone) | Done |
| C | RadioML → spectrogram pipeline | Done |
| D | Centralized baseline on spectrograms | Done |
| E | Sionna channel integration | Done  |
| F | Federation with Flower (FedAvg, non-IID) |  Next |
| G | Shared metrics module | Planned |
| H | Integrate, validate, freeze `backbone-v1` | Planned |

## Results so far

**Centralized baseline (Stage D):** ViT-tiny (ImageNet-pretrained, 1-channel input) on a 12k-example RadioML 2018.01A subset — 10 classes, SNR ∈ {10, 14, 18} dB, 80/20 split, seed 42.

- **89.1% val accuracy** (chance = 10%) after 8 epochs
- Spectrally distinct classes (QPSK, 64QAM, AM-DSB-WC, FM, OOK, 16QAM): **99–100%**
- Confusions concentrate exactly where theory predicts: **GMSK ↔ OQPSK** and **8PSK → BPSK** — constant-envelope schemes whose discriminating information lives in *phase*, which magnitude-only spectrograms discard. A known representation limit, kept deliberately for comparability with the base paper.

> This number is an internal reference under deliberately easy conditions (high SNR, 10/24 classes, no channel, no federation). It is **not** comparable to the paper's results — it exists to measure what Stages E and F cost.

**Found & fixed along the way:** baseband energy at DC splits across FFT array edges, swamping per-sample normalization and making all classes look identical. Fix: `fftshift` to center 0 Hz + `n_fft` 256→64, hop 128→16 (9 → 61 real time frames). Validated via intra-class consistency (5/5 samples per class share a stable signature).

## Repo structure

```
├── dataset.py            # RadioML subset → spectrogram PyTorch Dataset (the core transform)
├── sanity_plot.py        # one spectrogram per class — the "does this look right" check
├── intraclass_check.py   # N samples of one class — validates signature stability
├── debug_spectrogram.py  # per-frequency-bin energy diagnostics
├── metrics.py            # shared metrics module (Stage G — placeholder)
├── notebooks/
│   ├── stage_b_vit_cifar10.ipynb            # ViT fundamentals: scratch vs pretrained
│   └── stage_d_centralized_baseline.ipynb   # spectrogram baseline + confusion matrix
├── CONTRACT.md           # frozen experimental agreements (PROPOSED — pending supervisor sign-off)
├── environment.yml       # pinned conda env — the single source of truth
└── requirements-lock.txt # exact versions snapshot
```

## Setup

```bash
git clone git@github.com:RA-1020/tiny-wfm-backbone.git
cd tiny-wfm-backbone
conda env create -f environment.yml
conda activate wfm
```

**Data** (not in the repo — see `.gitignore`): extract `radioml_subset.npz` from the RadioML 2018.01A Kaggle mirror using the extraction snippet in the Stage D notebook, or grab it from the team Drive. Drop it in the repo root.

**Verify:**
```bash
python sanity_plot.py     # should show 10 visibly distinct class spectrograms
```

Experiments log to the shared W&B project [`ra-10/tiny-wfm-backbone`](https://wandb.ai/ra-10/tiny-wfm-backbone) — one API key per person, never committed.

## The contract

Before any cross-track experiments, three things are frozen in [`CONTRACT.md`](CONTRACT.md) so results stay combinable: **one data partition** (Dirichlet α=0.5, N=10 clients, seed 42), **one channel simulator** (Sionna — version pending confirmation against the lab paper), **one metrics schema** (top-1 accuracy, ε @ δ=1e-5, attack-success rate, model size, MACs, bytes/round). Quiet changes to any of these break the papers — don't.

## Roadmap beyond Phase 0

1. **Paper 1 — Make it private:** secure aggregation + DP on the backbone; the privacy-vs-accuracy-vs-communication trade-off curves
2. **Paper 2 — The flagship:** how compression reshapes privacy and robustness (the literature genuinely conflicts — settling it is the contribution)
3. **Paper 3 — Robust over real channels:** does fading hide poisoned updates? Channel-aware attacks and defenses

---

*IPT Lab · SEECS, NUST Islamabad — supervised by Prof. Syed Ali Hassan*
