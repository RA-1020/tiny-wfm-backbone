# The Contract
- Dataset: RadioML 2018.01A via kaggle mirror
- Non-IID split: Dirichlet alpha=0.5, N=10 clients, seed=42
- Channel: Sionna CDL-C NLOS @ 3.5 GHz, SNR -10..20 dB,Sionna >=2.0, model + params being finalized in Stage E
- Spectrogram: STFT n_fft=64, hop=16,fftshift applied, resize 224x224, per-sample standardized
- Metrics: top-1 accuracy, epsilon (delta=1e-5), attack-success rate,
  model size (MB, = MiB? state it), MACs via ptflops, bytes uploaded/client/round
