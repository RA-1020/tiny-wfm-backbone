# The Contract
- Dataset: RadioML 2018.01A via kaggle mirror
- Non-IID split: Dirichlet alpha=0.5, N=10 clients, seed=42
- Channel: Sionna 2.0.1, TDL-C NLOS (amended from CDL-C — SISO data has no antenna
  geometry), 3.5 GHz carrier, 300ns delay spread, fs=20MHz assumed, speed 3 m/s,
  AWGN 10dB first pass.
  NOTE: SNR label refers to the added AWGN stage only; RadioML frames carry native
  noise, so effective SNR is lower.
- Spectrogram: STFT n_fft=64, hop=16,fftshift applied, resize 224x224, per-sample standardized
- Metrics: top-1 accuracy, epsilon (delta=1e-5), attack-success rate,
  model size (MB, = MiB? state it), MACs via ptflops, bytes uploaded/client/round
