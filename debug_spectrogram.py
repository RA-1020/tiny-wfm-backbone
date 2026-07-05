import numpy as np
import torch
from dataset import RadioMLSpectrograms

ds = RadioMLSpectrograms("radioml_subset.npz")
iq = torch.complex(torch.from_numpy(ds.X[0, :, 0]), torch.from_numpy(ds.X[0, :, 1]))

spec = torch.stft(iq, n_fft=256, hop_length=128, window=torch.hann_window(256), return_complex=True)
print("STFT shape (freq_bins, time_frames):", spec.shape)   # <- this is the number I want to see

mag_db = 20 * torch.log10(spec.abs() + 1e-8)
print("mag_db min/max/mean:", mag_db.min().item(), mag_db.max().item(), mag_db.mean().item())
print("Per-frequency-bin mean energy (look for DC/Nyquist spikes):")
print(mag_db.mean(dim=1))   # mean over time, per frequency bin
