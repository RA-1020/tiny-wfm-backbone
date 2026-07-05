"""RadioML 2018.01A subset -> spectrogram Dataset.
Pipeline: IQ (1024 complex samples) -> STFT -> log-magnitude -> normalize -> resize 224x224.
Contract params: n_fft=256, hop=128 (CONTRACT.md).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def iq_to_spectrogram(iq: torch.Tensor, n_fft: int = 64, hop: int = 16) -> torch.Tensor:
    """iq: complex64 tensor of shape [1024] -> [1, 224, 224] float32 spectrogram."""
    spec = torch.stft(iq, n_fft=n_fft, hop_length=hop,
                      window=torch.hann_window(n_fft),
                      return_complex=True)              # [freq_bins, time_frames]
    spec = torch.fft.fftshift(spec, dim=0)              # center 0 Hz — untangles the split-edge energy
    mag_db = 20 * torch.log10(spec.abs() + 1e-8)
    mag_db = (mag_db - mag_db.mean()) / (mag_db.std() + 1e-8)
    img = mag_db.unsqueeze(0)
    img = TF.resize(img, [224, 224], antialias=True)
    return img


class RadioMLSpectrograms(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = data["X"]                # (N, 1024, 2) float32 — I and Q as 2 real channels
        self.y = data["y"].astype(np.int64)
        self.classes = [str(c) for c in data["classes"]]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        iq = torch.complex(torch.from_numpy(self.X[i, :, 0]),
                           torch.from_numpy(self.X[i, :, 1]))   # -> complex64 [1024]
        return iq_to_spectrogram(iq), int(self.y[i])
