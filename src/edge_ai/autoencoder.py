from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ConvAutoencoder1D(nn.Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if decoded.shape[-1] != self.window_size:
            decoded = torch.nn.functional.interpolate(decoded, size=self.window_size, mode="linear", align_corners=False)
        return decoded


class AutoencoderDetector:
    def __init__(self, window_size: int, threshold_scale: float = 3.0, min_threshold: float = 0.01) -> None:
        self.window_size = window_size
        self.threshold_scale = threshold_scale
        self.min_threshold = min_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvAutoencoder1D(window_size=window_size).to(self.device)
        self.threshold = 0.12
        self._trained = False

    def fit(self, normal_windows: np.ndarray, epochs: int = 6, batch_size: int = 32) -> float:
        if normal_windows.ndim != 2:
            raise ValueError("normal_windows must have shape (n_windows, window_size)")
        if normal_windows.shape[1] != self.window_size:
            raise ValueError("window size mismatch between detector and training windows")

        tensor = torch.tensor(normal_windows, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        self.model.train()

        for _ in range(epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                reconstructed = self.model(batch)
                loss = loss_fn(reconstructed, batch)
                loss.backward()
                optimizer.step()

        errors = np.array([self.reconstruction_error(window) for window in normal_windows], dtype=np.float32)
        self.threshold = float(np.mean(errors) + self.threshold_scale * np.std(errors))
        self.threshold = max(self.threshold, self.min_threshold)
        self._trained = True
        return self.threshold

    def reconstruction_error(self, window: np.ndarray) -> float:
        window = np.asarray(window, dtype=np.float32)
        tensor = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor)
            mse = torch.mean((reconstructed - tensor) ** 2)
        return float(mse.item())

    def score(self, window: np.ndarray) -> tuple[float, bool]:
        error = self.reconstruction_error(window)
        return error, error >= self.threshold

    @property
    def is_trained(self) -> bool:
        return self._trained
