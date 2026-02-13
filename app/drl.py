from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DRLDecision:
    """Normalized decision returned by a DRL policy."""

    action: str            # BUY | SELL | HOLD
    confidence: float      # 0..1
    frac: float | None = None  # optional position fraction suggestion


class DRLPolicy:
    """Lightweight PyTorch policy loader + inference wrapper.

    - Uses CPU by default.
    - Expects a TorchScript model OR a state_dict for a simple MLP.

    This is intentionally minimal so you can start integrating now.
    When you train, train to *match the observation vector* built by `build_obs()`.
    """

    def __init__(self, model_path: str, *, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._torch = None
        self._model = None

        if model_path:
            self._load(model_path)

    def _load(self, path: str) -> None:
        import torch  # local import so bot runs even if torch isn't installed yet

        self._torch = torch
        if not os.path.exists(path):
            raise FileNotFoundError(f"DRL model not found: {path}")

        # Prefer TorchScript for easiest deployment
        if path.endswith(".ts") or path.endswith(".torchscript"):
            self._model = torch.jit.load(path, map_location=self.device)
            self._model.eval()
            return

        # If it's a regular .pt/.pth, we assume it contains a TorchScript model.
        # (If you later want state_dict loading, we'll add a small MLP class.)
        obj = torch.load(path, map_location=self.device)
        if hasattr(obj, "eval"):
            self._model = obj
            self._model.eval()
            return

        raise ValueError(
            "Unsupported DRL model format. Use TorchScript (.ts) or save the whole model object in .pt/.pth."
        )

    def predict(self, obs: np.ndarray) -> DRLDecision:
        """Return BUY/SELL/HOLD + confidence.

        Convention:
          - model outputs logits for 3 actions: [HOLD, BUY, SELL]
        """
        if self._model is None:
            return DRLDecision("HOLD", 0.0)

        torch = self._torch
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits = self._model(x)

        # Support either (logits) or (logits, value) outputs
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())

        action = "HOLD" if idx == 0 else ("BUY" if idx == 1 else "SELL")
        return DRLDecision(action=action, confidence=conf)


def build_obs(
    df_1m,
    *,
    price: float,
    pos_qty: float,
    equity: float,
    window: int = 64,
) -> np.ndarray:
    """Build a simple, stable observation vector from your existing 1m candles.

    Obs design goals:
      - purely numeric
      - scale-stable (returns, z-scores)
      - small enough to run fast on CPU

    Current obs layout:
      [ r_1 ... r_window, z_price, pos_frac ]
    where r_i are log returns.

    IMPORTANT: Train your model to expect this *exact* layout.
    """
    import pandas as pd

    if df_1m is None or "close" not in df_1m.columns or len(df_1m) < (window + 2):
        # not enough history yet
        return np.zeros((window + 2,), dtype=np.float32)

    closes = df_1m["close"].astype(float).values
    closes = closes[-(window + 1) :]

    # log returns
    r = np.log(closes[1:] / closes[:-1] + 1e-12)
    r = r.astype(np.float32)

    # z-scored last price relative to window mean/std
    mu = float(np.mean(closes))
    sd = float(np.std(closes) + 1e-9)
    z_price = np.float32((price - mu) / sd)

    # position fraction of equity (spot long only)
    pos_value = float(pos_qty) * float(price)
    pos_frac = np.float32(pos_value / float(equity + 1e-9))

    obs = np.concatenate([r, np.array([z_price, pos_frac], dtype=np.float32)])
    return obs
