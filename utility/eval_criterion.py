import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# === CHANGED: helpers to normalize inputs to BCHW and numpy ===
def _to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _as_bchw(x):
    """
    Return x as numpy BCHW.

    Accepts:
      (C,H,W), (B,C,H,W),
      (H,W,C), (B,H,W,C),
      (B,C,T,H,W)  <-- NEW (collapse C*T -> C')
    """
    x = _to_numpy(x)
    if x.ndim == 3:
        # CHW or HWC
        if x.shape[-1] in (1, 3, 4, 8, 14, 16, 31, 32, 64):  # CHANGED
            H, W, C = x.shape  # CHANGED
            x = np.transpose(x, (2, 0, 1))[None, ...]  # CHANGED
        else:
            x = x[None, ...]  # assume CHW                               # CHANGED

    elif x.ndim == 4:
        # BCHW or BHWC
        if x.shape[-1] in (1, 3, 4, 8, 14, 16, 31, 32, 64) and x.shape[1] not in (
            1,
            3,
            4,
            8,
            14,
            16,
            31,
            32,
            64,
        ):  # CHANGED
            x = np.transpose(x, (0, 3, 1, 2))  # CHANGED
        # else assume already BCHW

    elif x.ndim == 5:  # CHANGED
        # Assume (B, C, T, H, W). Collapse (C,T) -> C' = C*T             # CHANGED
        B, C, T, H, W = x.shape  # CHANGED
        # sanity: last two must be spatial                                # CHANGED
        if H < 4 or W < 4:  # CHANGED
            raise ValueError(f"5D input expected (B,C,T,H,W). Got {x.shape}")  # CHANGED
        x = x.reshape(B, C * T, H, W)  # CHANGED

    else:
        raise ValueError(f"Expected 3D, 4D, or 5D tensor/array, got shape {x.shape}")
    return x


# === END CHANGED ===


class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        X = _as_bchw(X)  # CHANGED
        Y = _as_bchw(Y)  # CHANGED
        if X.shape != Y.shape:  # CHANGED
            raise ValueError(
                f"Pred/GT shapes differ after alignment: {X.shape} vs {Y.shape}"
            )  # CHANGED
        B, C, H, W = X.shape
        bwindex = []
        for b in range(B):
            for ch in range(C):
                x = X[b, ch]
                y = Y[b, ch]
                bwindex.append(self.index_fn(x, y))
        return bwindex


def _psnr_band(x, y):
    return psnr(x, y, data_range=1)


def _ssim_band(x, y):
    return ssim(x, y, data_range=1)


cal_bwssim = Bandwise(_ssim_band)
cal_bwpsnr = Bandwise(_psnr_band)


def cal_sam(X, Y, eps=1e-8):
    # === CHANGED: reuse alignment; SAM over channel axis (which now includes spectral bands) ===
    X = _as_bchw(X)  # CHANGED
    Y = _as_bchw(Y)  # CHANGED
    if X.shape != Y.shape:  # CHANGED
        raise ValueError(
            f"Pred/GT shapes differ after alignment: {X.shape} vs {Y.shape}"
        )  # CHANGED
    # vectorized over batch
    # X,Y: (B,C,H,W) -> compute SAM per pixel using channel axis=1, then mean over (B,H,W)
    num = np.sum(X * Y, axis=1) + eps  # CHANGED
    den = (np.sqrt(np.sum(X**2, axis=1)) + eps) * (
        np.sqrt(np.sum(Y**2, axis=1)) + eps
    )  # CHANGED
    cos = np.clip(num / den, -1.0, 1.0)  # CHANGED
    return float(np.mean(np.arccos(cos)))  # CHANGED


def MSIQA(X, Y):
    psnr_mean = float(np.mean(cal_bwpsnr(X, Y)))
    ssim_mean = float(np.mean(cal_bwssim(X, Y)))
    sam = cal_sam(X, Y)
    return psnr_mean, ssim_mean, sam
