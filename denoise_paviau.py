"""Run a trained 31-band denoising model on PaviaU.mat and save a MAT result."""

import argparse
import os
import time

import numpy as np
import torch
from scipy.io import loadmat, savemat

from hsi_setup import Engine, train_options
from utility import seed_everywhere


PAVIA_SELECTED_BANDS = 31
NORMALIZE_EPS = 1e-8
SPATIAL_PAD_MULTIPLE = 32


def _first_hsi_key(mat):
    candidates = []
    for key, value in mat.items():
        if key.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if value.ndim == 3:
            candidates.append((key, value.shape, value.size))

    if not candidates:
        raise ValueError("No 3D hyperspectral array found in MAT file.")

    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[0][0]


def _as_hwc(cube, source_key):
    if cube.ndim != 3:
        raise ValueError(
            f"MAT key '{source_key}' must contain a 3D array, got shape {cube.shape}."
        )

    cube = np.asarray(cube, dtype=np.float32)

    # Most remote-sensing MAT files are HWC. If the first axis is clearly the
    # spectral axis, convert CHW to HWC.
    if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
        cube = np.transpose(cube, (1, 2, 0))

    return cube


def _select_uniform_bands(cube, band_count=PAVIA_SELECTED_BANDS):
    spectral_bands = cube.shape[2]
    if spectral_bands < band_count:
        raise ValueError(
            f"Need at least {band_count} spectral bands, got {spectral_bands}."
        )

    selected_bands = np.linspace(0, spectral_bands - 1, band_count).round().astype(int)
    return cube[:, :, selected_bands], selected_bands


def _normalize(cube):
    min_value = float(np.min(cube))
    max_value = float(np.max(cube))
    data_range = max_value - min_value
    return (cube - min_value) / (data_range + NORMALIZE_EPS), min_value, max_value


def _to_model_tensor(cube_hwc, use_2dconv):
    cube_chw = np.transpose(cube_hwc, (2, 0, 1)).astype(np.float32, copy=False)
    tensor = torch.from_numpy(cube_chw)
    if use_2dconv:
        return tensor.unsqueeze(0)
    return tensor.unsqueeze(0).unsqueeze(0)


def _pad_spatial_cube(cube, multiple=SPATIAL_PAD_MULTIPLE, mode="reflect"):
    height, width = cube.shape[:2]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return cube, (height, width), (0, 0)

    pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    if mode == "constant":
        padded = np.pad(cube, pad_width, mode=mode, constant_values=0.0)
    else:
        np_mode = "edge" if mode == "replicate" else mode
        padded = np.pad(cube, pad_width, mode=np_mode)
    return padded, (height, width), (pad_h, pad_w)


def _crop_spatial_cube(cube, original_hw):
    height, width = original_hw
    return cube[:height, :width, :]


def _to_hwc(tensor, use_2dconv):
    output = tensor.detach().cpu()
    if use_2dconv:
        output = output[0]
    else:
        output = output[0, 0]
    return output.numpy().transpose((1, 2, 0))


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Denoise data/PAVIAU/PaviaU.mat with a trained 31-band checkpoint."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/PAVIAU/PaviaU.mat",
        help="Input PaviaU MAT file.",
    )
    parser.add_argument(
        "--mat-key",
        type=str,
        default=None,
        help="MAT variable to denoise. Defaults to the largest 3D array.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./result/paviau_denoised.mat",
        help="Output MAT file.",
    )
    parser.add_argument(
        "--save-normalized-only",
        action="store_true",
        help="Save denoised values in [0, 1] instead of restoring input scale.",
    )
    parser.add_argument(
        "--no-spectral-slice",
        action="store_true",
        help="Disable 31-band spectral slicing.",
    )
    parser.add_argument(
        "--pad-multiple",
        type=int,
        default=SPATIAL_PAD_MULTIPLE,
        help="Pad H/W to this multiple before inference, then crop back.",
    )
    parser.add_argument(
        "--pad-mode",
        type=str,
        default="reflect",
        choices=["reflect", "replicate", "constant"],
        help="Spatial padding mode used before model inference.",
    )
    return train_options(parser)


def main():
    opt = _build_parser()
    opt.no_log = True
    opt.no_ropt = True
    opt.resume = True
    opt.slice = not opt.no_spectral_slice

    if opt.resumePath is None:
        raise ValueError("Please pass a trained checkpoint with --resumePath/-rp.")

    print(opt)
    seed_everywhere(opt.seed)

    mat = loadmat(opt.input)
    mat_key = opt.mat_key or _first_hsi_key(mat)
    cube = _as_hwc(mat[mat_key], mat_key)
    original_shape = cube.shape
    cube, selected_bands = _select_uniform_bands(cube)
    cube_norm, min_value, max_value = _normalize(cube)
    cube_norm, inference_hw, spatial_pad = _pad_spatial_cube(
        cube_norm, multiple=opt.pad_multiple, mode=opt.pad_mode
    )

    engine = Engine(opt)
    engine.net.eval()

    inputs = _to_model_tensor(cube_norm, engine.get_net().use_2dconv)
    if not opt.no_cuda:
        inputs = inputs.to(engine.device)

    with torch.no_grad():
        start = time.time()
        outputs = engine.forward(inputs)
        elapsed = time.time() - start

    denoised_norm = np.clip(_to_hwc(outputs, engine.get_net().use_2dconv), 0.0, 1.0)
    denoised_norm = _crop_spatial_cube(denoised_norm, inference_hw)
    if opt.save_normalized_only:
        denoised = denoised_norm
    else:
        denoised = denoised_norm * (max_value - min_value) + min_value

    os.makedirs(os.path.dirname(os.path.abspath(opt.output)), exist_ok=True)
    savemat(
        opt.output,
        {
            "denoised": denoised.astype(np.float32),
            "denoised_normalized": denoised_norm.astype(np.float32),
            "source_key": mat_key,
            "source_shape": np.asarray(original_shape, dtype=np.int32),
            "selected_bands": selected_bands.astype(np.int32),
            "model_input_shape": np.asarray(cube_norm.shape, dtype=np.int32),
            "spatial_pad_hw": np.asarray(spatial_pad, dtype=np.int32),
            "pad_multiple": opt.pad_multiple,
            "pad_mode": opt.pad_mode,
            "source_min": min_value,
            "source_max": max_value,
            "arch": opt.arch,
            "checkpoint": opt.resumePath,
            "elapsed_seconds": elapsed,
        },
    )

    print(
        "Saved denoised PaviaU result to %s | key=%s | source_shape=%s | "
        "selected_bands=%s | pad_hw=%s | model_input_shape=%s | "
        "output_shape=%s | time=%.3fs"
        % (
            opt.output,
            mat_key,
            original_shape,
            selected_bands.tolist(),
            spatial_pad,
            cube_norm.shape,
            denoised.shape,
            elapsed,
        )
    )


if __name__ == "__main__":
    main()
