import argparse
import csv
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_CHECKPOINTS = {
    # "hcanet_v1": {
    #     "arch": "hcanet",
    #     "path": "/home/esen/hsi-denoising/icvl-pretrained-for-ip/hcanet_v1/complex/model_epoch_100_467951.pth",
    #     "label": "HCANet-v1",
    # },
    # "hsdssm++": {
    #     "arch": "hssm",
    #     "path": "/home/esen/hsi-denoising/icvl-pretrained-for-ip/hsdssm++/complex/model_epoch_100_159904.pth",
    #     "label": "HSDSSM++",
    # },
    "hsdssm": {
        "arch": "hssm",
        "path": "/home/esen/hsi-denoising/TRQ3DNet/checkpoints_old/hssms/complex_icvl/model_epoch_100_159904.pth",
        "label": "HSDSSM",
    },
    "lamamba": {
        "arch": "lamamba",
        "path": "/home/esen/hsi-denoising/icvl-pretrained-for-ip/lamamba/model_epoch_82_495257.pth",
        "label": "LaMamba",
    },
    # "ssumamba": {
    #     "arch": "ssumamba",
    #     "path": "/home/esen/hsi-denoising/icvl-pretrained-for-ip/ssumamba/model_epoch_83_496914.pth",
    #     "label": "SSUMamba",
    # },
    # "trq3d": {
    #     "arch": "trq3d",
    #     "path": "checkpoints/trq3dnet/model_epoch_100_159904.pth",
    #     "label": "TRQ3D",
    # },

}


def load_indian_pines(data_dir):
    data_dir = Path(data_dir)
    cube = np.load(data_dir / "indianpinearray.npy")
    labels = np.load(data_dir / "IPgt.npy")
    if cube.ndim != 3:
        raise ValueError(f"Expected a HxWxB cube, got shape {cube.shape}")
    if labels.shape != cube.shape[:2]:
        raise ValueError(
            f"Label shape {labels.shape} does not match cube spatial shape {cube.shape[:2]}"
        )
    return cube, labels


def extract_labeled_pixels(cube, labels):
    mask = labels > 0
    return cube[mask].astype(np.float32), labels[mask].astype(np.int64)


def make_stratified_split(labels, train_fraction=0.10, seed=2018):
    indices = np.arange(labels.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=seed,
        stratify=labels,
    )
    return np.sort(train_idx), np.sort(test_idx)


def classification_report_row(
    name,
    train_x,
    train_y,
    test_x,
    test_y,
    c=100.0,
    gamma="scale",
):
    classifier = make_pipeline(
        StandardScaler(),
        SVC(C=c, gamma=gamma, kernel="rbf"),
    )
    classifier.fit(train_x, train_y)
    predictions = classifier.predict(test_x)
    return {
        "name": name,
        "oa": accuracy_score(test_y, predictions) * 100.0,
        "kappa": cohen_kappa_score(test_y, predictions) * 100.0,
    }


def evaluate_cube(name, cube, labels, train_idx, test_idx, c=100.0, gamma="scale"):
    pixels, targets = extract_labeled_pixels(cube, labels)
    return classification_report_row(
        name,
        pixels[train_idx],
        targets[train_idx],
        pixels[test_idx],
        targets[test_idx],
        c=c,
        gamma=gamma,
    )


def normalize_cube(cube):
    cube = cube.astype(np.float32)
    minimum = float(np.nanmin(cube))
    maximum = float(np.nanmax(cube))
    if maximum <= minimum:
        raise ValueError("Cannot normalize a constant-valued cube")
    return (cube - minimum) / (maximum - minimum)


def _pad_to_multiple(tensor, multiple):
    import torch.nn.functional as F

    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, height, width
    padded = F.pad(tensor, (0, pad_w, 0, pad_h, 0, 0), mode="reflect")
    return padded, height, width


def restore_cube_with_checkpoint(
    cube,
    arch,
    checkpoint_path,
    device="cuda",
    spectral_window=31,
    spectral_stride=31,
    spatial_multiple=32,
):
    import torch
    import models

    normalized = normalize_cube(cube)
    bands = normalized.shape[2]
    if bands < spectral_window:
        raise ValueError(
            f"Cube has {bands} bands, but spectral_window={spectral_window} was requested"
        )

    factory_name = arch.lower()
    if factory_name not in models.__dict__:
        available = ", ".join(sorted(k for k, v in models.__dict__.items() if callable(v)))
        raise RuntimeError(
            f"Model architecture '{arch}' is not available. "
            f"Check the import warnings above and the ssu_env dependencies. "
            f"Available callable model factories include: {available}"
        )

    net = models.__dict__[factory_name]()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    net.load_state_dict(checkpoint["net"])
    net.to(device)
    net.eval()

    tensor = torch.from_numpy(normalized.transpose(2, 0, 1))[None, None].to(device)
    output = torch.zeros_like(tensor)
    counts = torch.zeros_like(tensor)
    starts = list(range(0, bands - spectral_window + 1, spectral_stride))
    last_start = bands - spectral_window
    if starts[-1] != last_start:
        starts.append(last_start)

    with torch.no_grad():
        for start in starts:
            end = start + spectral_window
            chunk = tensor[:, :, start:end]
            chunk, height, width = _pad_to_multiple(chunk, spatial_multiple)
            restored = net(chunk)
            if isinstance(restored, (list, tuple)):
                restored = restored[0]
            restored = restored[..., :height, :width]
            output[:, :, start:end] += restored
            counts[:, :, start:end] += 1

    restored_cube = (output / counts)[0, 0].detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(restored_cube, 0.0, 1.0).astype(np.float32)


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "oa", "kappa"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "oa": f"{row['oa']:.4f}",
                    "kappa": f"{row['kappa']:.4f}",
                }
            )


def format_latex_table(rows):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Downstream classification results on Indian Pines.}",
        "\\begin{tabular}{lcc}",
        "\\hline",
        "Method & OA (\\%) & Kappa (\\%) \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(f"{row['name']} & {row['oa']:.2f} & {row['kappa']:.2f} \\\\")
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\label{tab:ip_downstream_classification}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_latex(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_latex_table(rows))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Indian Pines downstream classification after denoising."
    )
    parser.add_argument("--data-dir", default="data/IP")
    parser.add_argument("--output-dir", default="result/ip_downstream_classification")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--train-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--svm-c", type=float, default=100.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spectral-window", type=int, default=31)
    parser.add_argument("--spectral-stride", type=int, default=31)
    parser.add_argument("--spatial-multiple", type=int, default=32)
    parser.add_argument(
        "--reuse-restored",
        action="store_true",
        help="Load restored cubes from output-dir/restored instead of recomputing them.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only evaluate the raw Indian Pines cube.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    restored_dir = output_dir / "restored"
    cube, labels = load_indian_pines(args.data_dir)
    pixels, targets = extract_labeled_pixels(cube, labels)
    train_idx, test_idx = make_stratified_split(
        targets, train_fraction=args.train_fraction, seed=args.seed
    )

    rows = [
        classification_report_row(
            "Raw IP",
            pixels[train_idx],
            targets[train_idx],
            pixels[test_idx],
            targets[test_idx],
            c=args.svm_c,
            gamma=args.svm_gamma,
        )
    ]

    if not args.raw_only:
        restored_dir.mkdir(parents=True, exist_ok=True)
        for model_name in args.models:
            if model_name not in DEFAULT_CHECKPOINTS:
                valid = ", ".join(DEFAULT_CHECKPOINTS)
                raise ValueError(f"Unknown model '{model_name}'. Valid choices: {valid}")
            config = DEFAULT_CHECKPOINTS[model_name]
            restored_path = restored_dir / f"{model_name.replace('+', 'p')}.npy"
            if args.reuse_restored and restored_path.exists():
                restored_cube = np.load(restored_path)
            else:
                restored_cube = restore_cube_with_checkpoint(
                    cube,
                    config["arch"],
                    config["path"],
                    device=args.device,
                    spectral_window=args.spectral_window,
                    spectral_stride=args.spectral_stride,
                    spatial_multiple=args.spatial_multiple,
                )
                np.save(restored_path, restored_cube)

            rows.append(
                evaluate_cube(
                    config["label"],
                    restored_cube,
                    labels,
                    train_idx,
                    test_idx,
                    c=args.svm_c,
                    gamma=args.svm_gamma,
                )
            )

    write_csv(rows, output_dir / "ip_downstream_classification.csv")
    write_latex(rows, output_dir / "ip_downstream_classification.tex")
    for row in rows:
        print(f"{row['name']}: OA={row['oa']:.2f}, Kappa={row['kappa']:.2f}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
