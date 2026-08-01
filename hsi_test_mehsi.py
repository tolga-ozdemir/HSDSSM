import argparse
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hsi_denosing_mehsi import MEHSITestDataset, _load_split_pairs
from hsi_setup import Engine, train_options
from utility import MSIQA, seed_everywhere
from utility.dataset import HSI2Tensor


MEHSI_EXPOSURES = ("20", "50", "100")


def _model_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _default_savedir(opt):
    prefix = os.path.basename(os.path.dirname(opt.resumePath))
    return os.path.join(
        "./result",
        opt.arch,
        prefix,
        "res_mehsi_" + _model_name(opt.resumePath),
    )


def _filter_pairs_by_exposure(pair_ids, exposure):
    suffix = "_" + str(exposure)
    pairs = [pair_id for pair_id in pair_ids if pair_id.endswith(suffix)]
    if not pairs:
        raise RuntimeError(f"No MEHSI test pairs found for exposure {exposure}.")
    return pairs


def evaluate_mehsi(engine, loader, name, savedir=None, filename=None, verbose=True):
    engine.net.eval()
    res_arr = np.zeros((len(loader), 5), dtype=np.float64)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if not engine.opt.no_cuda:
                inputs = inputs.to(engine.device)
                targets = targets.to(engine.device)

            time_start = time.time()
            outputs = engine.forward(inputs)
            time_cost = time.time() - time_start

            if outputs.shape != targets.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: outputs={tuple(outputs.shape)}, "
                    f"targets={tuple(targets.shape)}"
                )

            loss_data = engine.criterion(outputs, targets).item()
            psnr, ssim, sam = MSIQA(outputs, targets)
            res_arr[batch_idx] = [psnr, ssim, sam, loss_data, time_cost]

            if verbose:
                print(
                    "%s %d PSNR: %.4f | SSIM: %.4f | SAM: %.4f | "
                    "Loss: %.4e | Time: %.4f"
                    % (name, batch_idx, psnr, ssim, sam, loss_data, time_cost)
                )

    if savedir and filename:
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, filename)
        if path.endswith(".npy"):
            np.save(path, res_arr)
        elif path.endswith(".txt"):
            np.savetxt(path, res_arr)
        else:
            raise ValueError("Result filename must end with .npy or .txt")

    if verbose:
        avg = res_arr.mean(axis=0)
        print(
            "%s AVG PSNR: %.4f | SSIM: %.4f | SAM: %.4f | "
            "Loss: %.4e | Time: %.4f"
            % (name, avg[0], avg[1], avg[2], avg[3], avg[4])
        )

    return res_arr


def build_loader(opt, pair_ids, transform):
    dataset = MEHSITestDataset(
        root=opt.mehsi_root,
        pair_ids=pair_ids,
        crop_size=opt.test_crop_size,
        data_range=opt.data_range,
        transform=transform,
        target_transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.threads,
        pin_memory=not opt.no_cuda,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MEHSI test results for 20, 50, and 100 exposure inputs"
    )
    parser.add_argument("--mehsi-root", type=str, default="./data/MEHSI")
    parser.add_argument("--test-pairs", type=str, default="./data/MEHSI/test.txt")
    parser.add_argument("--test-scenes", type=str, default=None)
    parser.add_argument("--test-crop-size", type=int, default=512)
    parser.add_argument("--data-range", type=float, default=4095.0)
    parser.add_argument("--savedir", type=str, default=None)
    parser.add_argument(
        "--exposures",
        type=str,
        default="20,50,100",
        help="Comma-separated MEHSI exposures to evaluate.",
    )

    opt = train_options(parser)
    opt.no_log = True
    opt.no_ropt = True
    opt.resume = True

    if opt.resumePath is None:
        opt.resumePath = "./checkpoints/ssumamba/ssumamba/model_latest.pth"

    savedir = opt.savedir or _default_savedir(opt)

    print(opt)
    seed_everywhere(opt.seed)

    engine = Engine(opt)
    print(
        "model params: %.2f M"
        % (sum([t.nelement() for t in engine.net.parameters()]) / 10**6)
    )

    hsi_to_tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)()
    all_test_pairs = _load_split_pairs(opt.test_pairs, opt.test_scenes)
    exposures = [item.strip() for item in opt.exposures.split(",") if item.strip()]

    rows = []
    for exposure in exposures:
        if exposure not in MEHSI_EXPOSURES:
            raise ValueError(
                f"Invalid exposure '{exposure}'. Expected one of {MEHSI_EXPOSURES}."
            )

        pair_ids = _filter_pairs_by_exposure(all_test_pairs, exposure)
        loader = build_loader(opt, pair_ids, hsi_to_tensor)
        name = f"MEHSI_{exposure}"
        filename = f"{name}.npy"

        print(f"testing {name} with {len(pair_ids)} samples..................")
        res = evaluate_mehsi(
            engine,
            loader,
            name,
            savedir=savedir,
            filename=filename,
            verbose=True,
        )

        avg = res.mean(axis=0)
        rows.append(
            {
                "Exposure": exposure,
                "PSNR": avg[0],
                "SSIM": avg[1],
                "SAM": avg[2],
                "Loss": avg[3],
                "Time": avg[4],
            }
        )

    result_table = pd.DataFrame(rows).set_index("Exposure")
    os.makedirs(savedir, exist_ok=True)
    result_table.to_csv(os.path.join(savedir, "result_MEHSI.csv"))
    print(result_table)
