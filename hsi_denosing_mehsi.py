import argparse
import gc
import os
import warnings
from collections import OrderedDict
from functools import partial

import numpy as np
import tifffile
import torch
import torch.utils.data

from hsi_setup import Engine, train_options
from utility import adjust_learning_rate, display_learning_rate, seed_everywhere
from utility.dataset import HSI2Tensor


def _read_nonempty_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _scene_id_from_filename(scene_name):
    base = os.path.basename(scene_name)
    stem, _ = os.path.splitext(base)
    return stem


def _parse_pair_id(pair_id):
    parts = pair_id.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid MEHSI pair id '{pair_id}'. Expected format like '48_100'."
        )
    scene_id, exposure = parts
    if not scene_id.isdigit() or exposure not in {"20", "50", "100"}:
        raise ValueError(f"Invalid MEHSI pair id '{pair_id}'.")
    return scene_id, exposure


def _load_split_pairs(pair_file, scene_file=None):
    pair_ids = _read_nonempty_lines(pair_file)
    for pair_id in pair_ids:
        _parse_pair_id(pair_id)

    # Optional consistency check: scene file is not used to construct pairs.
    if scene_file:
        scene_ids = {
            _scene_id_from_filename(x) for x in _read_nonempty_lines(scene_file)
        }
        pair_scene_ids = {pair_id.split("_")[0] for pair_id in pair_ids}
        missing_from_pairs = sorted(scene_ids - pair_scene_ids)
        if missing_from_pairs:
            raise RuntimeError(
                f"Some scenes from {scene_file} are missing in {pair_file}: {missing_from_pairs}"
            )

    return pair_ids


def _resolve_hsi_path(folder, scene_id):
    tif_path = os.path.join(folder, f"{scene_id}.tif")
    tiff_path = os.path.join(folder, f"{scene_id}.tiff")
    if os.path.exists(tif_path):
        return tif_path
    if os.path.exists(tiff_path):
        return tiff_path
    raise FileNotFoundError(
        f"Missing file for scene {scene_id} in {folder} (.tif/.tiff)"
    )


def _sliding_positions(length, crop_size, stride):
    if crop_size > length:
        raise ValueError(
            f"crop_size={crop_size} cannot be larger than length={length}."
        )

    positions = list(range(0, length - crop_size + 1, stride))
    last = length - crop_size
    if positions[-1] != last:
        positions.append(last)
    return positions


class _TiffCache:
    def __init__(self, max_items=0):
        self.max_items = max_items
        self.cache = OrderedDict()

    def read(self, path, data_range):
        if self.max_items <= 0:
            return tifffile.imread(path).astype(np.float32) / data_range

        if path in self.cache:
            arr = self.cache.pop(path)
            self.cache[path] = arr
            return arr

        arr = tifffile.imread(path).astype(np.float32) / data_range
        self.cache[path] = arr
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return arr


class MEHSITrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        pair_ids,
        patch_size=128,
        stride=64,
        data_range=4095.0,
        transform=None,
        target_transform=None,
        cache_size=0,
    ):
        super().__init__()
        self.root = root
        self.pair_ids = pair_ids
        self.patch_size = patch_size
        self.stride = stride
        self.data_range = data_range
        self.transform = transform
        self.target_transform = target_transform
        self.cache = _TiffCache(max_items=cache_size)

        # MEHSI shape per README: C=34, H=690, W=512
        y_positions = _sliding_positions(690, patch_size, stride)
        x_positions = _sliding_positions(512, patch_size, stride)

        self.samples = []
        for pair_id in self.pair_ids:
            scene_id, exposure = _parse_pair_id(pair_id)
            input_path = _resolve_hsi_path(
                os.path.join(self.root, f"input{exposure}"), scene_id
            )
            gt_path = _resolve_hsi_path(os.path.join(self.root, "gt"), scene_id)

            for y in y_positions:
                for x in x_positions:
                    self.samples.append((input_path, gt_path, y, x))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, gt_path, y, x = self.samples[idx]
        p = self.patch_size

        noisy = self.cache.read(input_path, self.data_range)
        gt = self.cache.read(gt_path, self.data_range)

        noisy = noisy[:, y : y + p, x : x + p]
        gt = gt[:, y : y + p, x : x + p]

        if self.transform is not None:
            noisy = self.transform(noisy)
        if self.target_transform is not None:
            gt = self.target_transform(gt)

        return noisy, gt


class MEHSITestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        pair_ids,
        crop_size=512,
        data_range=4095.0,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.root = root
        self.pair_ids = pair_ids
        self.crop_size = crop_size
        self.data_range = data_range
        self.transform = transform
        self.target_transform = target_transform

        self.samples = []
        for pair_id in self.pair_ids:
            scene_id, exposure = _parse_pair_id(pair_id)
            input_path = _resolve_hsi_path(
                os.path.join(self.root, f"input{exposure}"), scene_id
            )
            gt_path = _resolve_hsi_path(os.path.join(self.root, "gt"), scene_id)

            self.samples.append((input_path, gt_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, gt_path = self.samples[idx]

        noisy = tifffile.imread(input_path).astype(np.float32) / self.data_range
        gt = tifffile.imread(gt_path).astype(np.float32) / self.data_range

        _, h, w = noisy.shape
        c = self.crop_size
        y = (h - c) // 2
        x = (w - c) // 2

        noisy = noisy[:, y : y + c, x : x + c]
        gt = gt[:, y : y + c, x : x + c]

        if self.transform is not None:
            noisy = self.transform(noisy)
        if self.target_transform is not None:
            gt = self.target_transform(gt)

        return noisy, gt


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="Hyperspectral Image Denoising on MEHSI"
    )
    parser.add_argument("--mehsi-root", type=str, default="./data/MEHSI")
    parser.add_argument("--train-pairs", type=str, default="./data/MEHSI/train.txt")
    parser.add_argument("--test-pairs", type=str, default="./data/MEHSI/test.txt")
    parser.add_argument("--train-scenes", type=str, default=None)
    parser.add_argument("--test-scenes", type=str, default=None)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-stride", type=int, default=64)
    parser.add_argument("--test-crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--cache-size", type=int, default=0)
    parser.add_argument("--data-range", type=float, default=4095.0)

    opt = train_options(parser)
    if opt.epochs <= 0:
        opt.epochs = 100

    print(f"opt settings: {opt}")

    seed_everywhere(opt.seed)

    engine = Engine(opt)
    print(
        "model params: %.2f M"
        % (sum([t.nelement() for t in engine.net.parameters()]) / 10**6)
    )

    HSI2TensorTransform = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)
    input_transform = HSI2TensorTransform()
    target_transform = HSI2TensorTransform()

    train_pairs = _load_split_pairs(opt.train_pairs, opt.train_scenes)
    test_pairs = _load_split_pairs(opt.test_pairs, opt.test_scenes)

    print(f"Training pairs: {len(train_pairs)}")
    print(f"Testing pairs: {len(test_pairs)}")

    train_dataset = MEHSITrainDataset(
        root=opt.mehsi_root,
        pair_ids=train_pairs,
        patch_size=opt.patch_size,
        stride=opt.patch_stride,
        data_range=opt.data_range,
        transform=input_transform,
        target_transform=target_transform,
        cache_size=opt.cache_size,
    )

    test_dataset = MEHSITestDataset(
        root=opt.mehsi_root,
        pair_ids=test_pairs,
        crop_size=opt.test_crop_size,
        data_range=opt.data_range,
        transform=input_transform,
        target_transform=target_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.threads,
        pin_memory=not opt.no_cuda,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.eval_batch_size,
        shuffle=False,
        num_workers=opt.threads,
        pin_memory=not opt.no_cuda,
    )

    print(f"Train patches: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    base_lr = opt.lr
    adjust_learning_rate(engine.optimizer, opt.lr)
    if opt.resume:
        if 40 < engine.epoch < 60:
            adjust_learning_rate(engine.optimizer, base_lr * 0.1)
        if 60 < engine.epoch < 70:
            adjust_learning_rate(engine.optimizer, base_lr)
        if 70 < engine.epoch < 90:
            adjust_learning_rate(engine.optimizer, base_lr * 0.1)
        if engine.epoch > 90:
            adjust_learning_rate(engine.optimizer, base_lr * 0.01)

    epoch_per_save = 1
    best_psnr = 0.0

    while engine.epoch < opt.epochs:
        if engine.epoch == 40:
            adjust_learning_rate(engine.optimizer, base_lr * 0.1)
        if engine.epoch == 60:
            adjust_learning_rate(engine.optimizer, base_lr)
        if engine.epoch == 70:
            adjust_learning_rate(engine.optimizer, base_lr * 0.1)
        if engine.epoch == 90:
            adjust_learning_rate(engine.optimizer, base_lr * 0.01)

        engine.train(train_loader)

        avg_psnr, avg_loss = engine.validate(test_loader, "MEHSI_test")
        print(f"Validation - PSNR: {avg_psnr:.4f}, Loss: {avg_loss:.4e}")

        gc.collect()
        torch.cuda.empty_cache()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            model_best_path = os.path.join(
                engine.basedir, engine.prefix, "model_best.pth"
            )
            engine.save_checkpoint(model_out_path=model_best_path)
            print(f"Best model saved. PSNR: {best_psnr:.4f}")

        model_latest_path = os.path.join(
            engine.basedir, engine.prefix, "model_latest.pth"
        )
        engine.save_checkpoint(model_out_path=model_latest_path)

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()
            metrics_path = os.path.join(engine.basedir, engine.prefix, "metrics.txt")
            with open(metrics_path, "a") as f:
                f.write(
                    f"Epoch: {engine.epoch}, PSNR: {avg_psnr:.4f}, Loss: {avg_loss:.4e}\n"
                )
