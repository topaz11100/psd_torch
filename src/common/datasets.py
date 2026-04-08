from __future__ import annotations

import gzip
from pathlib import Path
import os
import random
import pickle
import shutil
import struct
import tarfile
import time
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm

# Optional torchvision (MNIST/CIFAR10 can be simplified when torchvision is available).
# If torchvision is missing or fails to import (e.g., mismatched binary), we fall back to
# the minimal built-in loaders below.
try:  # pragma: no cover
    from torchvision import datasets as tv_datasets  # type: ignore
    from torchvision import transforms as tv_transforms  # type: ignore
    _HAS_TORCHVISION = True
    _TORCHVISION_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    tv_datasets = None
    tv_transforms = None
    _HAS_TORCHVISION = False
    _TORCHVISION_IMPORT_ERROR = _e

from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .fft_analysis import rfft_log_mag, bin_spectrum


# ----------------------------------------------------------------------------
# DataLoader seeding helpers
# ----------------------------------------------------------------------------

def _make_worker_init_fn(seed: int):
    """Initialize python/numpy/torch RNG per worker deterministically."""
    def _fn(worker_id: int):
        s = int(seed) + int(worker_id)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
    return _fn



# -----------------------------------------------------------------------------
# Download helpers (used for SHD/SSC; MNIST/CIFAR10 prefer torchvision when available)
# -----------------------------------------------------------------------------

def _download(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return
    tmp = dst_path + ".tmp"

    # Remove stale tmp to avoid confusing partial files.
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass

    tqdm.write(f"Downloading: {url} -> {dst_path}")

    # NOTE:
    # - We avoid urllib.request.urlretrieve() here because it has poor progress reporting
    #   and can appear to "hang" in nohup logs.
    # - urlopen(timeout=...) ensures we fail fast on offline clusters.
    req = urllib.request.Request(
        url,
        headers={
            # Some hosts may reject the default Python user-agent.
            "User-Agent": "Mozilla/5.0 (compatible; multi_base/1.0)"
        },
    )

    timeout_sec = 60
    chunk_size = 256 * 1024  # 256KB
    last_print = 0.0

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if (total is not None and str(total).isdigit()) else None

            # For interactive runs, use tqdm progress bar. For nohup logs, print
            # periodic progress lines to avoid excessive log spam.
            use_pbar = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
            pbar = None
            if use_pbar:
                pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024)

            downloaded = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if pbar is not None:
                        pbar.update(len(chunk))
                    else:
                        now = time.time()
                        if now - last_print >= 10:
                            if total_bytes:
                                pct = 100.0 * float(downloaded) / float(total_bytes)
                                tqdm.write(
                                    f"  ... {downloaded/1024/1024:.1f}MB / {total_bytes/1024/1024:.1f}MB ({pct:.1f}%)"
                                )
                            else:
                                tqdm.write(f"  ... {downloaded/1024/1024:.1f}MB")
                            last_print = now

            if pbar is not None:
                pbar.close()

        os.replace(tmp, dst_path)
    except Exception:
        # Clean up tmp to avoid future "resume" confusion.
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def _download_with_fallback(urls, dst_path: str) -> None:
    last_err = None
    for url in urls:
        try:
            _download(url, dst_path)
            return
        except Exception as e:
            last_err = e
            if os.path.exists(dst_path):
                return
    raise RuntimeError(f"Failed to download {dst_path}. Last error: {last_err}")


def _extract_tar_gz(tar_gz_path: str, dst_dir: str) -> None:
    marker = os.path.join(dst_dir, ".extracted")
    if os.path.exists(marker):
        return
    os.makedirs(dst_dir, exist_ok=True)
    tqdm.write(f"Extracting: {tar_gz_path} -> {dst_dir}")
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=dst_dir)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok")


# -----------------------------------------------------------------------------
# MNIST (IDX format)
# -----------------------------------------------------------------------------

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_URLS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def _read_idx_images(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file magic {magic} in {gz_path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def _read_idx_labels(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file magic {magic} in {gz_path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num)


class MNISTRaw(Dataset):
    def __init__(self, root: str, train: bool, download: bool = True):
        self.root = root
        self.train = bool(train)
        os.makedirs(self.root, exist_ok=True)

        if download:
            for key, fname in MNIST_FILES.items():
                dst = os.path.join(self.root, fname)
                if not os.path.exists(dst):
                    urls = [base + fname for base in MNIST_URLS]
                    _download_with_fallback(urls, dst)

        if self.train:
            img_path = os.path.join(self.root, MNIST_FILES["train_images"])
            lbl_path = os.path.join(self.root, MNIST_FILES["train_labels"])
        else:
            img_path = os.path.join(self.root, MNIST_FILES["test_images"])
            lbl_path = os.path.join(self.root, MNIST_FILES["test_labels"])

        self.images = _read_idx_images(img_path)  # (N,28,28) uint8
        self.labels = _read_idx_labels(lbl_path)  # (N,) uint8

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        x = self.images[idx].astype(np.float32) / 255.0  # (28,28)
        y = int(self.labels[idx])
        return torch.from_numpy(x).unsqueeze(0), y  # (1,28,28)


# -----------------------------------------------------------------------------
# CIFAR-10 (python pickle batches)
# -----------------------------------------------------------------------------

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _load_cifar_batch(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"]  # (N,3072)
    labels = d.get(b"labels", d.get(b"fine_labels"))
    x = data.reshape(-1, 3, 32, 32)
    y = np.array(labels, dtype=np.int64)
    return x, y


class CIFAR10Raw(Dataset):
    def __init__(self, root: str, train: bool, download: bool = True, normalize: bool = True, augment: bool = False):
        self.root = root
        self.train = bool(train)
        self.normalize = bool(normalize)
        self.augment = bool(augment)
        os.makedirs(self.root, exist_ok=True)

        tar_path = os.path.join(self.root, "cifar-10-python.tar.gz")
        extract_dir = os.path.join(self.root, "cifar-10-batches-py")
        if download and not os.path.exists(extract_dir):
            _download(CIFAR10_URL, tar_path)
            _extract_tar_gz(tar_path, self.root)

        if not os.path.exists(extract_dir):
            raise FileNotFoundError(
                f"CIFAR-10 not found in {extract_dir}. Set download=True or place extracted folder there."
            )

        if self.train:
            xs, ys = [], []
            for i in range(1, 6):
                x, y = _load_cifar_batch(os.path.join(extract_dir, f"data_batch_{i}"))
                xs.append(x)
                ys.append(y)
            self.images = np.concatenate(xs, axis=0)  # (50000,3,32,32)
            self.labels = np.concatenate(ys, axis=0)
        else:
            self.images, self.labels = _load_cifar_batch(os.path.join(extract_dir, "test_batch"))

        # Normalization constants (common CIFAR-10)
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _random_crop(self, x: torch.Tensor, pad: int = 4) -> torch.Tensor:
        # x: (3,32,32)
        if pad <= 0:
            return x
        x_p = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
        _, H, W = x_p.shape
        top = torch.randint(0, H - 32 + 1, (1,)).item()
        left = torch.randint(0, W - 32 + 1, (1,)).item()
        return x_p[:, top : top + 32, left : left + 32]

    def _random_hflip(self, x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        if torch.rand(()) < p:
            return torch.flip(x, dims=[2])
        return x

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.images[idx].astype(np.float32) / 255.0)  # (3,32,32)
        y = int(self.labels[idx])

        if self.train and self.augment:
            x = self._random_crop(x, pad=4)
            x = self._random_hflip(x, p=0.5)

        if self.normalize:
            x = (x - self.mean) / self.std

        return x, y


def _ensure_mnist_torchvision_layout(dataset_dir: str) -> None:
    """
    Torchvision's MNIST expects raw .gz files under: <dataset_dir>/raw/.

    This project historically stored MNIST *.gz directly under <dataset_dir> (legacy layout).
    To avoid re-downloading the same data, we migrate those files into <dataset_dir>/raw/
    if needed.
    """
    try:
        os.makedirs(dataset_dir, exist_ok=True)
        raw_dir = os.path.join(dataset_dir, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        for fname in MNIST_FILES.values():
            src = os.path.join(dataset_dir, fname)
            dst = os.path.join(raw_dir, fname)
            if os.path.exists(src) and (not os.path.exists(dst)):
                try:
                    os.replace(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                    os.remove(src)
    except Exception:
        # Best-effort migration only. If this fails, torchvision will still be able
        # to download when download=True.
        return


# -----------------------------------------------------------------------------
# Sequential wrappers
# -----------------------------------------------------------------------------


class SequentialMNIST(Dataset):
    """MNIST -> sequence of length 784 with input_dim=1.

    The current project policy keeps the original pixel range in ``[0, 1]`` and
    performs direct real-valued injection for 784 time steps. In other words,
    the 28 x 28 image is flattened along the time axis without mean/std
    normalization.
    """

    def __init__(self, root: str, train: bool, download: bool = True):
        # Prefer torchvision's dataset implementation when available.
        if _HAS_TORCHVISION and tv_datasets is not None and tv_transforms is not None:
            # Project layout uses dataset-specific folder: <data_root>/MNIST/
            tv_root = os.path.abspath(os.path.join(root, os.pardir))
            dataset_dir = os.path.join(tv_root, "MNIST")
            _ensure_mnist_torchvision_layout(dataset_dir)

            # Torchvision MNIST requires processed .pt files; if only raw .gz exists, we
            # enable download=True to trigger *processing* without re-downloading.
            processed_train = os.path.join(dataset_dir, "processed", "training.pt")
            processed_test = os.path.join(dataset_dir, "processed", "test.pt")
            processed_ok = os.path.exists(processed_train) and os.path.exists(processed_test)
            raw_ok = all(os.path.exists(os.path.join(dataset_dir, "raw", f)) for f in MNIST_FILES.values())
            tv_download = bool(download) or ((not processed_ok) and raw_ok)

            self.base = tv_datasets.MNIST(
                root=tv_root,
                train=bool(train),
                download=tv_download,
                # ``ToTensor`` converts uint8 pixels to float32 in ``[0, 1]``.
                # The sequential adapter then flattens the raster order into a
                # 784-step scalar sequence.
                transform=tv_transforms.Compose([
                    tv_transforms.ToTensor(),
                ]),
            )
        else:
            # Fallback: minimal pure-Python loader (no torchvision dependency).
            self.base = MNISTRaw(root=root, train=train, download=download)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]  # x: (1,28,28)
        # ``MNISTRaw`` already returns float32 in ``[0, 1]``. We keep that range
        # and inject one real-valued scalar per time step.
        x = x.to(torch.float32)
        x = x.view(1, -1).transpose(0, 1).contiguous()  # (784,1)
        return x, int(y)


class SequentialCIFAR10(Dataset):
    """
    CIFAR10 -> sequence.
    mode:
      - "parallel": T=1024, input_dim=3 (RGB vector per pixel)
      - "serial":   T=3072, input_dim=1 (R then G then B)
    """

    def __init__(self, root: str, train: bool, download: bool = True, mode: str = "parallel"):
        self.mode = mode
        # Prefer torchvision for CIFAR10 download/decoding/augmentation when available.
        if _HAS_TORCHVISION and tv_datasets is not None and tv_transforms is not None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            if bool(train):
                tfm = tv_transforms.Compose(
                    [
                        tv_transforms.RandomCrop(32, padding=4),
                        tv_transforms.RandomHorizontalFlip(),
                        tv_transforms.ToTensor(),
                        tv_transforms.Normalize(mean, std),
                    ]
                )
            else:
                tfm = tv_transforms.Compose(
                    [
                        tv_transforms.ToTensor(),
                        tv_transforms.Normalize(mean, std),
                    ]
                )
            self.base = tv_datasets.CIFAR10(
                root=root,
                train=bool(train),
                download=bool(download),
                transform=tfm,
            )
        else:
            # Fallback: minimal pure-Python loader (no torchvision dependency).
            if not _HAS_TORCHVISION and (_TORCHVISION_IMPORT_ERROR is not None):
                tqdm.write(
                    f"[WARN] torchvision import failed ({type(_TORCHVISION_IMPORT_ERROR).__name__}: "
                    f"{_TORCHVISION_IMPORT_ERROR}). Using built-in CIFAR10 loader."
                )
            self.base = CIFAR10Raw(root=root, train=train, download=download, normalize=True, augment=train)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]  # x: (3,32,32)
        if self.mode == "parallel":
            x = x.permute(1, 2, 0).contiguous().view(-1, 3)  # (1024,3)
        elif self.mode == "serial":
            x = x.contiguous().view(-1, 1)  # (3072,1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return x.to(torch.float32), int(y)


# -----------------------------------------------------------------------------
# SHD / SSC (h5)
# -----------------------------------------------------------------------------

class EventH5Dataset(Dataset):
    """
    SHD/SSC-style event dataset stored in an HDF5 file with groups:
      - spikes/times : vlen float (seconds)
      - spikes/units : vlen int
      - labels       : int

    ⚠️ Preprocessing equivalence (Origin/ verification)
    ---------------------------------------------------
    The default settings of this class are designed to be *semantically aligned* with the
    author-provided preprocessing scripts:

      - Origin/Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics/SHD/shd_generate_dataset.py
      - Origin/Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics/SSC/ssc_generate_dataset.py

    Concretely, when using:
      - max_time = 1.0
      - binning = "origin"        (time -> bin via i = ceil(t/dt) with i in [0, T-1])
      - channel_flip = True       (reverse channel order)
      - unit_indexing = "auto"    (infer 0-based vs 1-based unit IDs)
      - align_to_first_event = False
      - use_event_counts = False  (binary frames)

    ...the resulting (T, 700) binary frame sequence matches the intent of the Origin code
    while being vectorized (fast) and stable (no per-sample indexing heuristics).

    Parameters
    ----------
    h5_path:
        Path to *.h5.
    T:
        Number of time bins / steps.
    num_units:
        Number of input channels (SHD/SSC: 700).
    max_time:
        Window length in seconds. Origin preprocessing uses 1.0s.
    binning:
        - "origin": assign each event to the *earliest* bin i such that t <= i*dt.
                    This is equivalent to i = ceil(t/dt), and drops events with i >= T.
        - "floor":  standard left-closed binning i = floor(t/dt) with i in [0, T-1].
    unit_indexing:
        - "auto": infer whether raw unit IDs are 0-based (0..699) or 1-based (1..700)
                  by probing multiple samples at dataset construction time.
        - "0": force 0-based
        - "1": force 1-based
    channel_flip:
        If True, reverse channel order (Origin uses vector[700-vals] = 1).
    align_to_first_event:
        If True, shift times so that the first event in each sample starts at t=0.
        (Origin code does NOT do this; keep False for equivalence.)
    use_event_counts:
        If True, accumulate counts per (bin, channel). Origin uses binary frames, so
        keep False for equivalence.
    """

    _ALL_KEYS = ("dendrite_input", "dendrite_state", "soma_input", "soma_state", "output")

    # Cache inferred unit offsets per file (avoid repeated probing).
    _UNIT_OFFSET_CACHE = {}

    def __init__(
        self,
        h5_path: str,
        T: int = 250,
        num_units: int = 700,
        *,
        max_time: float = 1.0,
        binning: str = "origin",
        unit_indexing: str = "auto",
        channel_flip: bool = True,
        align_to_first_event: bool = False,
        use_event_counts: bool = False,
        probe_units: int = 2048,
    ):
        self.h5_path = str(h5_path)
        self.T = int(T)
        self.num_units = int(num_units)

        self.max_time = float(max_time)
        if self.T <= 0:
            raise ValueError(f"T must be >= 1, got {self.T}")
        if self.num_units <= 0:
            raise ValueError(f"num_units must be >= 1, got {self.num_units}")
        if not (self.max_time > 0.0):
            raise ValueError(f"max_time must be > 0, got {self.max_time}")

        self.dt = self.max_time / float(self.T)

        self.binning = str(binning).lower().strip()
        if self.binning not in ("origin", "floor"):
            raise ValueError(f"Unsupported binning={binning!r}. Use 'origin' or 'floor'.")

        self.channel_flip = bool(channel_flip)
        self.align_to_first_event = bool(align_to_first_event)
        self.use_event_counts = bool(use_event_counts)

        unit_indexing = str(unit_indexing).lower().strip()
        if unit_indexing in ("auto", "a"):
            self.unit_offset = self._infer_unit_offset(self.h5_path, self.num_units, probe=int(probe_units))
        elif unit_indexing in ("0", "0-based", "0based", "zero"):
            self.unit_offset = 0
        elif unit_indexing in ("1", "1-based", "1based", "one"):
            self.unit_offset = 1
        else:
            raise ValueError(f"Unsupported unit_indexing={unit_indexing!r}. Use 'auto', '0', or '1'.")

        # Lazy-open HDF5 handle per worker/process.
        self._h5 = None
        self._len = None

    # ------------------------------------------------------------------
    # HDF5 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_unit_offset(h5_path: str, num_units: int, probe: int = 2048) -> int:
        """
        Infer whether unit IDs are 0-based (0..num_units-1) or 1-based (1..num_units).

        We avoid per-sample heuristics (e.g., `min==1`) because they can silently break
        channel consistency across samples (catastrophic for learning).
        """
        key = (os.path.abspath(h5_path), int(num_units))
        if key in EventH5Dataset._UNIT_OFFSET_CACHE:
            return int(EventH5Dataset._UNIT_OFFSET_CACHE[key])

        try:
            import h5py  # type: ignore
        except Exception:
            # Fall back to safest assumption (0-based).
            EventH5Dataset._UNIT_OFFSET_CACHE[key] = 0
            return 0

        if not os.path.exists(h5_path):
            EventH5Dataset._UNIT_OFFSET_CACHE[key] = 0
            return 0

        saw_zero = False
        saw_num_units = False
        global_min = None
        global_max = None

        with h5py.File(h5_path, "r") as f:
            u_ds = f["spikes"]["units"]
            n = int(u_ds.shape[0])
            if n <= 0:
                EventH5Dataset._UNIT_OFFSET_CACHE[key] = 0
                return 0
            k = int(min(max(1, probe), n))
            # Evenly spaced probes -> more robust than first-k only.
            idxs = np.linspace(0, n - 1, num=k, dtype=np.int64)
            for i in idxs:
                u = np.asarray(u_ds[int(i)], dtype=np.int64)
                if u.size == 0:
                    continue
                if (u == 0).any():
                    saw_zero = True
                if (u == num_units).any():
                    saw_num_units = True
                mi = int(u.min())
                ma = int(u.max())
                global_min = mi if global_min is None else min(global_min, mi)
                global_max = ma if global_max is None else max(global_max, ma)

        # Strong signals.
        if saw_num_units:
            EventH5Dataset._UNIT_OFFSET_CACHE[key] = 1
            return 1
        if saw_zero:
            EventH5Dataset._UNIT_OFFSET_CACHE[key] = 0
            return 0

        # Weak signal: if everything we saw is within [1, num_units], treat as 1-based.
        # (For typical SHD/SSC downloads, seeing a 0 across ~2k samples is very likely if 0-based.)
        if global_min is not None and global_max is not None:
            if global_min >= 1 and global_max <= num_units:
                EventH5Dataset._UNIT_OFFSET_CACHE[key] = 1
                return 1

        EventH5Dataset._UNIT_OFFSET_CACHE[key] = 0
        return 0

    def _ensure_open(self):
        if self._h5 is None:
            import h5py  # type: ignore

            self._h5 = h5py.File(self.h5_path, "r")
            self._times = self._h5["spikes"]["times"]
            self._units = self._h5["spikes"]["units"]
            self._labels = self._h5["labels"]
            self._len = int(self._labels.shape[0])

    def __len__(self) -> int:
        if self._len is None:
            self._ensure_open()
        return int(self._len)

    def __getitem__(self, idx: int):
        self._ensure_open()
        times = np.asarray(self._times[int(idx)], dtype=np.float32)  # seconds
        units = np.asarray(self._units[int(idx)], dtype=np.int64)
        label = int(np.asarray(self._labels[int(idx)]).item())

        x = np.zeros((self.T, self.num_units), dtype=np.float32)

        if times.size == 0 or units.size == 0:
            return torch.from_numpy(x), label

        # Optional alignment (NOT used in Origin preprocessing).
        if self.align_to_first_event:
            t0 = float(times.min())
            times = times - t0

        # Convert units to 0-based indexing using the inferred offset.
        units0 = units.astype(np.int64) - int(self.unit_offset)

        # Validity mask: units within [0, num_units-1]
        m_u = (units0 >= 0) & (units0 < self.num_units)
        if not np.any(m_u):
            return torch.from_numpy(x), label
        units0 = units0[m_u]
        times = times[m_u]

        # Clamp negative times to 0 for origin-style thresholding.
        t = np.maximum(times.astype(np.float32), 0.0)

        # Origin-style time binning: i = ceil(t/dt), i in [0, T-1]
        if self.binning == "origin":
            bin_idx = np.ceil(t / float(self.dt)).astype(np.int64)
        else:  # "floor"
            bin_idx = np.floor(t / float(self.dt)).astype(np.int64)

        m_t = (bin_idx >= 0) & (bin_idx < self.T)
        if not np.any(m_t):
            return torch.from_numpy(x), label
        bin_idx = bin_idx[m_t]
        units0 = units0[m_t]

        # Channel mapping (Origin uses reversed index).
        if self.channel_flip:
            ch = (self.num_units - 1) - units0
        else:
            ch = units0

        # Fill dense frame tensor.
        if self.use_event_counts:
            np.add.at(x, (bin_idx, ch), 1.0)
        else:
            x[bin_idx, ch] = 1.0

        return torch.from_numpy(x), label


def ensure_shd_ssc_files(data_root: str, dataset: str, download: bool = True) -> Tuple[str, str, Optional[str]]:
    """
    Ensure dataset files exist in data_root/<dataset>/.
    Returns (train_path, test_path, valid_path).
    """
    dataset = dataset.upper()
    ddir = os.path.join(data_root, dataset)
    os.makedirs(ddir, exist_ok=True)

    if dataset == "SHD":
        train_h5 = os.path.join(ddir, "shd_train.h5")
        test_h5 = os.path.join(ddir, "shd_test.h5")
        valid_h5 = None

        if download and (not os.path.exists(train_h5) or not os.path.exists(test_h5)):
            base = "https://zenkelab.org/datasets/"
            train_gz = os.path.join(ddir, "shd_train.h5.gz")
            test_gz = os.path.join(ddir, "shd_test.h5.gz")
            try:
                if not os.path.exists(train_h5):
                    _download(base + "shd_train.h5.gz", train_gz)
                    _gunzip(train_gz, train_h5)
                if not os.path.exists(test_h5):
                    _download(base + "shd_test.h5.gz", test_gz)
                    _gunzip(test_gz, test_h5)
            except Exception as e:
                tqdm.write(f"[WARN] Automatic download failed: {e}")
                tqdm.write(f"Please download SHD files manually into: {ddir}")
        return train_h5, test_h5, valid_h5

    if dataset == "SSC":
        train_h5 = os.path.join(ddir, "ssc_train.h5")
        test_h5 = os.path.join(ddir, "ssc_test.h5")
        valid_h5 = os.path.join(ddir, "ssc_valid.h5")

        if download and (not os.path.exists(train_h5) or not os.path.exists(test_h5)):
            base = "https://zenkelab.org/datasets/"
            train_gz = os.path.join(ddir, "ssc_train.h5.gz")
            test_gz = os.path.join(ddir, "ssc_test.h5.gz")
            valid_gz = os.path.join(ddir, "ssc_valid.h5.gz")
            try:
                if not os.path.exists(train_h5):
                    _download(base + "ssc_train.h5.gz", train_gz)
                    _gunzip(train_gz, train_h5)
                if not os.path.exists(test_h5):
                    _download(base + "ssc_test.h5.gz", test_gz)
                    _gunzip(test_gz, test_h5)
                if not os.path.exists(valid_h5):
                    _download(base + "ssc_valid.h5.gz", valid_gz)
                    _gunzip(valid_gz, valid_h5)
            except Exception as e:
                tqdm.write(f"[WARN] Automatic download failed: {e}")
                tqdm.write(f"Please download SSC files manually into: {ddir}")
        return train_h5, test_h5, valid_h5

    raise ValueError(f"Unknown dataset: {dataset}")


def _gunzip(src_gz: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return
    tmp = dst_path + ".tmp"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    with gzip.open(src_gz, "rb") as f_in, open(tmp, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.replace(tmp, dst_path)



# -----------------------------------------------------------------------------
# DataLoader helpers
# -----------------------------------------------------------------------------

def get_smnist_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    seed: Optional[int] = None,
):
    root = os.path.join(data_root, "MNIST")
    train_ds = SequentialMNIST(root=root, train=True, download=download)
    test_ds = SequentialMNIST(root=root, train=False, download=download)

    g_train = None
    g_test = None
    worker_init_fn = None
    if seed is not None:
        g_train = torch.Generator()
        g_train.manual_seed(int(seed))
        g_test = torch.Generator()
        g_test.manual_seed(int(seed) + 1)
        worker_init_fn = _make_worker_init_fn(int(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_train, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_test, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    return train_loader, test_loader, 10, 1, 784


def get_scifar10_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    mode: str = "parallel",
    seed: Optional[int] = None,
):
    root = os.path.join(data_root, "CIFAR10")
    train_ds = SequentialCIFAR10(root=root, train=True, download=download, mode=mode)
    test_ds = SequentialCIFAR10(root=root, train=False, download=download, mode=mode)

    g_train = None
    g_test = None
    worker_init_fn = None
    if seed is not None:
        g_train = torch.Generator()
        g_train.manual_seed(int(seed))
        g_test = torch.Generator()
        g_test.manual_seed(int(seed) + 1)
        worker_init_fn = _make_worker_init_fn(int(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_train, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_test, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    input_dim = 3 if mode == "parallel" else 1
    T = 1024 if mode == "parallel" else 3072
    return train_loader, test_loader, 10, input_dim, T


def get_shd_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    T: int = 250,
    seed: Optional[int] = None,
    *,
    max_time: float = 1.0,
    binning: str = "origin",
    unit_indexing: str = "auto",
    channel_flip: bool = True,
    align_to_first_event: bool = False,
    use_event_counts: bool = False,
):
    train_h5, test_h5, _ = ensure_shd_ssc_files(data_root, "SHD", download=download)
    train_ds = EventH5Dataset(
        train_h5,
        T=T,
        num_units=700,
        max_time=float(max_time),
        binning=str(binning),
        unit_indexing=str(unit_indexing),
        channel_flip=bool(channel_flip),
        align_to_first_event=bool(align_to_first_event),
        use_event_counts=bool(use_event_counts),
    )
    test_ds = EventH5Dataset(
        test_h5,
        T=T,
        num_units=700,
        max_time=float(max_time),
        binning=str(binning),
        unit_indexing=str(unit_indexing),
        channel_flip=bool(channel_flip),
        align_to_first_event=bool(align_to_first_event),
        use_event_counts=bool(use_event_counts),
    )

    g_train = None
    g_test = None
    worker_init_fn = None
    if seed is not None:
        g_train = torch.Generator()
        g_train.manual_seed(int(seed))
        g_test = torch.Generator()
        g_test.manual_seed(int(seed) + 1)
        worker_init_fn = _make_worker_init_fn(int(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_train, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_test, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    return train_loader, test_loader, 20, 700, T


def get_ssc_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    T: int = 250,
    use_valid_as_test: bool = False,
    seed: Optional[int] = None,
):
    train_h5, test_h5, valid_h5 = ensure_shd_ssc_files(data_root, "SSC", download=download)
    train_ds = EventH5Dataset(train_h5, T=T, num_units=700, max_time=1.0, binning="origin", unit_indexing="auto", channel_flip=True, align_to_first_event=False, use_event_counts=False)
    if use_valid_as_test and valid_h5 is not None and os.path.exists(valid_h5):
        test_ds = EventH5Dataset(valid_h5, T=T, num_units=700, max_time=1.0, binning="origin", unit_indexing="auto", channel_flip=True, align_to_first_event=False, use_event_counts=False)
    else:
        test_ds = EventH5Dataset(test_h5, T=T, num_units=700, max_time=1.0, binning="origin", unit_indexing="auto", channel_flip=True, align_to_first_event=False, use_event_counts=False)

    g_train = None
    g_test = None
    worker_init_fn = None
    if seed is not None:
        g_train = torch.Generator()
        g_train.manual_seed(int(seed))
        g_test = torch.Generator()
        g_test.manual_seed(int(seed) + 1)
        worker_init_fn = _make_worker_init_fn(int(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_train, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_test, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    return train_loader, test_loader, 35, 700, T


# -----------------------------------------------------------------------------
# Optional visualization helpers for PSD-related debugging
# -----------------------------------------------------------------------------

def visualize_input_sequence(
    dataset: str,
    x_seq: torch.Tensor,
    out_dir: str,
    fft_band_edges=None,
    fft_band_reduce: str = "mean",
    title_prefix: str = "",
) -> None:
    """
    Save:
      - image.png: spatial/raster visualization
      - image_fft.png: exact rFFT spectrum of an aggregated 1D signal
      - image_fft_band.png: binned version if fft_band_edges is not None
    """
    os.makedirs(out_dir, exist_ok=True)

    x = x_seq.detach().cpu().to(torch.float32)
    dataset_u = dataset.upper()

    # image.png
    if dataset_u in ("S-MNIST", "SMNIST", "MNIST"):
        img = x.view(28, 28).numpy()
        plt.figure(figsize=(3.2, 3.2))
        plt.imshow(img, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "image.png"), dpi=200)
        plt.close()
        agg = x.view(-1).numpy()
    elif dataset_u in ("S-CIFAR10", "SCIFAR10", "CIFAR10"):
        if x.shape[1] == 3 and x.shape[0] == 1024:
            img = x.view(32, 32, 3).numpy()
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            plt.figure(figsize=(3.4, 3.4))
            plt.imshow(img, interpolation="nearest")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "image.png"), dpi=200)
            plt.close()
            agg = x.mean(dim=1).numpy()
        else:
            agg = x.view(-1).numpy()
            plt.figure(figsize=(3.4, 2.2))
            plt.plot(agg, linewidth=1.0)
            plt.grid(True, which="both", alpha=0.28)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "image.png"), dpi=200)
            plt.close()
    else:
        # SHD / SSC: raster (time x channel)
        mat = x.numpy().T  # (C,T)
        plt.figure(figsize=(6.2, 3.2))
        plt.imshow(mat, aspect="auto", interpolation="nearest")
        plt.xlabel("t")
        plt.ylabel("unit")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "image.png"), dpi=200)
        plt.close()
        agg = x.mean(dim=1).numpy()

    # image_fft.png (normalized frequency axis: cycles/sample)
    from .fft_analysis import rfft_freqs, band_edges_to_bin_ranges

    freqs = rfft_freqs(len(agg), d=1.0)
    S = rfft_log_mag(agg, dim=-1)  # numpy
    plt.figure(figsize=(6.2, 3.2))
    plt.plot(freqs, S, linewidth=1.2)
    plt.xlabel("frequency (cycles/sample)")
    plt.ylabel("log(1+|rFFT|)")
    plt.grid(True, which="both", alpha=0.28)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "image_fft.png"), dpi=200)
    plt.close()

    if fft_band_edges is not None and len(fft_band_edges) > 0:
        ranges = band_edges_to_bin_ranges(len(agg), fft_band_edges, d=1.0)
        Sb = bin_spectrum(S, ranges, dim=-1, reduce=fft_band_reduce)
        centers = [(float(fft_band_edges[i]) + float(fft_band_edges[i + 1])) / 2.0 for i in range(len(fft_band_edges) - 1)]
        plt.figure(figsize=(6.2, 3.2))
        plt.plot(centers, Sb, linewidth=1.2)
        plt.xlabel("frequency band center (cycles/sample)")
        plt.ylabel(f"binned ({fft_band_reduce}) log(1+|rFFT|)")
        plt.grid(True, which="both", alpha=0.28)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "image_fft_band.png"), dpi=200)
        plt.close()

# ----------------------------------------------------------------------------
# Generic benchmark dataset registry
# ----------------------------------------------------------------------------


_DATASET_ALIASES: Dict[str, str] = {
    's-mnist': 's-mnist',
    'smnist': 's-mnist',
    'mnist': 's-mnist',
    'seqmnist': 's-mnist',
    'sequential_mnist': 's-mnist',
    'dvsgesture': 'dvsgesture',
    'dvs128gesture': 'dvsgesture',
    'dvs128_gesture': 'dvsgesture',
    'dvs-gesture': 'dvsgesture',
    'shd': 'shd',
    'deap': 'deap',
    'forda': 'forda',
    'ford_a': 'forda',
}


@dataclass(frozen=True)
class DatasetBundle:
    dataset_name: str
    train_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    input_dim: int
    T: int
    metadata: Mapping[str, Any]


def normalize_dataset_name(name: str) -> str:
    key = str(name).strip().lower().replace(' ', '').replace('/', '').replace('\\', '')
    if key not in _DATASET_ALIASES:
        raise ValueError(f'Unknown dataset name: {name}. Supported datasets: {sorted(set(_DATASET_ALIASES.values()))}')
    return _DATASET_ALIASES[key]


def dataset_choices() -> list[str]:
    return sorted(set(_DATASET_ALIASES.values()))


class DVSGestureHDF5Dataset(Dataset):
    """DVS128 Gesture loader aligned with the First-spike released preprocessing.

    Expects HDF5 files generated from the official raw `.aedat` + label CSV inputs,
    with per-sample groups containing `time`, `data`, and `labels`. The output sample
    is returned as a dense `(T_total, C)` tensor where `C = 2 * H * W`.
    """

    def __init__(
        self,
        h5_path: str,
        *,
        group: str,
        chunk_size: int = 120,
        empty_size: int = 40,
        dt_ms: float = 10.0,
        ds: int = 4,
        data_size: tuple[int, int] = (128, 128),
    ):
        self.h5_path = str(h5_path)
        self.group = str(group)
        self.chunk_size = int(chunk_size)
        self.empty_size = int(empty_size)
        self.dt_us = float(dt_ms) * 1000.0
        self.ds = int(ds)
        self.data_size = (int(data_size[0]), int(data_size[1]))
        self.height = self.data_size[0] // self.ds
        self.width = self.data_size[1] // self.ds
        self.total_T = int(self.chunk_size + self.empty_size)
        self._h5 = None
        self._grp = None
        self._keys = None

    def _ensure_open(self) -> None:
        if self._h5 is None:
            import h5py  # type: ignore
            self._h5 = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
            self._grp = self._h5[self.group]
            self._keys = sorted(self._grp.keys(), key=lambda x: int(x))

    def __len__(self) -> int:
        self._ensure_open()
        return int(len(self._keys))

    def _slice_events(self, times: np.ndarray, data: np.ndarray) -> np.ndarray:
        if times.size == 0 or data.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        start_time = float(times[0])
        end_time = start_time + float(self.chunk_size) * float(self.dt_us)
        start_idx = int(np.searchsorted(times, start_time, side='left'))
        end_idx = int(np.searchsorted(times, end_time, side='left'))
        t = times[start_idx:end_idx].astype(np.float32, copy=False)
        d = data[start_idx:end_idx]
        if t.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        t = t - float(t[0])
        # Stored data layout is (x, y, p). Convert to (t, p, x, y), matching the origin helper path.
        x = d[:, 0].astype(np.float32, copy=False)
        y = d[:, 1].astype(np.float32, copy=False)
        p = d[:, 2].astype(np.float32, copy=False)
        return np.stack([t, p, x, y], axis=1).astype(np.float32, copy=False)

    def _events_to_dense(self, events_tpxy: np.ndarray) -> np.ndarray:
        dense = np.zeros((self.total_T, 2 * self.height * self.width), dtype=np.float32)
        if events_tpxy.size == 0:
            return dense
        times = events_tpxy[:, 0]
        pol = np.clip(events_tpxy[:, 1].astype(np.int64), 0, 1)
        x = np.clip((events_tpxy[:, 2] // self.ds).astype(np.int64), 0, self.width - 1)
        y = np.clip((events_tpxy[:, 3] // self.ds).astype(np.int64), 0, self.height - 1)
        tbin = np.floor(times / self.dt_us).astype(np.int64)
        valid = (tbin >= 0) & (tbin < self.chunk_size)
        if not np.any(valid):
            return dense
        tbin = tbin[valid]
        pol = pol[valid]
        x = x[valid]
        y = y[valid]
        flat = pol * (self.height * self.width) + y * self.width + x
        dense[tbin, flat] = 1.0
        return dense

    def __getitem__(self, idx: int):
        self._ensure_open()
        key = self._keys[int(idx)]
        dset = self._grp[key]
        times = np.asarray(dset['time'][()], dtype=np.float32)
        data = np.asarray(dset['data'][()], dtype=np.float32)
        label = int(np.asarray(dset['labels'][()]).item())
        events = self._slice_events(times, data)
        dense = self._events_to_dense(events)
        return torch.from_numpy(dense), int(label)


class DEAPSegmentsDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.ndim != 3:
            raise ValueError(f'DEAP segment tensor must be (N,T,C), got {x.shape}')
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64).reshape(-1))

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int):
        return self.x[int(idx)], int(self.y[int(idx)].item())


class TSNormalizer:
    def __init__(self, norm_type: str = 'standardization', mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.norm_type = str(norm_type)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    def fit(self, arr: np.ndarray) -> 'TSNormalizer':
        if self.norm_type != 'standardization':
            raise ValueError(f'Unsupported norm_type: {self.norm_type}')
        flat = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0)
        self.std[self.std < np.finfo(float).eps] = 1.0
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError('Normalizer must be fit before transform')
        return ((np.asarray(arr, dtype=np.float64) - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)).astype(np.float32)


class FordATSFileDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.ndim != 3:
            raise ValueError(f'FordA tensor must be (N,T,C), got {x.shape}')
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64).reshape(-1))

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int):
        return self.x[int(idx)], int(self.y[int(idx)].item())


def _vectorized_worker_settings(seed: Optional[int]):
    g_train = None
    g_test = None
    worker_init_fn = None
    if seed is not None:
        g_train = torch.Generator()
        g_train.manual_seed(int(seed))
        g_test = torch.Generator()
        g_test.manual_seed(int(seed) + 1)
        worker_init_fn = _make_worker_init_fn(int(seed))
    return g_train, g_test, worker_init_fn


def _subject_filepaths_sorted(root: str) -> list[str]:
    files = []
    for name in sorted(os.listdir(root)):
        if name.lower().endswith('.dat'):
            files.append(os.path.join(root, name))
    if len(files) == 0:
        raise FileNotFoundError(f'No DEAP subject .dat files found under: {root}')
    return files


def _quantize_deap_labels(raw_labels: np.ndarray, *, axis: int, num_classes: int) -> np.ndarray:
    labels = np.asarray(raw_labels, dtype=np.float32)
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError(f'DEAP labels must be (N,4)-like, got {labels.shape}')
    if int(axis) not in (0, 1):
        raise ValueError(f'DEAP label axis must be 0 (valence) or 1 (arousal), got {axis}')
    if int(num_classes) == 2:
        out = np.zeros(labels.shape[0], dtype=np.int64)
        out[(labels[:, int(axis)] > 5.0) & (labels[:, int(axis)] <= 9.0)] = 1
        return out
    if int(num_classes) == 3:
        out = np.zeros(labels.shape[0], dtype=np.int64)
        vals = labels[:, int(axis)]
        out[(vals >= 4.0) & (vals <= 6.0)] = 1
        out[(vals >= 7.0) & (vals <= 9.0)] = 2
        return out
    raise ValueError(f'DEAP num_classes must be 2 or 3, got {num_classes}')


def _deap_baseline_removed_segments(data: np.ndarray) -> np.ndarray:
    eeg = np.asarray(data, dtype=np.float32)[:, :32, :]
    baseline = eeg[:, :, :384].reshape(eeg.shape[0], 32, 3, 128).mean(axis=2)
    signal = eeg[:, :, 384:]
    baseline_tiled = np.tile(baseline, (1, 1, 60))
    signal = signal - baseline_tiled
    segments = signal.reshape(signal.shape[0], 32, 20, 384).transpose(0, 2, 1, 3).reshape(-1, 32, 384)
    return segments.transpose(0, 2, 1).astype(np.float32)


def _load_deap_subject_file(path: str, *, label_axis: int, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    with open(path, 'rb') as f:
        payload = pickle.load(f, encoding='latin1')
    data = np.asarray(payload['data'], dtype=np.float32)
    labels = np.asarray(payload['labels'], dtype=np.float32)
    q = _quantize_deap_labels(labels, axis=int(label_axis), num_classes=int(num_classes))
    seg = _deap_baseline_removed_segments(data)
    rep = np.repeat(q.reshape(-1, 1), 20, axis=1).reshape(-1)
    return seg, rep.astype(np.int64)


def _parse_ts_metadata(lines: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {'class_label': False}
    for line in lines:
        low = line.lower()
        if low.startswith('@classlabel'):
            parts = line.split()
            meta['class_label'] = len(parts) >= 2 and parts[1].lower() == 'true'
            if len(parts) > 2:
                meta['class_values'] = parts[2:]
        elif low.startswith('@univariate'):
            parts = line.split()
            meta['univariate'] = len(parts) >= 2 and parts[1].lower() == 'true'
        elif low.startswith('@timestamps'):
            parts = line.split()
            meta['timestamps'] = len(parts) >= 2 and parts[1].lower() == 'true'
    return meta


def _parse_ts_value_token(token: str) -> float:
    tok = token.strip()
    if tok in ('', '?', 'NaN', 'nan', 'NAN'):
        return float('nan')
    return float(tok)


def load_ts_classification_file(ts_path: str) -> tuple[list[np.ndarray], np.ndarray]:
    raw_lines = [line.strip() for line in Path(ts_path).read_text(encoding='utf-8').splitlines()]
    lines = [line for line in raw_lines if line and not line.startswith('#')]
    data_start = None
    for i, line in enumerate(lines):
        if line.lower() == '@data':
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f'@data marker not found in TS file: {ts_path}')
    meta = _parse_ts_metadata(lines[:data_start])
    series_list: list[np.ndarray] = []
    labels: list[str] = []
    for line in lines[data_start:]:
        if ':' in line:
            parts = line.split(':')
            dims = parts[:-1]
            label = parts[-1]
        else:
            if not bool(meta.get('class_label', False)):
                raise ValueError(f'Classification TS file is missing label separator: {ts_path}')
            raise ValueError(f'Unable to parse TS row with class label: {line[:120]}')
        if len(dims) != 1:
            raise ValueError(f'Only univariate TS files are supported here, got {len(dims)} dimensions in {ts_path}')
        values = np.asarray([_parse_ts_value_token(tok) for tok in dims[0].split(',')], dtype=np.float32)
        series_list.append(values)
        labels.append(label.strip())
    label_arr = np.asarray(labels)
    return series_list, label_arr


def _interpolate_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if not np.isnan(arr).any():
        return arr
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    idx = np.arange(arr.size, dtype=np.float32)
    interp = np.interp(idx, idx[valid], arr[valid])
    return np.asarray(interp, dtype=np.float32)


def _pad_front_to_length(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size >= int(length):
        return arr[:int(length)]
    padded = np.zeros(int(length), dtype=np.float32)
    padded[int(length) - arr.size:] = arr
    return padded


def get_dvsgesture_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: Optional[int] = None,
    *,
    chunk_size: int = 120,
    empty_size: int = 40,
    dt_ms: float = 10.0,
    ds: int = 4,
):
    root = os.path.join(data_root, 'DVS128Gesture', 'hdf5')
    train_h5 = os.path.join(root, 'DVS-Gesture-train10.hdf5')
    test_h5 = os.path.join(root, 'DVS-Gesture-test10.hdf5')
    if not os.path.exists(train_h5):
        raise FileNotFoundError(f'DVS128 Gesture train HDF5 not found: {train_h5}')
    if not os.path.exists(test_h5):
        raise FileNotFoundError(f'DVS128 Gesture test HDF5 not found: {test_h5}')
    train_ds = DVSGestureHDF5Dataset(train_h5, group='train', chunk_size=int(chunk_size), empty_size=int(empty_size), dt_ms=float(dt_ms), ds=int(ds))
    test_ds = DVSGestureHDF5Dataset(test_h5, group='test', chunk_size=int(chunk_size), empty_size=int(empty_size), dt_ms=float(dt_ms), ds=int(ds))
    g_train, g_test, worker_init_fn = _vectorized_worker_settings(seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, generator=g_train, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, generator=g_test, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    input_dim = 2 * (128 // int(ds)) * (128 // int(ds))
    T_total = int(chunk_size) + int(empty_size)
    return train_loader, test_loader, 10, input_dim, T_total


def get_deap_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: Optional[int] = None,
    *,
    label_axis: int = 0,
    num_classes: int = 3,
):
    root = os.path.join(data_root, 'DEAP', 'data_preprocessed_python')
    if not os.path.isdir(root):
        raise FileNotFoundError(f'DEAP preprocessed root not found: {root}')
    subject_paths = _subject_filepaths_sorted(root)
    train_x_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    test_x_parts: list[np.ndarray] = []
    test_y_parts: list[np.ndarray] = []
    for subject_path in subject_paths:
        seg, labels = _load_deap_subject_file(subject_path, label_axis=int(label_axis), num_classes=int(num_classes))
        idx = np.arange(labels.shape[0])
        train_idx, test_idx = train_test_split(idx, test_size=0.1, stratify=labels, shuffle=True, random_state=29)
        train_x_parts.append(seg[train_idx])
        train_y_parts.append(labels[train_idx])
        test_x_parts.append(seg[test_idx])
        test_y_parts.append(labels[test_idx])
    train_x = np.concatenate(train_x_parts, axis=0).astype(np.float32)
    train_y = np.concatenate(train_y_parts, axis=0).astype(np.int64)
    test_x = np.concatenate(test_x_parts, axis=0).astype(np.float32)
    test_y = np.concatenate(test_y_parts, axis=0).astype(np.int64)
    train_ds = DEAPSegmentsDataset(train_x, train_y)
    test_ds = DEAPSegmentsDataset(test_x, test_y)
    g_train, g_test, worker_init_fn = _vectorized_worker_settings(seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, generator=g_train, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, generator=g_test, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    return train_loader, test_loader, int(num_classes), 32, 384


def get_forda_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: Optional[int] = None,
):
    root = os.path.join(data_root, 'FordA')
    train_path = os.path.join(root, 'FordA_TRAIN.ts')
    test_path = os.path.join(root, 'FordA_TEST.ts')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f'FordA train file not found: {train_path}')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f'FordA test file not found: {test_path}')
    train_series, train_labels_raw = load_ts_classification_file(train_path)
    test_series, test_labels_raw = load_ts_classification_file(test_path)
    max_len = 0
    for seq in train_series + test_series:
        max_len = max(max_len, int(len(seq)))
    if max_len <= 0:
        raise ValueError('FordA sequences are empty')
    train_x = np.stack([_pad_front_to_length(_interpolate_1d(seq), max_len) for seq in train_series], axis=0)[..., None]
    test_x = np.stack([_pad_front_to_length(_interpolate_1d(seq), max_len) for seq in test_series], axis=0)[..., None]
    class_values = sorted(set(train_labels_raw.tolist()) | set(test_labels_raw.tolist()))
    class_to_idx = {value: idx for idx, value in enumerate(class_values)}
    train_y = np.asarray([class_to_idx[v] for v in train_labels_raw.tolist()], dtype=np.int64)
    test_y = np.asarray([class_to_idx[v] for v in test_labels_raw.tolist()], dtype=np.int64)
    normalizer = TSNormalizer('standardization').fit(train_x)
    train_x = normalizer.transform(train_x)
    test_x = normalizer.transform(test_x)
    train_ds = FordATSFileDataset(train_x, train_y)
    test_ds = FordATSFileDataset(test_x, test_y)
    g_train, g_test, worker_init_fn = _vectorized_worker_settings(seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, generator=g_train, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, generator=g_test, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    return train_loader, test_loader, len(class_values), 1, int(max_len)


def build_dataset_bundle(
    dataset_name: str,
    *,
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = False,
    seed: Optional[int] = None,
    shd_T: int = 250,
    shd_max_time: float = 1.0,
    shd_binning: str = 'origin',
    shd_unit_indexing: str = 'auto',
    shd_channel_flip: bool = True,
    shd_align_to_first_event: bool = False,
    shd_use_event_counts: bool = False,
    dvsgesture_chunk_size: int = 120,
    dvsgesture_empty_size: int = 40,
    dvsgesture_dt_ms: float = 10.0,
    dvsgesture_ds: int = 4,
    deap_label_axis: int = 0,
    deap_num_classes: int = 3,
) -> DatasetBundle:
    name = normalize_dataset_name(dataset_name)
    if name == 's-mnist':
        train_loader, test_loader, num_classes, input_dim, T = get_smnist_loaders(data_root, batch_size=batch_size, num_workers=num_workers, download=download, seed=seed)
        metadata = {'download': bool(download)}
    elif name == 'dvsgesture':
        train_loader, test_loader, num_classes, input_dim, T = get_dvsgesture_loaders(data_root, batch_size=batch_size, num_workers=num_workers, seed=seed, chunk_size=int(dvsgesture_chunk_size), empty_size=int(dvsgesture_empty_size), dt_ms=float(dvsgesture_dt_ms), ds=int(dvsgesture_ds))
        metadata = {
            'chunk_size': int(dvsgesture_chunk_size),
            'empty_size': int(dvsgesture_empty_size),
            'dt_ms': float(dvsgesture_dt_ms),
            'ds': int(dvsgesture_ds),
            'spatial_size_after_downsample': [128 // int(dvsgesture_ds), 128 // int(dvsgesture_ds)],
        }
    elif name == 'shd':
        train_loader, test_loader, num_classes, input_dim, T = get_shd_loaders(
            data_root, batch_size=batch_size, num_workers=num_workers, download=download, T=int(shd_T), seed=seed,
            max_time=float(shd_max_time), binning=str(shd_binning), unit_indexing=str(shd_unit_indexing), channel_flip=bool(shd_channel_flip),
            align_to_first_event=bool(shd_align_to_first_event), use_event_counts=bool(shd_use_event_counts),
        )
        metadata = {
            'download': bool(download),
            'max_time': float(shd_max_time),
            'binning': str(shd_binning),
            'unit_indexing': str(shd_unit_indexing),
            'channel_flip': bool(shd_channel_flip),
            'align_to_first_event': bool(shd_align_to_first_event),
            'use_event_counts': bool(shd_use_event_counts),
        }
    elif name == 'deap':
        train_loader, test_loader, num_classes, input_dim, T = get_deap_loaders(data_root, batch_size=batch_size, num_workers=num_workers, seed=seed, label_axis=int(deap_label_axis), num_classes=int(deap_num_classes))
        metadata = {'label_axis': int(deap_label_axis), 'num_classes': int(deap_num_classes)}
    elif name == 'forda':
        train_loader, test_loader, num_classes, input_dim, T = get_forda_loaders(data_root, batch_size=batch_size, num_workers=num_workers, seed=seed)
        metadata = {}
    else:
        raise AssertionError(f'Unhandled dataset: {name}')
    return DatasetBundle(
        dataset_name=name,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=int(num_classes),
        input_dim=int(input_dim),
        T=int(T),
        metadata=metadata,
    )
