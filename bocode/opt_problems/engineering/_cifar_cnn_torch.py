"""Offline, TensorFlow-free port of the Casmopolitan (Wan et al., ICML 2021) CIFAR-10
sparse adversarial-attack victim model and data. Support module for
:mod:`bocode.opt_problems.engineering.CifarAttack` (not a benchmark problem itself).

This module reproduces, in pure PyTorch, the pieces of the Casmopolitan repo
(https://github.com/xingchenwan/Casmopolitan, MIT) needed to evaluate the
85-D mixed-integer sparse-adversarial-attack objective OFFLINE, without any
TensorFlow dependency:

  * The Carlini CIFAR CNN (setup_cifar.py::CIFARModel), ported from the Keras/TF1
    HDF5 weights to a torch.nn.Module (softmax output, ~78% test accuracy).
  * The CIFAR-10 test set, preprocessed exactly like setup_cifar.py::load_batch
    (uint8 -> img/255 - 0.5, HWC 32x32x3).
  * utilities.py::upsample_projection (verbatim semantics).
  * The FIXED, seed-stable attack image / true-label / target-label selection
    performed by bbattack_objective.CNN + utilities.generate_attack_data_set.

Public API (see individual docstrings):
    weights_path() -> str
    testbin_path() -> str
    load_cifar_model(device="cpu") -> torch.nn.Module
    load_cifar_test() -> (images (10000,32,32,3) float32 in [-0.5,0.5], labels (10000,) int)
    upsample_projection(dim_reduction, X_low, low_dim, high_dim, nchannel=1, align_corners=True)
    pick_fixed_attack_image(model, images, labels, img_offset=0, target_label_id=0) -> dict

The module imports with NO side effects (all heavy loading is lazy, inside the
functions) and never imports tensorflow.

--- Layout-port correctness notes (the parts that are easy to get wrong) ---
Keras Conv2D is CROSS-CORRELATION (same as torch Conv2d) with kernel layout
(kh, kw, in_ch, out_ch); torch wants (out_ch, in_ch, kh, kw), so we transpose
(3, 2, 0, 1). Keras default padding is 'valid'.

Keras Flatten on NHWC flattens in (H, W, C) order; torch Flatten on NCHW
flattens in (C, H, W) order. The first Dense layer's weight rows are therefore
permuted: reshape (H*W*C, out) -> (H, W, C, out) -> transpose to (C, H, W, out)
-> reshape (C*H*W, out). Here H=W=5, C=128 (3200 = 5*5*128).

Dense weight is Keras (in, out) -> torch Linear (out, in), so transpose.

--- Verification (measured 2026-08-07, CPU, torch 2.6.0, numpy 1.26.4) ---
  * Top-1 accuracy on the 10000 CIFAR-10 test images: 0.7797 (7797/10000),
    matching the repo's documented "78% accuracy" (setup_cifar.py::CIFAR).
  * Ablation: loading dense_1 WITHOUT the HWC->CHW row permutation collapses
    accuracy to 0.1178 (chance) -- the permutation is load-bearing.
  * conv2d_1 output matches an explicit NHWC Keras-semantics cross-correlation
    to 4.8e-07 max abs error (float32 roundoff), confirming the kernel layout.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..._fetch import fetch_data_file

# ----------------------------------------------------------------------------
# Asset resolution. Both assets are too large to ship in the wheel (9.2 MB of CNN
# weights, 30 MB of CIFAR-10 test images), so they follow the same download-on-demand
# path as the other heavy BoCoDe data (``bocode/_fetch.py``): an in-repo/local copy is
# used when present, otherwise the file is fetched once into the bocode cache.
# ``BOCODE_CASMO_ATTACK_DIR`` points at a directory holding local copies:
#     <dir>/cifar_keras_weights.h5
#     <dir>/cifar-10-batches-bin/test_batch.bin   (or <dir>/cifar10_test_batch.bin)
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DIR = os.environ.get(
    "BOCODE_CASMO_ATTACK_DIR", os.path.join(_HERE, "CasmoAttack_Data")
)

WEIGHTS_FILENAME = "cifar_keras_weights.h5"
TESTBIN_FILENAME = "cifar10_test_batch.bin"


def _local_candidates(*relpaths):
    for rel in relpaths:
        path = os.path.join(_LOCAL_DIR, rel)
        if os.path.exists(path):
            return path
    return None


def weights_path() -> str:
    """Path to the Keras/TF1 HDF5 weights of the Carlini CIFAR CNN (9.2 MB)."""
    return fetch_data_file(
        WEIGHTS_FILENAME, local_fallback=_local_candidates(WEIGHTS_FILENAME)
    )


def testbin_path() -> str:
    """Path to the CIFAR-10 binary test batch, ``test_batch.bin`` (30 MB)."""
    return fetch_data_file(
        TESTBIN_FILENAME,
        local_fallback=_local_candidates(
            TESTBIN_FILENAME, os.path.join("cifar-10-batches-bin", "test_batch.bin")
        ),
    )


# ----------------------------------------------------------------------------
# The ported Carlini CIFAR CNN.
# ----------------------------------------------------------------------------
class CIFARCNNTorch(nn.Module):
    """Carlini CIFAR CNN (setup_cifar.py::CIFARModel), pure PyTorch.

    Input convention (matching Casmopolitan's CNN.evaluate): NHWC float32 in
    [-0.5, 0.5], i.e. shape (N, 32, 32, 3). Output: softmax probabilities (N, 10).

    Architecture: Conv(64,3x3)-relu, Conv(64,3x3)-relu, MaxPool2,
    Conv(128,3x3)-relu, Conv(128,3x3)-relu, MaxPool2, Flatten,
    Dense(256)-relu, Dense(256)-relu, Dense(10), softmax. All convs 'valid'.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)  # valid padding (default 0)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5 * 5 * 128, 256)  # 3200 -> 256
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # x: NHWC in [-0.5, 0.5]. Convert to torch NCHW.
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        # Flatten in torch NCHW (C, H, W) order. fc1 weights were permuted at load
        # time to consume this ordering identically to Keras NHWC flatten.
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def _load_keras_weights_into(model: CIFARCNNTorch, h5_path: str):
    """Copy the Keras/TF1 HDF5 weights into the torch module with the correct
    axis permutations (see module docstring)."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Reading the Casmopolitan CIFAR CNN weights requires the optional "
            "'casmoattack' dependency (h5py). "
            "Install it with: pip install 'bocode[casmoattack]'"
        ) from exc

    with h5py.File(h5_path, "r") as f:
        mw = f["model_weights"]

        def get(layer, kind):
            return np.asarray(mw[layer][layer][f"{kind}:0"])

        # Convs: Keras (kh, kw, in, out) -> torch (out, in, kh, kw).
        for tconv, kname in [
            (model.conv1, "conv2d_1"),
            (model.conv2, "conv2d_2"),
            (model.conv3, "conv2d_3"),
            (model.conv4, "conv2d_4"),
        ]:
            k = get(kname, "kernel")  # (kh, kw, in, out)
            b = get(kname, "bias")  # (out,)
            k_t = np.transpose(k, (3, 2, 0, 1))  # (out, in, kh, kw)
            tconv.weight.data.copy_(torch.from_numpy(np.ascontiguousarray(k_t)))
            tconv.bias.data.copy_(torch.from_numpy(np.ascontiguousarray(b)))

        # dense_1: Keras flatten is NHWC (H, W, C); torch flatten is NCHW (C, H, W).
        # Reorder the (3200, 256) weight rows: (H,W,C,out) -> (C,H,W,out).
        H = W = 5
        C = 128
        k1 = get("dense_1", "kernel")  # (3200, 256), rows in (H, W, C) order
        b1 = get("dense_1", "bias")
        k1 = k1.reshape(H, W, C, 256)  # (H, W, C, out)
        k1 = np.transpose(k1, (2, 0, 1, 3))  # (C, H, W, out)
        k1 = k1.reshape(H * W * C, 256)  # rows now in (C, H, W) order
        # Keras Dense (in, out) -> torch Linear (out, in): transpose.
        model.fc1.weight.data.copy_(torch.from_numpy(np.ascontiguousarray(k1.T)))
        model.fc1.bias.data.copy_(torch.from_numpy(np.ascontiguousarray(b1)))

        for tfc, kname in [(model.fc2, "dense_2"), (model.fc3, "dense_3")]:
            k = get(kname, "kernel")  # (in, out)
            b = get(kname, "bias")
            tfc.weight.data.copy_(torch.from_numpy(np.ascontiguousarray(k.T)))
            tfc.bias.data.copy_(torch.from_numpy(np.ascontiguousarray(b)))


def load_cifar_model(device="cpu") -> torch.nn.Module:
    """Build the ported Carlini CIFAR CNN, load the HDF5 weights, put it in eval
    mode on `device`, and return it. Input: NHWC in [-0.5, 0.5]; output: softmax.
    """
    h5_path = weights_path()
    model = CIFARCNNTorch()
    _load_keras_weights_into(model, h5_path)
    model.eval()
    model.to(device)
    return model


# ----------------------------------------------------------------------------
# CIFAR-10 test set (preprocessed exactly like setup_cifar.py::load_batch).
# ----------------------------------------------------------------------------
def load_cifar_test():
    """Load the 10000 CIFAR-10 test images from the CIFAR-10 binary test batch.

    Preprocessing matches setup_cifar.py::load_batch exactly: each record is
    1 label byte + 3072 image bytes (CHW 3x32x32, uint8); image = (img/255) - 0.5
    with a (1, 2, 0) transpose to HWC.

    Returns:
        images: (10000, 32, 32, 3) float32 in [-0.5, 0.5]
        labels: (10000,) int  (0..9)
    """
    bin_path = testbin_path()
    with open(bin_path, "rb") as fh:
        raw = fh.read()
    size = 32 * 32 * 3 + 1
    n = len(raw) // size
    images = np.empty((n, 32, 32, 3), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int64)
    for i in range(n):
        arr = np.frombuffer(raw[i * size : (i + 1) * size], dtype=np.uint8)
        labels[i] = int(arr[0])
        img = arr[1:].reshape((3, 32, 32)).transpose((1, 2, 0))  # CHW -> HWC
        images[i] = (img.astype(np.float32) / 255.0) - 0.5
    return images, labels


# ----------------------------------------------------------------------------
# Upsampling projection -- copied from utilities.py (IDENTICAL semantics).
# ----------------------------------------------------------------------------
def upsample_projection(
    dim_reduction, X_low, low_dim, high_dim, nchannel=1, align_corners=True
):
    """Various upsampling methods: CLUSTER, BILI (bilinear), NN (nearest),
    BICU (bicubic). Verbatim port of Casmopolitan utilities.py::upsample_projection.

    :param dim_reduction: 'CLUSTER' | 'BILI' | 'NN' | 'BICU'
    :param X_low: input data in low dimension, shape (n, low_dim * nchannel)
    :param low_dim: the low dimension (e.g. 14*14 = 196)
    :param high_dim: the high dimension (e.g. 32*32 = 1024)
    :param nchannel: number of image channels
    :param align_corners: align-corner option for F.interpolate
    :return X_high: input data in high dimension, (n, high_dim * nchannel)
    """
    if dim_reduction == "CLUSTER":
        n = X_low.shape[0]
        high_int = int(np.sqrt(high_dim))
        low_int = int(np.sqrt(low_dim))
        ratio = np.floor(high_dim / low_dim)
        low_edge = int(np.floor(np.sqrt(ratio)) + 1)
        high_edge = low_edge * low_int
        high_obs = np.zeros((n, high_edge, high_edge, nchannel))
        for ch in range(nchannel):
            for row in range(low_int):
                for col in range(low_int):
                    k = row * low_int + col
                    for jj in range(low_edge):
                        for ii in range(low_edge):
                            high_obs[
                                :, row * low_edge + jj, col * low_edge + ii, ch
                            ] = X_low[:, ch * low_dim + k]
        high_obs = high_obs[:, :high_int, :high_int, :]
        X_high = high_obs.reshape(X_low.shape[0], high_dim * nchannel)
    else:
        if dim_reduction == "BILI":
            upsample_mode = "bilinear"
        elif dim_reduction == "NN":
            upsample_mode = "nearest"
            align_corners = None
        elif dim_reduction == "BICU":
            upsample_mode = "bicubic"
        X_low_tensor = torch.FloatTensor(X_low).view(
            X_low.shape[0], nchannel, int(np.sqrt(low_dim)), int(np.sqrt(low_dim))
        )
        X_high_tensor_resize = F.interpolate(
            X_low_tensor,
            size=(int(np.sqrt(high_dim)), int(np.sqrt(high_dim))),
            mode=upsample_mode,
            align_corners=align_corners,
        )
        X_high = (
            X_high_tensor_resize.data.numpy()
            .squeeze()
            .reshape(X_low.shape[0], high_dim * nchannel)
        )
    return X_high


# ----------------------------------------------------------------------------
# Fixed, seed-stable attack-instance selection.
# ----------------------------------------------------------------------------
def _predict_labels(model, images, batch_size=1000):
    """Argmax predicted class over `images` (NHWC in [-0.5,0.5]) using `model`."""
    device = next(model.parameters()).device
    preds = np.empty((images.shape[0],), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, images.shape[0], batch_size):
            e = min(s + batch_size, images.shape[0])
            x = torch.from_numpy(images[s:e]).to(device)
            p = model(x).cpu().numpy()
            preds[s:e] = np.argmax(p, axis=1)
    return preds


def pick_fixed_attack_image(model, images, labels, img_offset=0, target_label_id=0):
    """Reproduce the FIXED, seed-stable attack instance selected by
    bbattack_objective.CNN(num_img=1, img_offset=<given>, attack_type='targeted',
    random_target_class=None) + utilities.generate_attack_data_set +
    CNN.get_data_sample(i=target_label_id).

    Semantics (verbatim to the original code path):
      1. The test set is first filtered to the CORRECTLY-CLASSIFIED subset
         (where model argmax == true label), preserving order.
      2. `orig_img_id = img_offset` indexes into that correct subset -> the
         attacked image is correct_subset[img_offset].
      3. The 9 targets are seq = list(range(10)) with the true label removed
         (deterministic; np.random.seed(img_offset) is set but unused on this
         non-random path). CNN.get_data_sample(i) then selects target = seq[i].

    RESOLVED INSTANCES (measured with this port's 7797-image correct subset).
    The repo defaults -- bbattack_wrapped.AdversarialAttack and
    bbattack_objective.__main__ both use img_offset=1, target_label=0,
    epsilon=0.3 -- give:

        img_offset=0, target_label_id=0 -> CIFAR test index 1, true=8 (ship),
                                           target=0 (airplane)
        img_offset=1, target_label_id=0 -> CIFAR test index 2, true=8 (ship),
                                           target=0 (airplane)   <-- repo default

    (CIFAR test index 0 is a cat that the CNN misclassifies, so it is dropped by
    the correct-subset filter and correct_subset[0] is original index 1.)
    Because the true label is 8, seq = [0,1,2,3,4,5,6,7,9], so target_label_id
    indexes that list -- note target_label_id=8 maps to class 9, not 8.

    :param model: the ported CIFAR CNN (softmax, NHWC in [-0.5,0.5]).
    :param images: (10000,32,32,3) float32 in [-0.5,0.5] (from load_cifar_test).
    :param labels: (10000,) int true labels.
    :param img_offset: index into the correctly-classified test subset.
    :param target_label_id: index `i` into the 9-element target sequence.
    :return dict with keys:
        image            : (32,32,3) float32 in [-0.5,0.5], the attacked image
        true_label       : int, the image's true (and correctly-predicted) class
        target_label     : int, the attack target class = seq[target_label_id]
        test_index       : int, ORIGINAL index (0..9999) into the CIFAR test set
        correct_subset_index : int, == img_offset (index into the correct subset)
        img_offset       : int, echoed input
        target_label_id  : int, echoed input
    """
    labels = np.asarray(labels).astype(np.int64)
    preds = _predict_labels(model, images)
    correct_idx = np.where(preds == labels)[
        0
    ]  # order-preserving, like np.where in orig

    if img_offset >= correct_idx.shape[0]:
        raise IndexError(
            f"img_offset={img_offset} out of range for "
            f"{correct_idx.shape[0]} correctly-classified images"
        )

    subset_pos = correct_idx[img_offset]
    true_label = int(labels[subset_pos])

    seq = list(range(10))
    seq.remove(true_label)
    if target_label_id > len(seq) - 1:
        raise IndexError(
            f"target_label_id={target_label_id} out of range (0..{len(seq) - 1})"
        )
    target_label = int(seq[target_label_id])

    return {
        "image": images[subset_pos].copy(),
        "true_label": true_label,
        "target_label": target_label,
        "test_index": int(subset_pos),
        "correct_subset_index": int(img_offset),
        "img_offset": int(img_offset),
        "target_label_id": int(target_label_id),
    }
