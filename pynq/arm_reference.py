"""
arm_reference.py -- ARM-only baseline: the identical integer network in NumPy
(same int8 weights, biases, requant multipliers; bit-exact with the RTL).
Reads deploy/int_params.npz written by model/export_rtl.py.
"""
import numpy as np

CONVS = ["conv1", "conv2", "conv3", "conv4", "conv5"]
POOL = {"conv1": True, "conv2": True, "conv3": False, "conv4": False, "conv5": True}


def _requant(acc, M, sh):
    shape = (1, -1) + (1,) * (acc.ndim - 2)
    y = (acc * M.reshape(shape) + (np.int64(1) << (sh.reshape(shape) - 1))) >> sh.reshape(shape)
    return np.clip(y, 0, 255)


def _conv3x3(x, w, b):
    N, C, H, W = x.shape
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.stack([xp[:, :, ky:ky + H, kx:kx + W] for ky in range(3) for kx in range(3)], 2)
    cols = cols.reshape(N, C * 9, H * W)
    wm = w.reshape(w.shape[0], C, 9).reshape(w.shape[0], -1).astype(np.float64)
    acc = np.rint(np.einsum("ok,nkp->nop", wm, cols)).astype(np.int64) + b[None, :, None]
    return acc.reshape(N, -1, H, W)


class ArmInt8:
    def __init__(self, path):
        z = np.load(path)
        self.p = {k: z[k].astype(np.int64) for k in z.files}

    def forward(self, x_u8, batch=50):
        out = []
        for k in range(0, len(x_u8), batch):
            x = x_u8[k:k + batch].transpose(0, 3, 1, 2).astype(np.int64)
            for n in CONVS:
                x = _requant(_conv3x3(x, self.p[n + "_wq"], self.p[n + "_bq"]), self.p[n + "_M"], self.p[n + "_sh"])
                if POOL[n]:
                    N, C, H, W = x.shape
                    x = x.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))
            v = x.reshape(x.shape[0], -1)
            for n in ["fc6", "fc7", "fc8"]:
                acc = np.rint(v.astype(np.float64) @ self.p[n + "_wq"].T.astype(np.float64)).astype(np.int64)
                acc += self.p[n + "_bq"]
                v = acc if n == "fc8" else _requant(acc, self.p[n + "_M"], self.p[n + "_sh"])
            out.append(v)
        return np.concatenate(out).astype(np.int32)
