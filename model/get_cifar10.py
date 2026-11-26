#!/usr/bin/env python3
"""
get_cifar10.py -- build build/data/cifar10_uint8.npz from the OFFICIAL archive.

  python model/get_cifar10.py                       # downloads cifar-10-python.tar.gz
  python model/get_cifar10.py --tar cifar-10-python.tar.gz     # offline
Source: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
Output: x_train (50000,32,32,3) uint8, y_train, x_test (10000,32,32,3), y_test
Sanity checks: 5000/1000 images per class, train mean ~ (0.4914, 0.4822, 0.4465).
"""
import argparse
import os
import pickle
import tarfile
import urllib.request

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "data", "cifar10_uint8.npz"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tar = a.tar or os.path.join(os.path.dirname(a.out), "cifar-10-python.tar.gz")
    if not os.path.exists(tar):
        print("downloading", URL)
        urllib.request.urlretrieve(URL, tar)
    xs, ys = {}, {}
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if name.startswith("data_batch_") or name == "test_batch":
                d = pickle.load(tf.extractfile(m), encoding="bytes")
                xs[name] = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                ys[name] = np.array(d[b"labels"], np.uint8)
    xtr = np.concatenate([xs[f"data_batch_{i}"] for i in range(1, 6)])
    ytr = np.concatenate([ys[f"data_batch_{i}"] for i in range(1, 6)])
    xte, yte = xs["test_batch"], ys["test_batch"]
    assert xtr.shape == (50000, 32, 32, 3) and xte.shape == (10000, 32, 32, 3)
    assert (np.bincount(ytr) == 5000).all() and (np.bincount(yte) == 1000).all()
    print("train mean", (xtr / 255).mean((0, 1, 2)).round(4))
    np.savez_compressed(a.out, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
