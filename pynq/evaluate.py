#!/usr/bin/env python3
"""
evaluate.py -- on-board CIFAR-10 evaluation of the HLS accelerator.
  sudo -E python3 evaluate.py               # all 10,000 test images
  sudo -E python3 evaluate.py --n 500 --arm # + ARM-only int8 baseline
Reports top-1 (and per class), bit-exactness vs the integer model, latency
(median / p95, start->done measured on the ARM) and the ARM-only baseline.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cifar_driver import CifarAccel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default=os.path.join(HERE, "overlay", "cifar_hls.bit"))
    ap.add_argument("--header", default=os.path.join(HERE, "overlay", "xcifar_accel_hw.h"))
    ap.add_argument("--deploy", default=os.path.join(HERE, "deploy"))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--arm", action="store_true")
    a = ap.parse_args()
    from pynq import Overlay
    d = np.load(os.path.join(a.deploy, "cifar10_test.npz"))
    info = json.load(open(os.path.join(a.deploy, "model_info.json")))
    x, y, exp = d["x"][:a.n], d["y"][:a.n].astype(int), d["expected_logits"][:a.n]
    ol = Overlay(a.overlay)
    preds, lat, mism = [], [], 0
    with CifarAccel(ol, a.header) as acc:
        t0 = time.perf_counter()
        for i in range(len(x)):
            r = acc.infer(x[i])
            preds.append(r["cls"])
            lat.append(r["wall_s"])
            mism += int(not np.array_equal(r["logits"], exp[i]))
        total = time.perf_counter() - t0
    preds, lat = np.array(preds), np.array(lat)
    print(f"images                  : {len(x)}")
    print(f"FPGA top-1              : {(preds == y).mean()*100:.2f}%  "
          f"(host integer model on 10k: {info['int_test_acc']*100:.2f}%)")
    print(f"logit mismatches        : {mism}  (must be 0)")
    for c, name in enumerate(info["classes"]):
        m = y == c
        if m.any():
            print(f"   {name:10s} {(preds[m] == c).mean()*100:6.2f}%")
    print(f"latency start->done     : median {np.median(lat)*1e6:.0f} us, p95 {np.percentile(lat, 95)*1e6:.0f} us"
          f"  (expected kernel ~{info['mac_steps']} + ~1.2k clocks)")
    print(f"throughput (python loop): {len(x)/total:.0f} images/s")
    if a.arm:
        from arm_reference import ArmInt8
        arm = ArmInt8(os.path.join(a.deploy, "int_params.npz"))
        n = min(len(x), 200)
        t0 = time.perf_counter()
        la = arm.forward(x[:n])
        ta = (time.perf_counter() - t0) / n
        print(f"ARM-only int8 (NumPy)   : {ta*1e3:.1f} ms/image, bit-exact: {np.array_equal(la, exp[:n])}")
        print(f"speed-up                : {ta/np.median(lat):.0f}x")


if __name__ == "__main__":
    main()
