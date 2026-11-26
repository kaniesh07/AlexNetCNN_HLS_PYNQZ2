#!/usr/bin/env python3
"""
classify.py -- classify image files on the PYNQ-Z2 (images resized to 32x32 RGB).
  sudo -E python3 classify.py --image dog.jpg cat.png
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cifar_driver import CifarAccel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default=os.path.join(HERE, "overlay", "cifar_hls.bit"))
    ap.add_argument("--header", default=os.path.join(HERE, "overlay", "xcifar_accel_hw.h"))
    ap.add_argument("--info", default=os.path.join(HERE, "deploy", "model_info.json"))
    ap.add_argument("--image", nargs="+", required=True)
    a = ap.parse_args()
    from pynq import Overlay
    info = json.load(open(a.info))
    ol = Overlay(a.overlay)
    with CifarAccel(ol, a.header) as acc:
        for path in a.image:
            img = np.asarray(Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR))
            r = acc.infer(img)
            z = (r["logits"] - r["logits"].max()) * info["fc8_logit_scale"]
            p = np.exp(z) / np.exp(z).sum()
            print(f"{path}: {info['classes'][r['cls']]} ({p[r['cls']]*100:.1f}%)  "
                  f"latency {r['wall_s']*1e6:.0f} us (start -> done, incl. 4 KB DMA)")


if __name__ == "__main__":
    main()
