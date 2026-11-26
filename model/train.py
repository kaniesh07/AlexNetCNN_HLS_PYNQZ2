#!/usr/bin/env python3
"""
train.py -- train AlexNet-C10 on CIFAR-10 (float, BN), then quantisation-aware
fine-tune (QAT) with the exact int8/uint8 scheme of the RTL accelerator.

    python model/train.py --data build/data/cifar10_uint8.npz --epochs 40 --qat-epochs 6
Outputs (build/):
    float_best.pt     float model (BN)                     + float test accuracy
    qat.pt            folded + QAT weights, activation scales
    train_log.json    per-epoch loss / accuracy
Data file: npz with x_train (50000,32,32,3) uint8, y_train, x_test, y_test
(model/get_cifar10.py builds it from the official python archive).

Recipe (CPU friendly): SGD momentum 0.9 nesterov, OneCycle LR (peak 0.1),
wd 5e-4, batch 128, label smoothing 0.1, random-crop(pad 4) + h-flip + cutout(8).
Validation uses the 10,000-image official test set (no early stopping on it:
the last epoch is kept, 'best' is only reported).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from net import AlexNetC10, QNet, folded_params  # noqa: E402


def load(path):
    d = np.load(path)
    to_t = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()  # noqa: E731  uint8 NCHW
    return to_t(d["x_train"]), torch.from_numpy(d["y_train"]).long(), \
        to_t(d["x_test"]), torch.from_numpy(d["y_test"]).long()


def augment(xb, cutout=8):
    """xb uint8 NCHW -> float [0,1] with random crop (pad 4), flip, cutout."""
    n = xb.shape[0]
    x = F.pad(xb.float() / 255.0, (4, 4, 4, 4))
    i = torch.randint(0, 9, (n,))
    j = torch.randint(0, 9, (n,))
    rows = (i[:, None] + torch.arange(32)[None, :])            # n,32
    cols = (j[:, None] + torch.arange(32)[None, :])
    x = x[torch.arange(n)[:, None, None, None], torch.arange(3)[None, :, None, None],
          rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(n) < 0.5
    x[flip] = x[flip].flip(3)
    if cutout:
        cy = torch.randint(0, 32, (n,))
        cx = torch.randint(0, 32, (n,))
        yy = torch.arange(32)[None, :, None]
        xx = torch.arange(32)[None, None, :]
        m = ((yy - cy[:, None, None]).abs() < cutout // 2) & ((xx - cx[:, None, None]).abs() < cutout // 2)
        x = x * (~m)[:, None].float()
    return x


@torch.no_grad()
def evaluate(model, xt, yt, bs=1000):
    model.eval()
    correct = 0
    for k in range(0, xt.shape[0], bs):
        out = model(xt[k:k + bs].float() / 255.0)
        correct += (out.argmax(1) == yt[k:k + bs]).sum().item()
    return correct / xt.shape[0]


def run_epochs(model, params, xtr, ytr, xte, yte, epochs, lr, wd, bs, log, tag, ckpt, start=0,
               sched_state=None, opt_state=None, on_epoch=None):
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    steps = (xtr.shape[0] // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps,
                                                pct_start=0.25, div_factor=10, final_div_factor=200)
    if opt_state:
        opt.load_state_dict(opt_state)
        sched.load_state_dict(sched_state)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    for ep in range(start, epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(xtr.shape[0])
        tot, n = 0.0, 0
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            x = augment(xtr[idx])
            loss = crit(model(x), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        if on_epoch:
            on_epoch(ep)
        acc = evaluate(model, xte, yte)
        rec = dict(phase=tag, epoch=ep + 1, loss=tot / n, test_acc=acc, sec=time.time() - t0)
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=ep + 1, log=log), ckpt)
    return model


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(HERE)
    ap.add_argument("--data", default=os.path.join(root, "build", "data", "cifar10_uint8.npz"))
    ap.add_argument("--out", default=os.path.join(root, "build"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--qat-epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0, help="time N training steps and exit")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    xtr, ytr, xte, yte = load(a.data)
    model = AlexNetC10()
    print("params:", sum(p.numel() for p in model.parameters()))

    if a.bench:
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        t0 = time.time()
        for s in range(a.bench):
            idx = torch.randint(0, 50000, (a.bs,))
            loss = F.cross_entropy(model(augment(xtr[idx])), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        dt = (time.time() - t0) / a.bench
        print(f"{dt*1000:.0f} ms/step -> {dt*50000/a.bs/60:.1f} min/epoch")
        return

    log = []
    ck_f = os.path.join(a.out, "float_ckpt.pt")
    start, ost, sst = 0, None, None
    if os.path.exists(ck_f):                     # resume
        ck = torch.load(ck_f, weights_only=False)
        model.load_state_dict(ck["model"])
        start, ost, sst, log = ck["epoch"], ck["opt"], ck["sched"], ck["log"]
        print("resuming float training at epoch", start)
    if start < a.epochs:
        run_epochs(model, model.parameters(), xtr, ytr, xte, yte, a.epochs, a.lr, a.wd, a.bs,
                   log, "float", ck_f, start, sst, ost)
    float_acc = evaluate(model, xte, yte)
    torch.save(model.state_dict(), os.path.join(a.out, "float_final.pt"))
    print(f"FLOAT test accuracy: {float_acc*100:.2f}%")

    # ---------------- QAT ----------------
    q = QNet(folded_params(model))
    # calibration of activation ranges (no weight updates)
    q.train()
    with torch.no_grad():
        for s in range(60):
            idx = torch.randint(0, 50000, (256,))
            q(xtr[idx].float() / 255.0)
    print("calibrated act amax:", [round(float(m.amax), 3) for m in q.aq])
    print(f"QAT model before fine-tune: {evaluate(q, xte, yte)*100:.2f}%")

    def freeze_after(ep):
        # freeze the activation ranges after the first QAT epoch
        if ep == 0:
            for m in q.aq:
                m.calibrating = False
    ck_q = os.path.join(a.out, "qat_ckpt.pt")
    if a.qat_epochs:
        run_epochs(q, q.parameters(), xtr, ytr, xte, yte, a.qat_epochs, a.lr * 0.05, a.wd * 0.2,
                   a.bs, log, "qat", ck_q, on_epoch=freeze_after)
    for m in q.aq:
        m.calibrating = False
    qacc = evaluate(q, xte, yte)
    print(f"QAT (fake-quant) test accuracy: {qacc*100:.2f}%")
    torch.save(dict(names=q.names, w=[p.detach() for p in q.w], b=[p.detach() for p in q.b],
                    act_scales=q.act_scales(), float_acc=float_acc, qat_acc=qacc),
               os.path.join(a.out, "qat.pt"))
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
