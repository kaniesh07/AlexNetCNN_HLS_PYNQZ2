"""
cifar_driver.py -- PYNQ (v3.x) driver for the Vitis HLS AlexNet-C10 IP (cifar_accel).

* register offsets are NOT hard-coded: they are parsed from the Vitis HLS
  generated header xcifar_accel_hw.h (copied next to the bitstream by
  vivado/build.tcl):  AP_CTRL, AP_RETURN, LOGITS_BASE
* image path: 1024 x uint32 beats {0,B,G,R} -> AXI DMA MM2S -> cifar_accel/img
* results: ap_return = class (or -1 = bad frame), logits[10] (int32, s_axilite)
"""
import re
import time

import numpy as np

AP_START, AP_DONE, AP_IDLE = 0x1, 0x2, 0x4


class AccelError(RuntimeError):
    pass


def parse_hls_header(path):
    txt = open(path).read()
    addr = {m.group(1): int(m.group(2), 16)
            for m in re.finditer(r"#define\s+XCIFAR_ACCEL_CONTROL_ADDR_(\w+)\s+0x([0-9a-fA-F]+)", txt)}
    need = ["AP_CTRL", "AP_RETURN", "LOGITS_BASE"]
    missing = [k for k in need if k not in addr]
    if missing:
        raise AccelError(f"{path}: {missing} not found -- is this the cifar_accel header?")
    return addr


def _find(ol, substr):
    hits = [k for k, v in ol.ip_dict.items() if substr in v.get("type", "")]
    if len(hits) != 1:
        raise AccelError(f"expected one IP *{substr}*, found {hits}; IPs: {sorted(ol.ip_dict)}")
    return hits[0]


def pack_image(img_u8):
    """(32,32,3) uint8 RGB -> 1024 uint32 beats {8'h00, B, G, R}, raster order."""
    p = np.asarray(img_u8, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    return p[:, 0] | (p[:, 1] << 8) | (p[:, 2] << 16)


class CifarAccel:
    def __init__(self, overlay, header_path, timeout_s=0.5):
        from pynq import MMIO, allocate
        name = _find(overlay, "xilinx.com:hls:cifar_accel")
        ent = overlay.ip_dict[name]
        self.mmio = MMIO(ent["phys_addr"], ent["addr_range"])
        self.dma = getattr(overlay, _find(overlay, "xilinx.com:ip:axi_dma"))
        self.reg = parse_hls_header(header_path)
        self.timeout_s = timeout_s
        self.buf = allocate(shape=(1024,), dtype=np.uint32)
        if not self.mmio.read(self.reg["AP_CTRL"]) & AP_IDLE:
            raise AccelError("accelerator not idle after overlay load")

    def close(self):
        try:
            self.buf.freebuffer()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def infer(self, img_u8):
        """-> dict(cls, logits[int32 x10], hw_s (start->done), wall_s)."""
        self.buf[:] = pack_image(img_u8)
        self.buf.flush()
        t0 = time.perf_counter()
        self.mmio.write(self.reg["AP_CTRL"], AP_START)
        self.dma.sendchannel.transfer(self.buf)
        while True:
            ctrl = self.mmio.read(self.reg["AP_CTRL"])      # ap_done is clear-on-read
            if ctrl & AP_DONE:
                break
            if time.perf_counter() - t0 > self.timeout_s:
                raise AccelError(f"timeout (AP_CTRL=0x{ctrl:x}): check DMA / stream / clocks")
        t1 = time.perf_counter()
        self.dma.sendchannel.wait()
        cls = np.int32(np.uint32(self.mmio.read(self.reg["AP_RETURN"])))
        if cls < 0:
            raise AccelError("kernel returned ERR_FRAME: image must be exactly 1024 beats")
        base = self.reg["LOGITS_BASE"]
        logits = np.array([self.mmio.read(base + 4 * i) for i in range(10)], np.uint32).view(np.int32)
        return dict(cls=int(cls), logits=logits, wall_s=t1 - t0)
