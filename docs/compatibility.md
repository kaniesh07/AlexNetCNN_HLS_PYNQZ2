# Compatibility and installation checklist

| | Host PC (build) | PYNQ-Z2 (run) |
|---|---|---|
| Tools | Vivado 2024.2 + Vitis HLS 2024.2 (Vitis unified installer, Zynq-7000 support) | PYNQ v3.x SD image (v3.1 has a Z2 image) |
| Python | ≥ 3.9 with torch, torchvision, numpy, Pillow (training/export only) | PYNQ's Python with pynq, numpy, Pillow (preinstalled) |
| Produces / uses | `cifar_hls.bit`, `cifar_hls.hwh`, `xcifar_accel_hw.h` | the same files + `deploy/` |

- [ ] Device: XC7Z020-1CLG400C, which has 53,200 LUTs, 106,400 FFs, 140 BRAM36 and 220 DSP48E1. Part string: `xc7z020clg400-1`.
- [ ] `vitis-run -h` lists `--mode hls` (or the legacy `vitis_hls` is installed).
- [ ] PYNQ-Z2 board files are copied to `<Xilinx>/Vivado/2024.2/data/xhub/boards/XilinxBoardStore/boards/Xilinx/pynq-z2` (or `PYNQ_Z2_BOARD_REPO` is set). Check with `get_board_parts *pynq-z2*`.
- [ ] Record the board image version: `cat /etc/pynq_version`. PYNQ 3.1's own overlays were built with Vivado 2024.1; a 2024.2 bitstream is loaded the same way, but test it with `evaluate.py`.
- [ ] No DPU or Vitis AI is used. Only Vitis **HLS** (C++ → RTL) is used.
- [ ] Only 4 KB of contiguous memory is needed on the board (the image buffer), so there are no CMA constraints.
