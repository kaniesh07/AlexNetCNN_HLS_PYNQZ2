# AlexNet-C10 on the PYNQ-Z2: Vitis HLS workflow for CIFAR-10

This is an AlexNet-style network (5 conv + 3 FC, ReLU, max-pool) trained on **CIFAR-10**. It runs on the PYNQ-Z2 (Zynq XC7Z020) as **one Vitis HLS C++ IP**; no handwritten RTL is involved.

| | |
|---|---|
| Accuracy | **88.99%** top-1 on all 10,000 CIFAR-10 test images (exact int8 model, the same arithmetic as the HLS kernel) |
| Latency (estimate) | ≈ 77.8 k clocks ≈ **0.52 ms at 150 MHz** (0.78 ms at 100 MHz) |
| Power measures | all weights on chip (no DDR traffic except the 4 KB image), 2 MACs per DSP, 1-DSP requantiser, minimal resets, `power_opt_design` |
| Compute | 256 int8 MAC/clock on ≈160 DSP48E1, ≈105 BRAM36 (estimates) |

**Verified here:**
- The integer model is exact: 88.99% on 10k images.
- The HLS C++ compiled with g++ gives bit-exact logits and class for 20 test images, and the bad-frame check works.

**Not run here:** Vitis HLS synthesis or co-simulation, Vivado, and the board. The latency and resource figures above are estimates until you build.

## What was reused (not redone)
The CIFAR-10 dataset tools, the trained AlexNet-C10 (`model/trained/qat.pt`), its quantisation code (`model/qmodel.py`) and the ARM-only baseline all come from the earlier CIFAR-10 work. No retraining was done. The ImageNet model and the handwritten-Verilog parts of the original project were removed to keep the flow 100% HLS.

## Layout
```
model/   get_cifar10.py  net.py  train.py  qmodel.py  export_hls.py  trained/{qat.pt,float_final.pt,train_log.json}
hls/     include/cifar_accel.h  include/cifar_weights.h (generated weights + layer table)
         src/cifar_accel.cpp    testbench/tb_cifar_accel.cpp + data/ (100 golden images)
         scripts/run_hls.tcl    Makefile.gcc + csim_stubs/ (g++ C-sim without Vitis)
vivado/  create_project.tcl  create_block_design.tcl  build.tcl
pynq/    cifar_driver.py  classify.py  evaluate.py  arm_reference.py  requirements.txt
docs/    architecture.md  deployment.md  compatibility.md
```

## Quick start
```bash
python model/get_cifar10.py && python model/export_hls.py         # board test set (header already included)
cd hls && vitis-run --mode hls --tcl scripts/run_hls.tcl && cd ..  # csim, csynth, export IP
vivado -mode batch -source vivado/create_project.tcl
vivado -mode batch -source vivado/build.tcl                        # -> build/overlay/
# on the board:
sudo -E python3 evaluate.py --n 10000 --arm
sudo -E python3 classify.py --image dog.jpg
```
For details see `docs/deployment.md`. If timing fails, set `CLK_MHZ=100` for both the HLS and Vivado steps.
