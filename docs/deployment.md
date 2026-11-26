# Build and deployment (in order)

## 0. Model: already done (reused, not retrained)
`model/trained/qat.pt` is the trained, quantisation-aware AlexNet-C10. To retrain:
```bash
python model/get_cifar10.py                         # official CIFAR-10 -> build/data/cifar10_uint8.npz
python model/train.py --epochs 40 --qat-epochs 6    # -> build/qat.pt
```

## 1. Export weights for HLS
```bash
python model/get_cifar10.py                 # only needed once (test set for golden vectors)
python model/export_hls.py                  # -> hls/include/cifar_weights.h, hls/testbench/data, build/deploy
```
The script prints the exact integer accuracy (88.99%), which is what the hardware computes. `hls/include/cifar_weights.h` is already included in this package, so step 1 is only needed after retraining or to regenerate `build/deploy` for the board.

## 2. Vitis HLS 2024.2
```bash
cd hls
vitis-run --mode hls --tcl scripts/run_hls.tcl                         # csim + csynth + export
HLS_STEPS="cosim export" vitis-run --mode hls --tcl scripts/run_hls.tcl  # optional RTL co-sim
# without Vitis (quick check):  make -f Makefile.gcc run
```
Check in `cifar_hls/solution1/syn/report/cifar_accel_csynth.rpt`:
- Loop `STEP` in `run_layer` reaches **II = 1**.
- The estimated clock period is below the target.
- DSP and BRAM usage fit the device.

If timing is missed, set `CLK_MHZ=100` here **and** in step 3.

## 3. Vivado 2024.2
```bash
vivado -mode batch -source vivado/create_project.tcl    # needs PYNQ-Z2 board files
vivado -mode batch -source vivado/build.tcl             # -> build/overlay/cifar_hls.{bit,hwh}, xcifar_accel_hw.h
```

## 4. Board (PYNQ v3.x, no internet needed)
```
cifar/  cifar_driver.py classify.py evaluate.py arm_reference.py
        overlay/cifar_hls.bit overlay/cifar_hls.hwh overlay/xcifar_accel_hw.h
        deploy/cifar10_test.npz deploy/model_info.json deploy/int_params.npz
```
```bash
scp -r pynq/*.py build/overlay build/deploy xilinx@pynq:/home/xilinx/cifar/
sudo -E python3 evaluate.py --n 10000 --arm
sudo -E python3 classify.py --image my_photo.jpg
```

## Troubleshooting
| Symptom | Fix |
|---|---|
| `IP *xilinx.com:hls:cifar_accel* not found` | `.bit` and `.hwh` must have the same name and come from the same build |
| `LOGITS_BASE not found` | copy `xcifar_accel_hw.h` from the same HLS export (`build/overlay/`) |
| timeout | DMA not running: check the HP0 clock, the address map, and that MM2S is enabled |
| `ERR_FRAME` | the image buffer must be exactly 1,024 uint32 words |
| II > 1 in csynth | check the BRAM bank conflicts reported for `act`; the `DEPENDENCE` pragmas must remain |
| timing fails | rebuild everything with `CLK_MHZ=100` |
