// ============================================================================
// cifar_accel.h -- Vitis HLS 2024.2 AlexNet-C10 (CIFAR-10) accelerator
//
// INTERFACE CONTRACT
//   img     AXI4-Stream slave (32-bit, TLAST).  Exactly 1024 beats per image,
//           raster order (y*32+x), beat = {8'h00, B, G, R}, TLAST on beat 1023.
//   logits  int32[10] in the AXI4-Lite register space (s_axilite array):
//           raw fc8 accumulators, real logit = value * fc8_logit_scale
//           (build/deploy/model_info.json).
//   return  (ap_return, AXI4-Lite) predicted class 0..9 (argmax, lowest index
//           wins ties), or ERR_FRAME (-1) when TLAST is not on beat 1023.
//   control ap_ctrl_hs on s_axilite "control".  All offsets are assigned by
//           Vitis HLS: read them from the generated xcifar_accel_hw.h (the
//           PYNQ driver parses it; nothing is hard-coded).
//
// The whole network (361,568 int8 weights) is held in on-chip ROMs; the only
// external data movement is the 4 KB image.
//
// NUMERICAL CONTRACT (bit-exact with model/qmodel.py, verified by the C TB)
//   x0 = uint8 pixel (scale 1/255); weights int8 in [-127,127]
//   acc = bias + sum(w*x)                                  int32
//   y   = clamp((min(max(acc,0), 2^24-1) * M + 2^(sh-1)) >> sh, 0, 255)
//         (identical to clamp((acc*M + 2^(sh-1)) >> sh, 0, 255) for every
//          channel -- checked by model/export_hls.py)
//   2x2 max-pool on uint8; fc8 = raw int32 accumulators -> argmax.
// ============================================================================
#ifndef CIFAR_ACCEL_H
#define CIFAR_ACCEL_H

#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

typedef ap_axiu<32, 0, 0, 0> axis_t;

static const int PI         = 8;     // input channels per 64-bit activation word
static const int PO         = 32;    // output channels computed in parallel
static const int NLAYERS    = 8;
static const int NCLS       = 10;
static const int IMG_PIX    = 1024;  // 32 x 32
static const int BANK_DEPTH = 256;   // words per activation bank (4 banks per buffer)
static const int ERR_FRAME  = -1;

struct LayerCfg {
    int K;      // kernel size (3 conv, 4 fc6, 1 fc7/fc8)
    int pad;    // zero padding
    int pool;   // fused 2x2 / stride-2 max-pool
    int raw;    // 1: output raw int32 logits (fc8)
    int lgW;    // log2 of input width (= height)
    int lgWo;   // log2 of stored output width (after pooling)
    int Hc;     // conv output width before pooling
    int cig;    // input channel groups of 8
    int nt;     // output tiles of 32 channels
    int wbase;  // first weight-ROM word of the layer
    int tbase;  // first param-ROM row of the layer
};

int cifar_accel(hls::stream<axis_t> &img, int logits[NCLS]);

#endif
