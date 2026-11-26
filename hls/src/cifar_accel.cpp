// ============================================================================
// cifar_accel.cpp -- AlexNet-C10 inference kernel for Vitis HLS 2024.2
// (see include/cifar_accel.h for the interface and numerical contract)
//
// Architecture (single time-multiplexed layer engine, everything on chip)
//   ibuf          image, 1024 x 24 bit (BRAM)
//   act[2][4]     ping-pong activation buffers, 4 banks each (BRAM, 1R+1W);
//                 channel group cg lives in bank cg%4 at
//                 address ((cg/4) << 2lgW) | (y << lgW) | x
//   WROM[n][32]   int8 weights, 32 banks x 64 bit (BRAM ROM)
//   BIAS/MULT/SHIFT per-channel params (LUT ROM)
//   run_layer()   one pipelined loop (II = 1) per layer:
//                   8 input channels x 32 output channels = 256 MAC / clock
//                   on 128 DSP48E1 (two MACs packed per DSP, mac2())
//                 fused: bias, requantisation (32 x 1 DSP), ReLU, 2x2 max-pool,
//                 write-back of 4 x 64-bit words (one per bank) per output pixel
// Loop nest per layer, flattened:
//   tile t (32 out ch) > py > px > pool sub-pixel s > cg > ky > kx
// fc6 is executed as an exact 4x4 convolution on the 64 x 4 x 4 pool5 map,
// fc7 / fc8 as 1x1 convolutions on a 1x1 map.
// ============================================================================
#include "cifar_accel.h"
#include "cifar_weights.h"

typedef ap_uint<64> word_t;

// two signed int8 x unsigned uint8 products in ONE DSP48E1:
//   P = (w_hi * 2^17 + w_lo) * x ;  lo = P[16:0] ; hi = P[33:17] + P[16]
// exact because |w| <= 127 and x <= 255 (|x*w| < 2^15)
static void mac2(ap_int<8> w_lo, ap_int<8> w_hi, ap_uint<8> x,
                 ap_int<17> &p_lo, ap_int<17> &p_hi) {
#pragma HLS INLINE
    ap_int<25> pa = ((ap_int<25>)w_hi << 17) + (ap_int<25>)w_lo;
    ap_int<18> b  = (ap_int<18>)x;
    ap_int<43> p  = pa * b;
#pragma HLS BIND_OP variable=p op=mul impl=dsp
    ap_int<17> lo, hi;
    lo.range(16, 0) = p.range(16, 0);
    hi.range(16, 0) = p.range(33, 17);
    p_lo = lo;
    p_hi = hi + (ap_int<17>)(p[16] ? 1 : 0);
}

// int32 accumulator (+bias) -> uint8 activation, one DSP per lane
static ap_uint<8> requant(ap_int<33> tot, ap_uint<16> m, ap_uint<6> sh) {
#pragma HLS INLINE
    ap_uint<24> a = (tot <= 0) ? ap_uint<24>(0)
                  : (tot >= 0xFFFFFF) ? ap_uint<24>(0xFFFFFF) : ap_uint<24>(tot);
    ap_uint<40> p = a * m;
#pragma HLS BIND_OP variable=p op=mul impl=dsp
    ap_uint<41> rnd = ap_uint<41>(1) << (sh - 1);
    ap_uint<41> r   = (p + rnd) >> sh;
    return (r > 255) ? ap_uint<8>(255) : ap_uint<8>(r);
}

static void run_layer(int li, const ap_uint<24> ibuf[IMG_PIX],
                      word_t act[2][4][BANK_DEPTH], int src, int lg[NCLS]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=WROM  type=complete dim=2
#pragma HLS ARRAY_PARTITION variable=BIAS  type=complete dim=2
#pragma HLS ARRAY_PARTITION variable=MULT  type=complete dim=2
#pragma HLS ARRAY_PARTITION variable=SHIFT type=complete dim=2
#pragma HLS BIND_STORAGE variable=WROM  type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=BIAS  type=rom_1p impl=lutram
#pragma HLS BIND_STORAGE variable=MULT  type=rom_1p impl=lutram
#pragma HLS BIND_STORAGE variable=SHIFT type=rom_1p impl=lutram

    const int K     = LCFG[li].K;
    const int pad   = LCFG[li].pad;
    const int pool  = LCFG[li].pool;
    const int raw   = LCFG[li].raw;
    const int lgW   = LCFG[li].lgW;
    const int lgWo  = LCFG[li].lgWo;
    const int Hc    = LCFG[li].Hc;
    const int cig   = LCFG[li].cig;
    const int nt    = LCFG[li].nt;
    const int tbase = LCFG[li].tbase;
    const int W     = 1 << lgW;
    const int Wl    = pool ? (Hc >> 1) : Hc;       // output loop width
    const int nsub  = pool ? 4 : 1;
    const int cigkk = cig * K * K;
    const int total = nt * Wl * Wl * nsub * cigkk;
    const int dst   = 1 - src;

    ap_int<32> acc[PO];
    ap_uint<8> pm[PO];
#pragma HLS ARRAY_PARTITION variable=acc type=complete
#pragma HLS ARRAY_PARTITION variable=pm  type=complete

    int t = 0, py = 0, px = 0, s = 0, cg = 0, ky = 0, kx = 0;
    int wtile = LCFG[li].wbase, wptr = LCFG[li].wbase;

STEP:
    for (int n = 0; n < total; n++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=20736 avg=9578
#pragma HLS DEPENDENCE variable=act type=inter false
#pragma HLS DEPENDENCE variable=act type=intra false
        // ---- input activation word (zero padding) ----------------------
        const int oy = pool ? ((py << 1) | (s >> 1)) : py;
        const int ox = pool ? ((px << 1) | (s & 1)) : px;
        const int iy = oy + ky - pad;
        const int ix = ox + kx - pad;
        const bool oob = (iy < 0) || (ix < 0) || (iy >= W) || (ix >= W);
        word_t x = 0;
        if (!oob) {
            if (li == 0) x = (word_t)ibuf[(iy << 5) | ix];
            else         x = act[src][cg & 3][((cg >> 2) << (2 * lgW)) | (iy << lgW) | ix];
        }
        const bool first     = (cg == 0) && (ky == 0) && (kx == 0);
        const bool e_kx      = (kx == K - 1);
        const bool e_ky      = e_kx && (ky == K - 1);
        const bool last      = e_ky && (cg == cig - 1);      // output pixel complete
        const bool sub_first = (s == 0);
        const bool sub_last  = (s == nsub - 1);
        const int  gt        = tbase + t;

        // ---- 256 MACs: 16 lane pairs x 8 inputs, two MACs per DSP --------
        ap_int<17> prod[PO][PI];
#pragma HLS ARRAY_PARTITION variable=prod type=complete dim=0
        for (int j = 0; j < PO / 2; j++) {
#pragma HLS UNROLL
            const word_t wl = WROM[wptr][2 * j];
            const word_t wh = WROM[wptr][2 * j + 1];
            for (int i = 0; i < PI; i++) {
#pragma HLS UNROLL
                mac2((ap_int<8>)wl.range(8 * i + 7, 8 * i), (ap_int<8>)wh.range(8 * i + 7, 8 * i),
                     (ap_uint<8>)x.range(8 * i + 7, 8 * i), prod[2 * j][i], prod[2 * j + 1][i]);
            }
        }

        // ---- accumulate; on 'last': bias, requant, ReLU, pool, write ----
        word_t outw[4];
#pragma HLS ARRAY_PARTITION variable=outw type=complete
        for (int g = 0; g < 4; g++) {
#pragma HLS UNROLL
            outw[g] = 0;
        }
        for (int o = 0; o < PO; o++) {
#pragma HLS UNROLL
            ap_int<20> sum = 0;
            for (int i = 0; i < PI; i++) {
#pragma HLS UNROLL
                sum += prod[o][i];
            }
            const ap_int<32> a = (first ? ap_int<32>(0) : acc[o]) + sum;
            acc[o] = a;
            if (last) {
                const ap_int<33> tot = (ap_int<33>)a + (ap_int<33>)BIAS[gt][o];
                if (raw) {
                    if (o < NCLS) lg[o] = (int)tot;
                } else {
                    const ap_uint<8> y = requant(tot, MULT[gt][o], SHIFT[gt][o]);
                    const ap_uint<8> v = (sub_first || y > pm[o]) ? y : pm[o];
                    pm[o] = v;
                    outw[o / 8].range(8 * (o % 8) + 7, 8 * (o % 8)) = v;
                }
            }
        }
        if (last && !raw && sub_last) {
            const int waddr = (t << (2 * lgWo)) | (py << lgWo) | px;
            for (int g = 0; g < 4; g++) {
#pragma HLS UNROLL
                act[dst][g][waddr] = outw[g];
            }
        }

        // ---- advance the flattened loop nest ------------------------------
        const bool e_s  = last && sub_last;
        const bool e_px = e_s && (px == Wl - 1);
        const bool e_py = e_px && (py == Wl - 1);
        kx = e_kx ? 0 : kx + 1;
        if (e_kx) ky = e_ky ? 0 : ky + 1;
        if (e_ky) cg = last ? 0 : cg + 1;
        if (last) s = sub_last ? 0 : s + 1;
        if (e_s)  px = e_px ? 0 : px + 1;
        if (e_px) py = e_py ? 0 : py + 1;
        if (e_py) t++;
        if (last) {                      // next pixel restarts at the tile's first weight word
            if (e_py) wtile += cigkk;    // next tile
            wptr = wtile;
        } else {
            wptr++;
        }
    }
}

int cifar_accel(hls::stream<axis_t> &img, int logits[NCLS]) {
#pragma HLS INTERFACE mode=axis      port=img
#pragma HLS INTERFACE mode=s_axilite port=logits bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    static ap_uint<24> ibuf[IMG_PIX];
    static word_t act[2][4][BANK_DEPTH];
#pragma HLS ARRAY_PARTITION variable=act type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=act type=complete dim=2
#pragma HLS BIND_STORAGE variable=act  type=ram_s2p impl=bram
#pragma HLS BIND_STORAGE variable=ibuf type=ram_s2p impl=bram

    // ---- image: exactly 1024 beats, TLAST on the last one ------------------
    int nbeats = 0;
    bool seen_last = false;
READ:
    for (int i = 0; i < IMG_PIX; i++) {
#pragma HLS PIPELINE II=1
        axis_t b = img.read();
        ibuf[i] = b.data.range(23, 0);
        nbeats = i + 1;
        if (b.last) {
            seen_last = true;
            break;
        }
    }
    if (!seen_last || nbeats != IMG_PIX) return ERR_FRAME;

    // ---- 8 layers on one engine (ping-pong buffers) ---------------------------
    int lg[NCLS];
#pragma HLS ARRAY_PARTITION variable=lg type=complete
LAYERS:
    for (int li = 0; li < NLAYERS; li++) {
        run_layer(li, ibuf, act, li & 1, lg);
    }

    // ---- argmax (lowest index wins ties) + logits to AXI-Lite ---------------------
    int best = lg[0], cls = 0;
ARGMAX:
    for (int i = 0; i < NCLS; i++) {
#pragma HLS PIPELINE II=1
        logits[i] = lg[i];
        if (i > 0 && lg[i] > best) {
            best = lg[i];
            cls  = i;
        }
    }
    return cls;
}
