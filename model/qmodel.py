"""
qmodel.py -- the integer contract of the RTL accelerator (numpy only).

Everything the Verilog computes is defined here, bit-exactly:

  layer table (HW_LAYERS)   geometry, tiling, ROM bases, loop bounds
  quantize()                float (BN-folded) params + activation scales -> ints
  forward_int()             batched exact integer inference -> int32 fc8 accumulators
  rom images                weight ROM (2048-bit words), param ROM, layer cfg header

Numerical contract
  x0      = uint8 pixel, scale 1/255 (channels R,G,B in lanes 0..2 of conv1)
  acc     = bias_q + sum(w_q * x)                         int32 (w_q int8 in [-127,127])
  hidden  : y = clamp((acc*M + 2^(sh-1)) >> sh, 0, 255)   uint8, ReLU implied by clamp
            M in [0, 32767], sh in [1, 47], >> is arithmetic
  pool    : 2x2 stride 2 max of uint8 (exact after requant: requant is monotonic)
  fc8     : raw int32 accumulators, one weight scale for all classes ->
            class = argmax (lowest index wins ties)
Hardware parallelism: PI = 8 input channels x PO = 32 output channels
(256 int8 MAC / cycle on 128 DSP48E1, two MACs packed per DSP).
"""
import numpy as np

PI = 8          # input channels per activation word (64-bit word)
PO = 32         # output channels per tile
MIN_CIGKK = 9   # output-stage serialisation needs >= 9 cycles per output pixel

# --------------------------------------------------------------------------
# Network geometry (hardware view). cin is padded to a multiple of 8,
# cout to a multiple of 32.
# --------------------------------------------------------------------------
def build_layers(cfg):
    C = [3, cfg["C1"], cfg["C2"], cfg["C3"], cfg["C4"], cfg["C5"]]
    pool = [1, 1, 0, 0, 1]
    L = []
    h = 32
    for i in range(5):
        L.append(dict(name=f"conv{i+1}", fc=0, cin=C[i], cout=C[i + 1], H=h, K=3, pad=1, pool=pool[i], raw=0))
        if pool[i]:
            h //= 2
    nin = C[5] * h * h
    L.append(dict(name="fc6", fc=1, cin=nin, cout=cfg["F6"], H=1, K=1, pad=0, pool=0, raw=0))
    L.append(dict(name="fc7", fc=1, cin=cfg["F6"], cout=cfg["F7"], H=1, K=1, pad=0, pool=0, raw=0))
    L.append(dict(name="fc8", fc=1, cin=cfg["F7"], cout=cfg["NCLS"], H=1, K=1, pad=0, pool=0, raw=1))
    wbase = tbase = 0
    for l in L:
        l["cin_p"] = -(-l["cin"] // PI) * PI
        l["cout_p"] = -(-l["cout"] // PO) * PO
        l["cig"] = l["cin_p"] // PI
        l["nt"] = l["cout_p"] // PO
        l["cigkk"] = l["cig"] * l["K"] * l["K"]
        l["Ho"] = l["H"] // 2 if l["pool"] else l["H"]
        l["lgW"] = int(np.log2(l["H"]))
        l["lgHW"] = 2 * l["lgW"]
        l["lgWo"] = int(np.log2(l["Ho"]))
        l["lgHWo"] = 2 * l["lgWo"]
        l["wbase"], l["tbase"] = wbase, tbase
        wbase += l["nt"] * l["cigkk"]
        tbase += l["nt"]
        assert 2 ** l["lgW"] == l["H"], "H=W must be a power of two"
        assert l["cigkk"] >= MIN_CIGKK, f"{l['name']}: cig*K*K={l['cigkk']} < {MIN_CIGKK}"
        # cycles: one 256-MAC step per (tile, output pixel, cin group, ky, kx)
        l["cycles"] = l["nt"] * l["H"] * l["H"] * l["cigkk"]
    # FC input of fc6 comes from a C5 x h x h map stored as words (cg, y, x)
    L[5]["src_c"], L[5]["src_hw"] = C[5], h * h
    return L


# --------------------------------------------------------------------------
# Quantisation
# --------------------------------------------------------------------------
def choose_multiplier(m):
    m = np.asarray(m, dtype=np.float64)
    M = np.zeros(m.shape, np.int64)
    sh = np.ones(m.shape, np.int64)
    for i, mi in np.ndenumerate(m):
        if not np.isfinite(mi) or mi <= 0:
            continue
        s = min(max(14 - int(np.floor(np.log2(mi))), 1), 47)
        q = int(round(mi * (1 << s)))
        if q >= 32768:
            s -= 1
            q = int(round(mi * (1 << s)))
        M[i], sh[i] = min(q, 32767), s
    return M, sh


def quantize(folded, act_scales, layers):
    """folded: list of (name, w float, b float) for conv1..fc8 (BN folded).
    act_scales: 7 floats = output scales of conv1..conv5, fc6, fc7.
    Returns per-layer dicts with int tensors in TORCH layout."""
    s_in = [1.0 / 255.0] + list(act_scales)
    out = []
    for i, ((name, w, b), L) in enumerate(zip(folded, layers)):
        w = np.asarray(w, np.float64)
        b = np.asarray(b, np.float64)
        O = w.shape[0]
        if L["raw"]:
            sw = np.full(O, max(np.abs(w).max(), 1e-12) / 127.0)
        else:
            sw = np.maximum(np.abs(w.reshape(O, -1)).max(1), 1e-12) / 127.0
        wq = np.clip(np.rint(w / sw.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127).astype(np.int64)
        bq = np.rint(b / (s_in[i] * sw)).astype(np.int64)
        if L["raw"]:
            M, sh = np.zeros(O, np.int64), np.ones(O, np.int64)
            s_out = None
        else:
            s_out = act_scales[i]
            M, sh = choose_multiplier(s_in[i] * sw / s_out)
        fan = np.abs(wq.reshape(O, -1)).sum(1) * 255 + np.abs(bq)
        assert fan.max() < 2 ** 31, f"{name}: accumulator overflow possible"
        out.append(dict(name=name, wq=wq, bq=bq, M=M, sh=sh, s_in=s_in[i], s_w=sw, s_out=s_out,
                        worst_acc=int(fan.max())))
    return out


# --------------------------------------------------------------------------
# Exact integer forward pass (batched).  x: uint8 (N,32,32,3) HWC.
# --------------------------------------------------------------------------
def _conv3x3_acc(x, wq, bq):
    """x int (N,C,H,W), wq (O,C,3,3) -> exact int64 (N,O,H,W). float64 GEMM is
    exact: every partial sum < 2^31 << 2^53."""
    N, C, H, W = x.shape
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.empty((N, C, 3, 3, H, W))
    for ky in range(3):
        for kx in range(3):
            cols[:, :, ky, kx] = xp[:, :, ky:ky + H, kx:kx + W]
    cols = cols.reshape(N, C * 9, H * W)
    acc = np.einsum("ok,nkp->nop", wq.reshape(wq.shape[0], -1).astype(np.float64), cols, optimize=True)
    acc = np.rint(acc).astype(np.int64) + bq[None, :, None]
    return acc.reshape(N, -1, H, W)


def requant(acc, M, sh, axis=1):
    shape = [1] * acc.ndim
    shape[axis] = -1
    Mv = M.reshape(shape).astype(np.int64)
    sv = sh.reshape(shape).astype(np.int64)
    y = (acc.astype(np.int64) * Mv + (np.int64(1) << (sv - 1))) >> sv
    return np.clip(y, 0, 255)


def pool2(y):
    N, C, H, W = y.shape
    return y.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


def forward_int(x_u8, Q, layers, batch=250, return_acts=False):
    N = x_u8.shape[0]
    logits = np.zeros((N, layers[-1]["cout"]), np.int64)
    acts = []
    for k in range(0, N, batch):
        x = x_u8[k:k + batch].transpose(0, 3, 1, 2).astype(np.int64)
        rec = []
        for q, L in zip(Q[:5], layers[:5]):
            x = requant(_conv3x3_acc(x, q["wq"], q["bq"]), q["M"], q["sh"])
            if L["pool"]:
                x = pool2(x)
            rec.append(x)
        v = x.reshape(x.shape[0], -1)                 # torch flatten order (c, y, x)
        for q, L in zip(Q[5:], layers[5:]):
            acc = np.rint(v.astype(np.float64) @ q["wq"].T.astype(np.float64)).astype(np.int64) + q["bq"]
            if L["raw"]:
                v = acc
            else:
                v = requant(acc, q["M"], q["sh"])
                rec.append(v)
        logits[k:k + batch] = v
        if return_acts:
            acts.append(rec)
    return (logits, acts) if return_acts else logits


def argmax_first(logits):
    return np.argmax(logits, axis=1)          # numpy returns the first maximal index


# --------------------------------------------------------------------------
# ROM images for the RTL
# --------------------------------------------------------------------------
def fc6_perm(src_c, src_hw):
    """kernel input element (word j, lane l) -> torch flatten index."""
    n = src_c * src_hw
    j = np.arange(n) // PI
    l = np.arange(n) % PI
    c = (j // src_hw) * PI + l
    return c * src_hw + (j % src_hw)


def weight_rom(Q, layers):
    """-> uint8 array (DEPTH, PO, PI) of int8 weights (two's complement)."""
    depth = sum(L["nt"] * L["cigkk"] for L in layers)
    rom = np.zeros((depth, PO, PI), np.int64)
    for q, L in zip(Q, layers):
        O, K = L["cout"], L["K"]
        if not L["fc"]:
            w = np.zeros((L["cout_p"], L["cin_p"], K, K), np.int64)
            w[:O, :L["cin"]] = q["wq"]
            # word (t, n=(cg*K+ky)*K+kx), lane o, input i -> w[t*PO+o, cg*PI+i, ky, kx]
            v = w.reshape(L["nt"], PO, L["cig"], PI, K, K).transpose(0, 2, 4, 5, 1, 3)
        else:
            wt = q["wq"]
            if "src_c" in L:
                wt = wt[:, fc6_perm(L["src_c"], L["src_hw"])]
            w = np.zeros((L["cout_p"], L["cin_p"]), np.int64)
            w[:O, :L["cin"]] = wt
            v = w.reshape(L["nt"], PO, L["cig"], PI).transpose(0, 2, 1, 3)
        rom[L["wbase"]:L["wbase"] + L["nt"] * L["cigkk"]] = v.reshape(-1, PO, PI)
    return (rom & 0xFF).astype(np.uint8)


def param_rom(Q, layers):
    """-> uint64 array (NTILES, PO): [31:0] bias, [47:32] M, [53:48] sh."""
    nt = sum(L["nt"] for L in layers)
    rom = np.zeros((nt, PO), np.uint64)
    for q, L in zip(Q, layers):
        b = np.zeros(L["cout_p"], np.int64)
        M = np.zeros(L["cout_p"], np.int64)
        sh = np.ones(L["cout_p"], np.int64)
        b[:L["cout"]], M[:L["cout"]], sh[:L["cout"]] = q["bq"], q["M"], q["sh"]
        word = (b & 0xFFFFFFFF) | (M << 32) | (sh << 48)
        rom[L["tbase"]:L["tbase"] + L["nt"]] = word.reshape(L["nt"], PO).astype(np.uint64)
    return rom


def write_wide_hex(path, rows_bytes):
    """rows_bytes: (DEPTH, NBYTES) uint8, byte 0 = bits [7:0]. One hex line per word, MSB first."""
    with open(path, "w") as f:
        for r in rows_bytes:
            f.write(r[::-1].tobytes().hex() + "\n")


def write_layer_header(path, layers):
    """Verilog include: function layer_cfg(idx) -> packed descriptor (see cnn_core.v)."""
    lines = ["// generated by model/export_rtl.py -- DO NOT EDIT",
             f"localparam NLAYERS   = {len(layers)};",
             f"localparam WROM_DEPTH = {sum(L['nt'] * L['cigkk'] for L in layers)};",
             f"localparam PROM_DEPTH = {sum(L['nt'] for L in layers)};",
             f"localparam NCLASSES  = {layers[-1]['cout']};",
             "// descriptor fields: {raw,pool,pad,k3,lgW[2:0],lgWo[2:0],cig[7:0],nt[3:0],cigkk[11:0],wbase[11:0],tbase[5:0]}",
             "function [63:0] layer_cfg;",
             "    input [3:0] idx;",
             "    begin",
             "        case (idx)"]
    for i, L in enumerate(layers):
        k3 = 1 if L["K"] == 3 else 0
        f = (f"{{1'b{L['raw']}, 1'b{L['pool']}, 1'b{L['pad']}, 1'b{k3}, 3'd{L['lgW']}, 3'd{L['lgWo']}, "
             f"8'd{L['cig']}, 4'd{L['nt']}, 12'd{L['cigkk']}, 12'd{L['wbase']}, 6'd{L['tbase']}}}")
        lines.append(f"        4'd{i}: layer_cfg = {{8'd0, {f}}};   // {L['name']}")
    lines += ["        default: layer_cfg = 64'd0;", "        endcase", "    end", "endfunction", ""]
    open(path, "w").write("\n".join(lines))


def image_words(img_u8):
    """(32,32,3) uint8 -> 1024 uint32 AXIS beats {8'h00, B, G, R}, raster order."""
    p = img_u8.reshape(-1, 3).astype(np.uint32)
    return p[:, 0] | (p[:, 1] << 8) | (p[:, 2] << 16)
