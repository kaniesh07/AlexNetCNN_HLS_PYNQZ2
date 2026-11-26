// Minimal stand-in for Vitis HLS <ap_axi_sdata.h>, used ONLY by Makefile.gcc.
#ifndef AP_AXI_SDATA_STUB_H
#define AP_AXI_SDATA_STUB_H
#include <ap_int.h>
template <int D, int U, int TI, int TD> struct ap_axiu {
    ap_uint<D> data;
    ap_uint<(D + 7) / 8> keep;
    ap_uint<(D + 7) / 8> strb;
    ap_uint<(U > 0) ? U : 1> user;
    ap_uint<1> last;
    ap_uint<(TI > 0) ? TI : 1> id;
    ap_uint<(TD > 0) ? TD : 1> dest;
};
#endif
