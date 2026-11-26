// ============================================================================
// tb_cifar_accel.cpp -- C simulation / C-RTL co-simulation testbench
//
//  1. negative test: a 37-beat frame (TLAST too early) must return ERR_FRAME
//  2. N CIFAR-10 test images (data/images.txt, written by model/export_hls.py)
//     through cifar_accel(); every logit and the class are compared
//     bit-exactly with the Python integer model (data/expected.txt).
//     Also reports top-1 against the true labels.
//  N = env CIFAR_N (default 20; use 2-3 for co-simulation).
// Exit code 0 = PASS.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cifar_accel.h"

#ifndef TB_DATA_DIR
#define TB_DATA_DIR "data"
#endif

static void push_frame(hls::stream<axis_t> &s, const std::vector<unsigned> &px, int len) {
    for (int i = 0; i < len; i++) {
        axis_t b;
        b.data = px[i];
        b.keep = 0xF;
        b.strb = 0xF;
        b.last = (i == len - 1);
        s.write(b);
    }
}

int main() {
    int N = 20;
    if (const char *e = getenv("CIFAR_N")) N = atoi(e);
    FILE *fi = fopen(TB_DATA_DIR "/images.txt", "r");
    FILE *fe = fopen(TB_DATA_DIR "/expected.txt", "r");
    if (!fi || !fe) {
        printf("missing " TB_DATA_DIR "/images.txt or expected.txt -- run model/export_hls.py\n");
        return 1;
    }
    int errors = 0, correct = 0, done = 0;
    int logits[NCLS];

    {   // ---- negative test --------------------------------------------------
        hls::stream<axis_t> s;
        std::vector<unsigned> px(IMG_PIX, 0);
        push_frame(s, px, 37);
        int r = cifar_accel(s, logits);
        if (r != ERR_FRAME) { printf("ERROR: short frame returned %d\n", r); errors++; }
        else printf("short frame -> ERR_FRAME  PASS\n");
    }

    for (int n = 0; n < N; n++) {
        std::vector<unsigned> px(IMG_PIX);
        int exp[NCLS], ecls, label;
        bool ok = true;
        for (int i = 0; i < IMG_PIX; i++) ok &= fscanf(fi, "%x", &px[i]) == 1;
        for (int i = 0; i < NCLS; i++) ok &= fscanf(fe, "%d", &exp[i]) == 1;
        ok &= fscanf(fe, "%d %d", &ecls, &label) == 2;
        if (!ok) break;
        hls::stream<axis_t> s;
        push_frame(s, px, IMG_PIX);
        int cls = cifar_accel(s, logits);
        int bad = 0;
        for (int i = 0; i < NCLS; i++) bad += (logits[i] != exp[i]);
        if (bad || cls != ecls) {
            errors++;
            printf("img %d: MISMATCH class %d (model %d), %d logits differ\n", n, cls, ecls, bad);
        }
        correct += (cls == label);
        done++;
        if (n < 3) printf("img %d: class %d label %d  logits[0..2] %d %d %d\n", n, cls, label,
                          logits[0], logits[1], logits[2]);
    }
    fclose(fi);
    fclose(fe);
    printf("images %d, mismatches vs integer model %d, top-1 %d/%d\n", done, errors, correct, done);
    printf("%s\n", (errors == 0 && done > 0) ? "TEST PASSED" : "TEST FAILED");
    return (errors == 0 && done > 0) ? 0 : 1;
}
