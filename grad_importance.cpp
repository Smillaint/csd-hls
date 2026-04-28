#include <iostream>
#include <cmath>
#include "grad_importance.h"

// ── L2 范数 ───────────────────────────────────────────────────
static fixed_t l2_norm(fixed_t g[GRAD_DIM]) {
#pragma HLS INLINE off
    fixed_t sum_sq = 0;
    NORM_LOOP:
    for (int k = 0; k < GRAD_DIM; k++) {
#pragma HLS PIPELINE II=1
        sum_sq += g[k] * g[k];
    }
    return (fixed_t)std::sqrt((float)sum_sq);
}

// ── 点积 ──────────────────────────────────────────────────────
static fixed_t dot_product(fixed_t a[GRAD_DIM], fixed_t b[GRAD_DIM]) {
#pragma HLS INLINE off
    fixed_t sum = 0;
    DOT_LOOP:
    for (int k = 0; k < GRAD_DIM; k++) {
#pragma HLS PIPELINE II=1
        sum += a[k] * b[k];
    }
    return sum;
}

// ── 主函数：计算余弦相似度分数 ───────────────────────────────
void grad_importance(fixed_t sample_grads[][GRAD_DIM],
                     fixed_t global_grad[GRAD_DIM],
                     fixed_t scores[MAX_SAMPLES],
                     int n_samples)
{
#pragma HLS INTERFACE m_axi port=sample_grads
#pragma HLS INTERFACE m_axi port=global_grad
#pragma HLS INTERFACE m_axi port=scores
#pragma HLS INTERFACE s_axilite port=n_samples
#pragma HLS INTERFACE s_axilite port=return

    fixed_t norm_g = l2_norm(global_grad);

    SAMPLE_LOOP:
    for (int i = 0; i < n_samples; i++) {
        fixed_t norm_s = l2_norm(sample_grads[i]);
        fixed_t dot    = dot_product(sample_grads[i], global_grad);

        fixed_t denom = norm_s * norm_g;
        if (denom == 0)
            scores[i] = 0;
        else
            scores[i] = dot / denom;
    }
}
