#ifndef GRAD_IMPORTANCE_H
#define GRAD_IMPORTANCE_H

#include <ap_fixed.h>

// ================================================================
// SimpleCNN 参数总量 = 80202
// ================================================================
#define GRAD_DIM    80202   // ← 已确认
#define MAX_SAMPLES 100     // 与 Python SPEED 一致
#define TOP_K       20      // 与 Python BUFFER_SIZE 一致

// 整数4位，小数28位
typedef ap_fixed<32, 4> fixed_t;

void grad_importance(fixed_t sample_grads[][GRAD_DIM],
                     fixed_t global_grad[GRAD_DIM],
                     fixed_t scores[MAX_SAMPLES],
                     int n_samples);

#endif

