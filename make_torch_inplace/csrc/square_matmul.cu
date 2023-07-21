//
// Created by jimmy on 22-6-11.
//
#include <c10/cuda/CUDAGuard.h>
#include <math_constants.h>
#include <torch/extension.h>

#include <iostream>

#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "compat.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define ROW_PER_BLOCK 4

template <typename T, typename T_W>
__global__ void square_matmul_inplace_kernel_(T *mat, T_W *square_mat, int rows,
                                              int cols, bool transpose_square) {
  int lane_id = threadIdx.x % 32;
  int C_row = blockIdx.x * ROW_PER_BLOCK + threadIdx.x / 32;

  int cols_per_thread = (cols + 31) / 32;
  int cols_this_thread = cols_per_thread;
  int last_y = cols / cols_per_thread;

  if (lane_id == last_y) {
    cols_this_thread = cols - cols_per_thread * last_y;
  } else if (lane_id > last_y) {
    cols_this_thread = 0;
  }
  T *row_input = mat + C_row * cols;
  float thread_sum_buf[64];
  if (C_row < rows) {
    for (int i = 0; i < cols_this_thread; i++) {
      int C_col = lane_id * cols_per_thread + i;
      float thread_sum = 0.0;
      for (int j = 0; j < cols; j++) {
        int mat2_idx;
        if (transpose_square) {
          mat2_idx = C_col * cols + j;
        } else {
          mat2_idx = j * cols + C_col;
        }
        float v1 = static_cast<float>(row_input[j]);
        float v2 = static_cast<float>(square_mat[mat2_idx]);
        thread_sum += v1 * v2;
      }
      thread_sum_buf[i] = thread_sum;
    }
  }
  __syncthreads();
  if (C_row < rows) {
    for (int i = 0; i < cols_this_thread; i++) {
      int C_col = lane_id * cols_per_thread + i;
      row_input[C_col] = static_cast<T>(thread_sum_buf[i]);
    }
  }
}

void square_matmul_inplace_core(at::Tensor mat, at::Tensor square_mat, int rows,
                                int cols, bool transpose_square) {
  CHECK_INPUT(mat);
  const at::cuda::OptionalCUDAGuard device_guard_a(device_of(mat));
  CHECK_INPUT(square_mat);
  const at::cuda::OptionalCUDAGuard device_guard_b(device_of(square_mat));

  int grid = (rows + ROW_PER_BLOCK - 1) / ROW_PER_BLOCK;
  dim3 block(ROW_PER_BLOCK * 32);
  if (mat.dtype() == torch::kFloat32) {
    if (square_mat.dtype() == torch::kFloat32) {
      square_matmul_inplace_kernel_<float, float><<<grid, block>>>(
          (float *)mat.data_ptr(), (float *)square_mat.data_ptr(), rows, cols,
          transpose_square);
    } else {
      square_matmul_inplace_kernel_<float, c10::Half><<<grid, block>>>(
          (float *)mat.data_ptr(), (c10::Half *)square_mat.data_ptr(), rows,
          cols, transpose_square);
    }
  } else if (mat.dtype() == torch::kFloat16) {
    if (square_mat.dtype() == torch::kFloat32) {
      square_matmul_inplace_kernel_<c10::Half, float><<<grid, block>>>(
          (c10::Half *)mat.data_ptr(), (float *)square_mat.data_ptr(), rows,
          cols, transpose_square);
    } else {
      square_matmul_inplace_kernel_<c10::Half, c10::Half><<<grid, block>>>(
          (c10::Half *)mat.data_ptr(), (c10::Half *)square_mat.data_ptr(), rows,
          cols, transpose_square);
    }
  } else {
    if (square_mat.dtype() != torch::kBFloat16) {
      printf("[WARN] input tensor is BFloat16 but weight tensor not.");
    }
    square_matmul_inplace_kernel_<c10::BFloat16, c10::BFloat16>
        <<<grid, block>>>((c10::BFloat16 *)mat.data_ptr(),
                          (c10::BFloat16 *)square_mat.data_ptr(), rows, cols,
                          transpose_square);
  }
}

void square_matmul_inplace_(at::Tensor input, at::Tensor square_mat, int rows,
                            int cols) {
  square_matmul_inplace_core(input, square_mat, rows, cols, false);
}
void square_matmul_inplace_T_(at::Tensor input, at::Tensor square_mat, int rows,
                              int cols) {
  square_matmul_inplace_core(input, square_mat, rows, cols, true);
}