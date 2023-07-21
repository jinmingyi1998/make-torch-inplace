//
// Created by jimmy on 22-6-11.
//
#include <torch/extension.h>

void square_matmul_inplace_(at::Tensor input, at::Tensor square_mat, int rows,
                            int cols);

void square_matmul_inplace_T_(at::Tensor input, at::Tensor square_mat, int rows,
                              int cols);

void attn_softmax_inplace_forward_(at::Tensor input, long long rows, int cols);
void attn_softmax_inplace_backward_(at::Tensor output, at::Tensor d_ov,
                                    at::Tensor values, long long rows,
                                    int cols_output, int cols_values);
void layernorm_inplace_forward_(at::Tensor input, long long rows, int cols);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("square_matmul", &square_matmul_inplace_, "Matmal inplace (CUDA)");
  m.def("square_matmul_transposed", &square_matmul_inplace_T_,
        "Matmal inplace transposed(CUDA)");
  m.def("softmax_forward_", &attn_softmax_inplace_forward_,
        "Softmax forward (CUDA)");
  m.def("softmax_backward_", &attn_softmax_inplace_backward_,
        "Softmax backward (CUDA)");
  m.def("layernorm_", &layernorm_inplace_forward_, "Layer Norm(CUDA)");
}