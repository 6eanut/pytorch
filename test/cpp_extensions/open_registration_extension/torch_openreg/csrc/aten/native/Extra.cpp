#include "Extra.h"

namespace at::native::openreg {

at::Tensor quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype) {
  return at::native::quantize_per_tensor(self, scale, zero_point, dtype);
}

int64_t _fused_sdp_choice(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  auto backend = sdp::SDPBackend::overrideable;
  return static_cast<int64_t>(backend);
}

void quantize_tensor_per_tensor_affine_stub(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    double scale,
    int64_t zero_point) {}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
_scaled_dot_product_fused_attention_overrideable(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim_v = value.size(3);
  const int64_t max_seqlen_q = query.size(2);
  const int64_t max_seqlen_kv = key.size(2);

  auto opts = query.options();
  auto output =
      at::empty({batch_size, num_heads, max_seqlen_q, head_dim_v}, opts);
  auto logsumexp =
      at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
  auto debug_attn_mask = at::empty(
      {batch_size, num_heads, max_seqlen_q, max_seqlen_kv},
      opts.dtype(at::kFloat));
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));

  return std::make_tuple(
      output,
      logsumexp,
      at::Tensor(),
      at::Tensor(),
      max_seqlen_q,
      max_seqlen_kv,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_fused_attention_overrideable_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale) {
  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
      at::empty_like(query),
      at::empty_like(key),
      at::empty_like(value),
      at::empty_like(attn_bias));
}

namespace {
struct CustomAutogradFnReturnsSelf
    : public torch::autograd::Function<CustomAutogradFnReturnsSelf> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};

struct CustomAutogradFnAliasing
    : public torch::autograd::Function<CustomAutogradFnAliasing> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self.view_symint(self.sym_sizes());
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};
} // namespace

at::Tensor custom_autograd_fn_returns_self(at::Tensor x) {
  return CustomAutogradFnReturnsSelf::apply(x);
}

at::Tensor custom_autograd_fn_aliasing(at::Tensor x) {
  return CustomAutogradFnAliasing::apply(x);
}

/*
 This implementation is only used to test stub registration, so not all
 capabilities are fully supported.

 Current Limitations:
 - dtype: Float only
 - input tensor: must be contiguous layout
*/
// LITERALINCLUDE START: STUB ABS
void abs_kernel(at::TensorIteratorBase& iter) {
  TORCH_CHECK(iter.ntensors() == 2, "Abs kernel expects 2 tensors");
  TORCH_CHECK(
      iter.common_dtype() == at::ScalarType::Float,
      "Abs kernel only supports float type");

  auto& output_tensor = iter.tensor(0);
  auto& input_tensor = iter.tensor(1);

  TORCH_CHECK(
      input_tensor.sizes() == output_tensor.sizes(),
      "Input and output tensor sizes must match.");

  auto abs_loop = [](float* out_ptr, const float* in_ptr, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = std::abs(in_ptr[i]);
    }
  };

  MemoryGuard guard(input_tensor, output_tensor);

  if (iter.is_contiguous()) {
    abs_loop(
        static_cast<float*>(iter.data_ptr(0)),
        static_cast<float*>(iter.data_ptr(1)),
        iter.numel());
  } else {
    TORCH_CHECK(
        input_tensor.is_contiguous(), "Input tensor must be contiguous.")

    auto output = at::empty(
        input_tensor.sizes(),
        input_tensor.options().memory_format(
            input_tensor.suggest_memory_format()));

    MemoryGuard guard(output);

    abs_loop(
        static_cast<float*>(output.data_ptr()),
        static_cast<float*>(iter.data_ptr(1)),
        iter.numel());

    output_tensor.copy_(output);
  }
}
// LITERALINCLUDE END: STUB ABS

at::Tensor& abs_out(const at::Tensor& self, at::Tensor& out) {
  return at::native::abs_out(self, out);
}

at::Tensor custom_abs(at::Tensor x) {
  return at::abs(x) + 1;
}

// Helper: multiply two 2x2 tensors
at::Tensor multiply_2x2_blocks(const at::Tensor& A_block, const at::Tensor& B_block) {
    TORCH_CHECK(A_block.sizes() == c10::IntArrayRef({2, 2}), "A_block must be 2x2");
    TORCH_CHECK(B_block.sizes() == c10::IntArrayRef({2, 2}), "B_block must be 2x2");

    std::cout << "\n  ┌─ [multiply_2x2_blocks] 进入函数" << std::endl;
    std::cout << "  │  输入 A_block (2x2):\n";
    for (int i = 0; i < 2; ++i) {
        std::cout << "  │    [";
        for (int j = 0; j < 2; ++j)
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                      << A_block[i][j].item<float>() << (j < 1 ? "," : "");
        std::cout << " ]\n";
    }
    std::cout << "  │  输入 B_block (2x2):\n";
    for (int i = 0; i < 2; ++i) {
        std::cout << "  │    [";
        for (int j = 0; j < 2; ++j)
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                      << B_block[i][j].item<float>() << (j < 1 ? "," : "");
        std::cout << " ]\n";
    }

    auto options = A_block.options();
    at::Tensor C_block = at::empty({2, 2}, options);

    std::cout << "  │  开始逐元素点积计算 C = A × B:" << std::endl;
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 2; ++j) {
            float sum = 0.0f;
            std::cout << "  │    C[" << i << "," << j << "] = ";
            for (int64_t k = 0; k < 2; ++k) {
                float a_val = A_block[i][k].item<float>();
                float b_val = B_block[k][j].item<float>();
                float contrib = a_val * b_val;
                sum += contrib;
                std::cout << "A[" << i << "," << k << "](" << a_val << ")"
                          << " * B[" << k << "," << j << "](" << b_val << ")"
                          << " = " << contrib;
                if (k < 1) std::cout << "  +  ";
            }
            C_block[i][j] = sum;
            std::cout << "  =>  sum = " << sum << std::endl;
        }
    }

    std::cout << "  │  输出 C_block (2x2):\n";
    for (int i = 0; i < 2; ++i) {
        std::cout << "  │    [";
        for (int j = 0; j < 2; ++j)
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                      << C_block[i][j].item<float>() << (j < 1 ? "," : "");
        std::cout << " ]\n";
    }
    std::cout << "  └─ [multiply_2x2_blocks] 返回\n" << std::endl;

    return C_block;
}

// Custom 2x2 block matrix multiplication with print debug
at::Tensor custom_matrix_multiply(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix inner dimensions must match");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(M % 2 == 0 && K % 2 == 0 && N % 2 == 0,
                "Currently only support multiples of 2");

    // ── 打印整体输入信息 ──────────────────────────────────────────
    std::cout << "╔══════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        [custom_matrix_multiply] 开始执行          ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════╝" << std::endl;
    std::cout << "输入矩阵 A，形状: [" << M << " x " << K << "]，内容:\n";
    for (int i = 0; i < M; ++i) {
        std::cout << "  [";
        for (int j = 0; j < K; ++j)
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                      << A[i][j].item<float>() << (j < K-1 ? "," : "");
        std::cout << " ]\n";
    }
    std::cout << "输入矩阵 B，形状: [" << K << " x " << N << "]，内容:\n";
    for (int i = 0; i < K; ++i) {
        std::cout << "  [";
        for (int j = 0; j < N; ++j)
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                      << B[i][j].item<float>() << (j < N-1 ? "," : "");
        std::cout << " ]\n";
    }

    // ── 分块策略说明 ──────────────────────────────────────────────
    std::cout << "\n>>> 分块策略: 将 A[" << M << "x" << K << "] 和 B["
              << K << "x" << N << "] 均以 2x2 为单元进行分块矩阵乘法" << std::endl;
    std::cout << "    C[" << M << "x" << N << "] 共需计算 "
              << (M/2) << " x " << (N/2) << " = " << (M/2)*(N/2)
              << " 个输出块，每个输出块由 " << K/2 << " 对 A/B 子块累加得到\n" << std::endl;

    auto options = A.options();
    at::Tensor C = at::empty({M, N}, options);

    for (int64_t i = 0; i < M; i += 2) {
        for (int64_t j = 0; j < N; j += 2) {

            std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
            std::cout << "▶ 计算输出块 C[" << i << ":" << i+2
                      << ", " << j << ":" << j+2 << "]" << std::endl;
            std::cout << "  初始化 C_block = zeros(2x2)，准备对 k 方向累加" << std::endl;

            at::Tensor C_block = at::zeros({2, 2}, options);

            for (int64_t k = 0; k < K; k += 2) {
                auto A_block = A.slice(0, i, i+2).slice(1, k, k+2);
                auto B_block = B.slice(0, k, k+2).slice(1, j, j+2);

                std::cout << "\n  [k=" << k << "] 第 " << k/2+1 << "/" << K/2
                          << " 次累加：" << std::endl;
                std::cout << "    取 A_block = A[" << i << ":" << i+2
                          << ", " << k << ":" << k+2 << "]" << std::endl;
                std::cout << "    取 B_block = B[" << k << ":" << k+2
                          << ", " << j << ":" << j+2 << "]" << std::endl;
                std::cout << "    → 调用 multiply_2x2_blocks(A_block, B_block)" << std::endl;

                at::Tensor partial = multiply_2x2_blocks(A_block, B_block);

                std::cout << "    ← multiply_2x2_blocks 返回 partial (2x2):\n";
                for (int pi = 0; pi < 2; ++pi) {
                    std::cout << "        [";
                    for (int pj = 0; pj < 2; ++pj)
                        std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                                  << partial[pi][pj].item<float>() << (pj < 1 ? "," : "");
                    std::cout << " ]\n";
                }

                C_block += partial;

                std::cout << "    C_block 累加后 (当前中间值):\n";
                for (int pi = 0; pi < 2; ++pi) {
                    std::cout << "        [";
                    for (int pj = 0; pj < 2; ++pj)
                        std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                                  << C_block[pi][pj].item<float>() << (pj < 1 ? "," : "");
                    std::cout << " ]\n";
                }
            }

            std::cout << "\n  ✔ C_block 累加完毕，最终值写回 C["
                      << i << ":" << i+2 << ", " << j << ":" << j+2 << "]:\n";
            for (int pi = 0; pi < 2; ++pi) {
                std::cout << "      [";
                for (int pj = 0; pj < 2; ++pj)
                    std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                              << C_block[pi][pj].item<float>() << (pj < 1 ? "," : "");
                std::cout << " ]\n";
            }
            C.slice(0, i, i+2).slice(1, j, j+2).copy_(C_block);
        }
    }

    std::cout << "\n╔══════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     [custom_matrix_multiply] 全部块计算完毕        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════╝" << std::endl;
    std::cout << "输出矩阵 C，形状: [" << M << " x " << N << "]，内容:\n";
    for (int i = 0; i < M; ++i) {
        std::cout << "  [";
        for (int j = 0; j < N; ++j)
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                      << C[i][j].item<float>() << (j < N-1 ? "," : "");
        std::cout << " ]\n";
    }

    return C;
}

// // Helper: multiply two 2x2 tensors
// at::Tensor multiply_2x2_blocks(const at::Tensor& A_block, const at::Tensor& B_block) {
//     TORCH_CHECK(A_block.sizes() == c10::IntArrayRef({2, 2}), "A_block must be 2x2");
//     TORCH_CHECK(B_block.sizes() == c10::IntArrayRef({2, 2}), "B_block must be 2x2");

//     auto options = A_block.options();
//     at::Tensor C_block = at::empty({2, 2}, options);

//     for (int64_t i = 0; i < 2; ++i) {
//         for (int64_t j = 0; j < 2; ++j) {
//             float sum = 0.0f;
//             for (int64_t k = 0; k < 2; ++k) {
//                 sum += A_block[i][k].item<float>() * B_block[k][j].item<float>();
//             }
//             C_block[i][j] = sum;
//             std::cout << "C_block[" << i << "," << j << "] = " << sum << std::endl;
//         }
//     }
//     return C_block;
// }

// // Custom 2x2 block matrix multiplication with print debug
// at::Tensor custom_matrix_multiply(const at::Tensor& A, const at::Tensor& B) {
//     TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
//     TORCH_CHECK(A.size(1) == B.size(0), "Matrix inner dimensions must match");

//     int64_t M = A.size(0);
//     int64_t K = A.size(1);
//     int64_t N = B.size(1);

//     TORCH_CHECK(M % 2 == 0 && K % 2 == 0 && N % 2 == 0,
//                 "Currently only support multiples of 2");

//     auto options = A.options();
//     at::Tensor C = at::empty({M, N}, options);

//     std::cout << "Starting custom_matrix_multiply with shape: "
//               << A.sizes() << " x " << B.sizes() << std::endl;

//     for (int64_t i = 0; i < M; i += 2) {
//         for (int64_t j = 0; j < N; j += 2) {
//             at::Tensor C_block = at::zeros({2, 2}, options);  // accumulate sum over k-blocks
//             for (int64_t k = 0; k < K; k += 2) {
//                 auto A_block = A.slice(0, i, i+2).slice(1, k, k+2);
//                 auto B_block = B.slice(0, k, k+2).slice(1, j, j+2);

//                 std::cout << "Multiplying block A(" << i << ":" << i+2
//                           << "," << k << ":" << k+2 << ") x B("
//                           << k << ":" << k+2 << "," << j << ":" << j+2
//                           << ")" << std::endl;

//                 // multiply 2x2 blocks using helper
//                 C_block += multiply_2x2_blocks(A_block, B_block);
//             }
//             // write the result block back to C
//             C.slice(0, i, i+2).slice(1, j, j+2).copy_(C_block);
//         }
//     }

//     return C;
// }

} // namespace at::native::openreg
