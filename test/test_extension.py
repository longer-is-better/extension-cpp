import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import extension_cpp
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
import torch.nn as nn

torch.set_printoptions(threshold=float('inf'))
torch.set_printoptions(linewidth=200)
torch.set_printoptions(sci_mode=True)

import torch

def flashattention_2_single_head(Q, K, V, mask=None, eps=1e-10):
    """
    FlashAttention-2 单头简化模拟实现

    Args:
        Q: (N, D)
        K: (M, D)
        V: (M, D)
        mask: (N, M) 或 None
        block_size: 分块大小

    Returns:
        O: (N, D)
    """
    N, D = Q.shape
    M = K.shape[0]

    # 缩放 Q
    scale = D ** -0.5
    Q_scaled = Q * scale

    # 初始化输出和归一化变量
    O = torch.zeros_like(Q)                # (N, D)
    l = torch.zeros(N, device=Q.device)    # 归一化因子 l_i
    m = torch.full((N,), fill_value=-float('inf'), device=Q.device)  # 当前行最大值 m_i

    # 遍历 Q 的行块
    for i in range(0, N, 4):
        i_end = min(i + 4, N)
        Q_block = Q_scaled[i:i_end]  # (block_N, D)
        block_N = Q_block.size(0)

        # 初始化块级变量
        O_block = torch.zeros(block_N, D, device=Q.device)
        l_block = torch.zeros(block_N, device=Q.device)
        m_block = torch.full((block_N,), -float('inf'), device=Q.device)

        # 遍历 K, V 的列块
        for j in range(0, M, 32):
            j_end = min(j + 32, M)
            K_block = K[j:j_end]  # (block_M, D)
            V_block = V[j:j_end]  # (block_M, D)

            # 计算 S_block = Q_block @ K_block^T
            S = torch.matmul(Q_block, K_block.transpose(-1, -2))  # (block_N, block_M)

            if mask is not None:
                mask_block = mask[i:i_end, j:j_end]
                S = S.masked_fill(mask_block == 0, float('-inf'))

            # === 在线 Softmax 更新 ===
            # 1. 获取当前块的最大值
            S_max = S.max(dim=-1).values  # (block_N,)
            new_max = torch.maximum(m_block, S_max)

            # 2. 调整指数项：exp(S - new_max)
            exp_S = torch.exp(S - new_max.unsqueeze(-1))  # (block_N, block_M)

            # 3. 更新归一化因子
            # l_block = l_block * exp(m_block - new_max) + sum(exp_S, dim=-1)
            l_block = l_block * torch.exp(m_block - new_max) + exp_S.sum(dim=-1)

            # 4. 更新输出
            # 先调整旧输出：O_block *= exp(m_block - new_max).unsqueeze(-1)
            O_block = O_block * torch.exp(m_block - new_max).unsqueeze(-1)
            # 加上新贡献
            O_block = O_block + torch.matmul(exp_S, V_block)

            # 5. 更新 m_block
            m_block = new_max

        # 存储结果
        O[i:i_end] = O_block
        l[i:i_end] = l_block
        m[i:i_end] = m_block

    # 最终归一化：O[i] /= l[i]
    # 防止除以 0
    l = l.clamp(min=eps)
    O = O / l.unsqueeze(-1)
    return O


def reference_muladd(a, b, c):
    return a * b + c


def reference_attention(q, k, v):
    k = k.T
    S = torch.matmul(q, k)
    P = torch.softmax( S / torch.sqrt(torch.Tensor([q.shape[1]]).to("cuda")),
            dim=-1
        )
    return torch.matmul(P, v)

class Testflashattention(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor_rcsum(*size):
            rows, cols = size
            i = torch.arange(rows, dtype=torch.float).unsqueeze(1)  # 列向量 [rows, 1]
            j = torch.arange(cols, dtype=torch.float).unsqueeze(0)  # 行向量 [1, cols]
            result = i + 100*j  # 广播相加，得到 [rows, cols]
            
            # result[2:, :] = 0
            return result.to(device)

        def make_tensor_r0(*size):
            t = torch.randn(size, device=device, requires_grad=requires_grad)
            t[:, 1:] = -999
            return t

        def make_tensor_d0(*size):
            t = torch.randn(size, device=device, requires_grad=requires_grad)
            t[0, :] = 0
            t[2:, :] = 0
            return t

        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)
            # return torch.ones(size, device=device, requires_grad=requires_grad)

        def make_tensor1(*size):
            return torch.ones(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)



        def make_tensor_q0(*size):
            rows, cols = size
            i = torch.arange(rows, dtype=torch.float).unsqueeze(1)  # 列向量 [rows, 1]
            j = torch.arange(cols, dtype=torch.float).unsqueeze(0)  # 行向量 [1, cols]
            t = 0.1 *i + 0.01* j  # 广播相加，得到 [rows, cols]
            t= t.to(device)
            # t = torch.randn(size, device=device, requires_grad=requires_grad)
            # t = torch.ones(size, device=device, requires_grad=requires_grad)
            t[:32, :] = 0
            t[32, 4:] = 0
            t[33:, :] = 0

            return t

        def make_tensor_k0(*size):
            rows, cols = size
            i = torch.arange(rows, dtype=torch.float).unsqueeze(1)  # 列向量 [rows, 1]
            j = torch.arange(cols, dtype=torch.float).unsqueeze(0)  # 行向量 [1, cols]
            res = 0.1* i + 0.05*j  # 广播相加，得到 [rows, cols]
            res = res.to(device)
            # res = torch.randn(size, device=device, requires_grad=requires_grad)
            # res = torch.ones(size, device=device, requires_grad=requires_grad)
            res[:32, :] = 0
            res[32, 30:] = 0
            res[33:, :] = 0
            return res

        def make_tensor_v0(*size):
            rows, cols = size
            i = torch.arange(rows, dtype=torch.float).unsqueeze(1)  # 列向量 [rows, 1]
            j = torch.arange(cols, dtype=torch.float).unsqueeze(0)  # 行向量 [1, cols]
            result = i + j  # 广播相加，得到 [rows, cols]
            
            result[0, 2:] = 0
            result[1:, :] = 0
            return result.to(device)
            # return torch.randn(size, device=device, requires_grad=requires_grad)


        def make_tensor_q1(*size):
            rows, cols = size
            i = torch.arange(rows, dtype=torch.float).unsqueeze(1) + 1 # 列向量 [rows, 1]
            j = torch.arange(cols, dtype=torch.float).unsqueeze(0) + 1  # 行向量 [1, cols]
            t = 0.1 *i + 0.01* j  # 广播相加，得到 [rows, cols]
            t= t.to(device)
            # t = torch.randn(size, device=device, requires_grad=requires_grad)
            # t = torch.ones(size, device=device, requires_grad=requires_grad)
            t[4:, :] = 0
            # t[32, 4:] = 0
            # t[33:, :] = 0

            return t

        def make_tensor_k1(*size):
            rows, cols = size
            i = torch.arange(rows, dtype=torch.float).unsqueeze(1)+1  # 列向量 [rows, 1]
            j = torch.arange(cols, dtype=torch.float).unsqueeze(0)+1  # 行向量 [1, cols]
            res = 0.1* i + 0.05*j  # 广播相加，得到 [rows, cols]
            res = res.to(device)
            # res = torch.randn(size, device=device, requires_grad=requires_grad)
            # res = torch.ones(size, device=device, requires_grad=requires_grad)
            res[:, 1:] = 0
            res[:32, 0] = 0
            res[33:, 0] = 0
            # res[32, 30:] = 0
            # res[33:, :] = 0
            return res

        def make_tensor_v1(*size):
            rows, cols = size
            i = torch.arange(rows, dtype=torch.float).unsqueeze(1) + 1  # 列向量 [rows, 1]
            j = torch.arange(cols, dtype=torch.float).unsqueeze(0) + 1  # 行向量 [1, cols]
            result = i + j  # 广播相加，得到 [rows, cols]
            
            result[0, 2:] = 0
            result[1:, :] = 0
            return result.to(device)
            # return torch.randn(size, device=device, requires_grad=requires_grad)


        def make_tensor_q(*size):
            t = torch.randn(size, device=device, requires_grad=requires_grad)
            t[0, :] = 0
            t[1, :] = 100*torch.arange(32)
            t[2:, :] = 0
            return t

        def make_tensor_k(*size):
            rows, cols = size
            res = torch.arange(rows, dtype=torch.float).unsqueeze(1).repeat(1, cols)
            return res.to(device)
            # return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_tensor_v(*size):
            rows, cols = size
            i = torch.arange(rows, dtype=torch.float).unsqueeze(1)  # 列向量 [rows, 1]
            j = torch.arange(cols, dtype=torch.float).unsqueeze(0)  # 行向量 [1, cols]
            result = i + 100*j  # 广播相加，得到 [rows, cols]
            
            # result[2:, :] = 0
            return result.to(device)



        return [
            # [make_tensor_rcsum(32, 32), make_tensor_rcsum(32, 32), make_tensor_rcsum(32, 32)],
            # [make_tensor(32, 32), make_tensor(32, 32), make_tensor(32, 32)],
            # [make_tensor(32, 64), make_tensor(32, 64), make_tensor(32, 64)],
            # [make_tensor_q0(64, 32), make_tensor_k0(64, 32), make_tensor_v0(64, 32)],
            # [make_tensor_q1(64, 32), make_tensor_k1(64, 32), make_tensor_v1(64, 32)],
            # [make_tensor_q(64, 32), make_tensor_k(64, 32), make_tensor_v(64, 32)],
            # [make_tensor_d0(64, 32), make_tensor(64, 32), make_tensor(64, 32)],
            # [make_tensor(64, 32), make_tensor(64, 32), make_tensor1(64, 32)],
            # [make_tensor(64, 32), make_tensor(64, 32), make_tensor(64, 32)],
            # [make_tensor(32, 64), make_tensor(32, 64), make_tensor_r0(32, 64)],
            # [make_tensor(64, 64), make_tensor(64, 64), make_tensor1(64, 64)],
            [make_tensor(512, 32), make_tensor(512, 32), make_tensor(512, 32)],
            # [make_tensor(32, 32), make_tensor(32, 32), make_tensor(32, 32)],
            # [make_tensor(512, 128), make_tensor(512, 128), make_tensor(512, 128)],
            # [make_tensor(1024, 64), make_tensor(1024, 64), make_tensor(1024, 64),],
        ]

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            expected = reference_attention(*args)
            # fa_python = flashattention_2_single_head(*args)
            result = extension_cpp.ops.flashattention2(*args)
            print(f"expected {expected[:3]} \n result {result[:3]}")
            torch.testing.assert_close(result, expected)
            # torch.testing.assert_close(fa_python, expected)

    # def test_correctness_cpu(self):
    #     self._test_correctness("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")

    # def _test_gradients(self, device):
    #     samples = self.sample_inputs(device, requires_grad=True)
    #     for args in samples:
    #         diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
    #         out = extension_cpp.ops.flashattention(*args)
    #         grad_out = torch.randn_like(out)
    #         result = torch.autograd.grad(out, diff_tensors, grad_out)

    #         out = reference_muladd(*args)
    #         expected = torch.autograd.grad(out, diff_tensors, grad_out)

    #         torch.testing.assert_close(result, expected)

    # def test_gradients_cpu(self):
    #     self._test_gradients("cpu")

    # @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    # def test_gradients_cuda(self):
    #     self._test_gradients("cuda")

    # def _opcheck(self, device):
    #     # Use opcheck to check for incorrect usage of operator registration APIs
    #     samples = self.sample_inputs(device, requires_grad=True)
    #     samples.extend(self.sample_inputs(device, requires_grad=False))
    #     for args in samples:
    #         opcheck(torch.ops.extension_cpp.flashattention.default, args)

    # # def test_opcheck_cpu(self):
    # #     self._opcheck("cpu")

    # @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    # def test_opcheck_cuda(self):
    #     self._opcheck("cuda")


# class TestMyMulAdd(TestCase):
#     def sample_inputs(self, device, *, requires_grad=False):
#         def make_tensor(*size):
#             return torch.randn(size, device=device, requires_grad=requires_grad)

#         def make_nondiff_tensor(*size):
#             return torch.randn(size, device=device, requires_grad=False)

#         return [
#             [make_tensor(3), make_tensor(3), 1],
#             [make_tensor(20), make_tensor(20), 3.14],
#             [make_tensor(20), make_nondiff_tensor(20), -123],
#             [make_nondiff_tensor(2, 3), make_tensor(2, 3), -0.3],
#         ]

#     def _test_correctness(self, device):
#         samples = self.sample_inputs(device)
#         for args in samples:
#             result = extension_cpp.ops.mymuladd(*args)
#             expected = reference_muladd(*args)
#             torch.testing.assert_close(result, expected)

#     def test_correctness_cpu(self):
#         self._test_correctness("cpu")

#     @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
#     def test_correctness_cuda(self):
#         self._test_correctness("cuda")

#     def _test_gradients(self, device):
#         samples = self.sample_inputs(device, requires_grad=True)
#         for args in samples:
#             diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
#             out = extension_cpp.ops.mymuladd(*args)
#             grad_out = torch.randn_like(out)
#             result = torch.autograd.grad(out, diff_tensors, grad_out)

#             out = reference_muladd(*args)
#             expected = torch.autograd.grad(out, diff_tensors, grad_out)

#             torch.testing.assert_close(result, expected)

#     def test_gradients_cpu(self):
#         self._test_gradients("cpu")

#     @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
#     def test_gradients_cuda(self):
#         self._test_gradients("cuda")

#     def _opcheck(self, device):
#         # Use opcheck to check for incorrect usage of operator registration APIs
#         samples = self.sample_inputs(device, requires_grad=True)
#         samples.extend(self.sample_inputs(device, requires_grad=False))
#         for args in samples:
#             opcheck(torch.ops.extension_cpp.mymuladd.default, args)

#     def test_opcheck_cpu(self):
#         self._opcheck("cpu")

#     @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
#     def test_opcheck_cuda(self):
#         self._opcheck("cuda")


# class TestMyAddOut(TestCase):
#     def sample_inputs(self, device, *, requires_grad=False):
#         def make_tensor(*size):
#             return torch.randn(size, device=device, requires_grad=requires_grad)

#         def make_nondiff_tensor(*size):
#             return torch.randn(size, device=device, requires_grad=False)

#         return [
#             [make_tensor(3), make_tensor(3), make_tensor(3)],
#             [make_tensor(20), make_tensor(20), make_tensor(20)],
#         ]

#     def _test_correctness(self, device):
#         samples = self.sample_inputs(device)
#         for args in samples:
#             result = args[-1]
#             extension_cpp.ops.myadd_out(*args)
#             expected = torch.add(*args[:2])
#             torch.testing.assert_close(result, expected)

#     def test_correctness_cpu(self):
#         self._test_correctness("cpu")

#     @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
#     def test_correctness_cuda(self):
#         self._test_correctness("cuda")

#     def _opcheck(self, device):
#         # Use opcheck to check for incorrect usage of operator registration APIs
#         samples = self.sample_inputs(device, requires_grad=True)
#         samples.extend(self.sample_inputs(device, requires_grad=False))
#         for args in samples:
#             opcheck(torch.ops.extension_cpp.myadd_out.default, args)

#     def test_opcheck_cpu(self):
#         self._opcheck("cpu")

#     @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
#     def test_opcheck_cuda(self):
#         self._opcheck("cuda")


# class TestTorchCompileStreamSync(TestCase):
#     """Test for GitHub issue pytorch/pytorch#157363 - stream synchronization with torch.compile"""
    
#     @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
#     def test_compile_with_linear_layer(self):
#         """Test custom CUDA kernels with nn.Linear + torch.compile (the original failing case)"""
        
#         class Model(nn.Module):
#             def __init__(self, size):
#                 super().__init__()
#                 self.linear = nn.Linear(size, size, device="cuda", dtype=torch.float32)
            
#             def forward(self, x):
#                 return extension_cpp.ops.mymuladd(self.linear(x), self.linear(x), 0.0)
        
#         # Test sizes that previously failed
#         for size in [1000, 5000, 10000]:
#             with self.subTest(size=size):
#                 torch.manual_seed(42)
#                 model = Model(size)
#                 x = torch.randn((1, size), device="cuda", dtype=torch.float32)
                
#                 with torch.no_grad():
#                     expected = model(x)
#                     compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
#                     actual = compiled_model(x)
                
#                 self.assertEqual(actual, expected)

#     @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
#     def test_compile_custom_only(self):
#         """Test custom operations alone with torch.compile"""
        
#         def model(x):
#             return extension_cpp.ops.mymuladd(x, x, 1.0)
        
#         for size in [1000, 5000, 10000]:
#             with self.subTest(size=size):
#                 torch.manual_seed(42)
#                 x = torch.randn((size,), device="cuda", dtype=torch.float32)
                
#                 with torch.no_grad():
#                     expected = model(x)
#                     compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
#                     actual = compiled_model(x)
                
#                 self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
