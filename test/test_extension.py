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
            result = i + j  # 广播相加，得到 [rows, cols]
            return result.to(device)

        def make_tensor_r0(*size):
            t = torch.randn(size, device=device, requires_grad=requires_grad)
            # t[:, 3:] = 0
            return t

        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)
            # return torch.ones(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor_rcsum(32, 32), make_tensor_rcsum(32, 32), make_tensor_rcsum(32, 32)],
            # [make_tensor(33, 32), make_tensor(33, 32), make_tensor(33, 32)],
            [make_tensor(32, 64), make_tensor(32, 64), make_tensor_r0(32, 64)],
            [make_tensor(64, 64), make_tensor(64, 64), make_tensor_r0(64, 64)],
            [make_tensor(512, 32), make_tensor(512, 32), make_tensor(512, 32)],
            [make_tensor(512, 128), make_tensor(512, 128), make_tensor(512, 128)],
            [make_tensor(1024, 64), make_tensor(1024, 64), make_tensor(1024, 64),],
        ]

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            expected = reference_attention(*args)
            result = extension_cpp.ops.flashattention(*args)
            # print(f"expected {expected[:10]} \n result {result[:10]}")
            torch.testing.assert_close(result, expected)

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
