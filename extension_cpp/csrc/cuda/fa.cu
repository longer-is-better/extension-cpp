#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace extension_cpp {



// one block on one SM
// 
__global__ void flashattention_kernel(
    uint Tr, uint Tc, uint Br, uint Bc, uint head_dim, uint element_size,
    const float* q, const float* k, const float* v, const float* l, const float* m, float* o
) {
    extern __shared__ float smem[];
    float* shared_q = smem;
    float* shared_k = shared_q + Br * head_dim * element_size;
    float* shared_v = shared_k + Bc * head_dim * element_size;
    float* shared_o = shared_v + Bc * head_dim * element_size;
    float* shared_l = shared_o + Br * head_dim * element_size;
    float* shared_m = shared_l + Br * element_size;

    uint load_kv_blocknum = (head_dim + Br - 1) / Br;
    uint load_qo_blocknum = (head_dim + Bc - 1) / Bc;

    for (int c = 0; c < Tc; c++) {
        // load k v
        for (int i = 0; i < load_kv_blocknum; i++) {
            if (i * Br + threadIdx.y >= head_dim) continue; 

            shared_k[
                i * Br * Bc + threadIdx.y * Bc + threadIdx.x
            ] = k[
                c * Bc * head_dim + threadIdx.x * head_dim + i * Br + threadIdx.y
            ];

            shared_v[
                threadIdx.x * head_dim + i * Br + threadIdx.y
            ] = v[
                c * Bc * head_dim + threadIdx.x * head_dim + i * Br + threadIdx.y
            ];
        }
        // load q o l m
        for (int r = 0; r < Tr; r++) {
            for (int i = 0; i < load_qo_blocknum; i++) {
                shared_q[
                    threadIdx.y * head_dim + Bc * i + threadIdx.x
                ] = q[
                    r * Br * head_dim + threadIdx.y * head_dim + Bc * i + threadIdx.x
                ];
                
                shared_o[
                    threadIdx.y * head_dim + Bc * i + threadIdx.x
                ] = o[
                    r * Br * head_dim + threadIdx.y * head_dim + Bc * i + threadIdx.x
                ];

                if (threadIdx.x == 0) {
                    shared_l[threadIdx.y] = l[r * Br + threadIdx.y];
                    shared_m[threadIdx.y] = m[r * Br + threadIdx.y];
                }
            }
        }
        // calculate one cell of QKt
        float s = 0;
        for (int i = 0; i < head_dim; i++) {
            __fmaf_rn(shared_q[threadIdx.y * head_dim + i], shared_k[i * Bc + threadIdx.x], s);
        }
        // _m
    }

}




at::Tensor flashattention_cuda(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v
) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    uint M = deviceProp.sharedMemPerBlock;
    uint element_size = q.element_size();
    uint seq_len = q.size(0);
    uint head_dim = q.size(1);
    uint Bc = M / (4 * head_dim * element_size);
    uint Br = min(Bc, head_dim);
    uint Tr = (seq_len + Br - 1) / Br;
    uint Tc = (seq_len + Bc - 1) / Bc;
    at::Tensor q_contig = q.contiguous();
    at::Tensor k_contig = k.contiguous();
    at::Tensor v_contig = v.contiguous();
    at::Tensor l = at::zeros({seq_len}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    at::Tensor m = at::full(
        {seq_len},
        -std::numeric_limits<float>::infinity(),
        torch::dtype(torch::kFloat32).device(torch::kCUDA)
    );
    at::Tensor result = at::zeros({seq_len, head_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    flashattention_kernel<<<{1}, {Bc, Br}, (2 * (Bc + Br) * head_dim + 2 * Br) * element_size>>>(
        Tr, Tc, Br, Bc, head_dim, element_size,
        q_contig.data_ptr<float>(),
        k_contig.data_ptr<float>(),
        v_contig.data_ptr<float>(),
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        result.data_ptr<float>()
    );
    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
    m.impl("flashattention", &flashattention_cuda);
  }
  
}