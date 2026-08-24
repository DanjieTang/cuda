// Layer Normalization in CUDA, mirroring PyTorch's:
//   torch.nn.LayerNorm(normalized_shape=N, eps=1e-5)
//   F.layer_norm(x, normalized_shape=(N,), weight=gamma, bias=beta, eps=eps)
//
// For each row x of shape (N,):
//   mean = sum(x) / N
//   var  = sum((x - mean)^2) / N          (biased, like PyTorch)
//   y_i  = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i
//
// Strategy: one thread block per row, block-wide reductions (shared memory)
// for the mean and variance.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                      \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// Block-wide sum reduction.
// Each thread passes its partial value; returns the total to every thread.
// Requires shared memory sized blockDim.x floats (passed dynamically).
// ---------------------------------------------------------------------------
__device__ float blockReduceSum(float val, float* shared) {
  int tid = threadIdx.x;
  shared[tid] = val;
  __syncthreads();

  // Tree reduction: stride halves each round.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  float total = shared[0];
  __syncthreads();  // avoid overwriting shared before everyone read it
  return total;
}

// ---------------------------------------------------------------------------
// LayerNorm kernel: grid = (rows) blocks, block = blockSize threads.
// Each block normalizes one row of `x` into `y`.
//   x, y   : (rows, N)
//   gamma  : (N,) scale    (PyTorch `weight`)
//   beta   : (N,) shift    (PyTorch `bias`)
// ---------------------------------------------------------------------------
__global__ void layerNormKernel(const float* x, float* y,
                                const float* gamma, const float* beta,
                                int N, float eps) {
  extern __shared__ float shared[];  // blockDim.x floats

  const int row = blockIdx.x;
  const float* x_row = x + (long)row * N;
  float* y_row = y + (long)row * N;

  // Pass 1: mean. Each thread accumulates a strided partial sum.
  float partial = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) partial += x_row[i];
  float mean = blockReduceSum(partial, shared) / N;

  // Pass 2: variance (biased, dividing by N like PyTorch).
  partial = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float d = x_row[i] - mean;
    partial += d * d;
  }
  float var = blockReduceSum(partial, shared) / N;

  // Pass 3: normalize + affine transform.
  float rstd = rsqrtf(var + eps);  // 1 / sqrt(var + eps)
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    y_row[i] = (x_row[i] - mean) * rstd * gamma[i] + beta[i];
  }
}

// ---------------------------------------------------------------------------
// CPU reference (what PyTorch would compute) for verification.
// ---------------------------------------------------------------------------
void layerNormCPU(const float* x, float* y,
                  const float* gamma, const float* beta,
                  int rows, int N, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + (long)r * N;
    float* yr = y + (long)r * N;

    float mean = 0.0f;
    for (int i = 0; i < N; ++i) mean += xr[i];
    mean /= N;

    float var = 0.0f;
    for (int i = 0; i < N; ++i) {
      float d = xr[i] - mean;
      var += d * d;
    }
    var /= N;

    float rstd = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < N; ++i)
      yr[i] = (xr[i] - mean) * rstd * gamma[i] + beta[i];
  }
}

int main() {
  const int rows = 1024;   // batch size
  const int N = 512;       // features per row (normalized_shape)
  const float eps = 1e-5f;
  const int blockSize = 256;

  size_t matBytes = (size_t)rows * N * sizeof(float);
  size_t vecBytes = N * sizeof(float);

  // Host allocations
  float* h_x     = (float*)malloc(matBytes);
  float* h_y     = (float*)malloc(matBytes);
  float* h_ref   = (float*)malloc(matBytes);
  float* h_gamma = (float*)malloc(vecBytes);
  float* h_beta  = (float*)malloc(vecBytes);

  // Fill input with deterministic pseudo-random data
  srand(42);
  for (int i = 0; i < rows * N; ++i) h_x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
  for (int i = 0; i < N; ++i) {
    h_gamma[i] = 1.0f + 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    h_beta[i]  = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
  }

  // Device allocations + copies
  float *d_x, *d_y, *d_gamma, *d_beta;
  CUDA_CHECK(cudaMalloc(&d_x, matBytes));
  CUDA_CHECK(cudaMalloc(&d_y, matBytes));
  CUDA_CHECK(cudaMalloc(&d_gamma, vecBytes));
  CUDA_CHECK(cudaMalloc(&d_beta, vecBytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x, matBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, vecBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, h_beta, vecBytes, cudaMemcpyHostToDevice));

  // Launch: one block per row
  size_t sharedBytes = blockSize * sizeof(float);
  layerNormKernel<<<rows, blockSize, sharedBytes>>>(d_x, d_y, d_gamma, d_beta, N, eps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_y, d_y, matBytes, cudaMemcpyDeviceToHost));

  // Verify against CPU reference
  layerNormCPU(h_x, h_ref, h_gamma, h_beta, rows, N, eps);

  float maxErr = 0.0f;
  for (int i = 0; i < rows * N; ++i) {
    float err = fabsf(h_y[i] - h_ref[i]);
    if (err > maxErr) maxErr = err;
  }
  printf("max abs error vs CPU reference: %g\n", maxErr);
  printf("%s\n", maxErr < 1e-4f ? "PASS" : "FAIL");

  // Show a few values from row 0 (like printing a tensor slice in PyTorch)
  printf("x[0, :4] = % .4f % .4f % .4f % .4f\n", h_x[0], h_x[1], h_x[2], h_x[3]);
  printf("y[0, :4] = % .4f % .4f % .4f % .4f\n", h_y[0], h_y[1], h_y[2], h_y[3]);

  // Cleanup
  cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
  free(h_x); free(h_y); free(h_ref); free(h_gamma); free(h_beta);
  return 0;
}
