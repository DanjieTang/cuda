#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <stdio.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

float random_normal() {
    static std::mt19937 gen(std::random_device{}());
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

void random_init_tensor(std::vector<float>& tensor){
    for (size_t i = 0; i < tensor.size(); i++){
        tensor[i] = random_normal();
    }
}

__global__ void matrix_multiplication(float* A, float* B, float* C, int M, int K, int N){
    __shared__ float tileA[tileSize][tileSize];
    __shared__ float tileB[tileSize][tileSize];

    int C_col = blockIdx.x * blockDim.x + threadIdx.x;
    int C_row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    for(int i = 0; i < K; i += tileSize){
        // Step 1, load in both tile A & B
        int tileCol = i + threadIdx.x;
        int tileRow = i + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = A[C_row * K + tileCol]; // Load in tile A
        tileB[threadIdx.y][threadIdx.x] = B[tileRow * N + C_col]; // Load in tile B

        __syncthreads();

        for(int j = 0; j < tileSize; j++){
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    C[C_row * N + C_col] = sum;
}

__device__ float reduce_sum(float local_sum, float* shared){
    int local_index = threadIdx.x;
    shared[local_index] = local_sum;
    __syncthreads();

    int stride = blockDim.x / 2;

    while(stride > 0){
        if (local_index < stride){
            shared[local_index] += shared[local_index + stride];
        }
        stride /= 2;
        __syncthreads();
    }

    float total = shared[0];
    __syncthreads();  // make sure every thread has read shared[0] before shared is reused
    return total;
}

__global__ void layer_norm(float* input_tensor, float* output_tensor, int dim){
    extern __shared__ float shared[];
    int local_index = threadIdx.x;
    int global_index = blockIdx.x * dim;
    float epsilon = 1e-5f;
    
    // Step 1, find the mean
    float local_sum = 0;
    for (int i = global_index + local_index; i < global_index + dim; i += blockDim.x){
        local_sum += input_tensor[i];
    }
    float dim_sum = reduce_sum(local_sum, shared, dim);
    float dim_avg = dim_sum / dim;

    // Step 2, find the variance
    float local_sum_square = 0;
    for (int i = global_index + local_index; i < global_index + dim; i += blockDim.x){
        float diff_square = input_tensor[i] - dim_avg;
        local_sum_square += diff_square * diff_square;
    }
    float dim_sum_square = reduce_sum(local_sum_square, shared, dim);
    float variance = dim_sum_square / dim;

    // Step 3, do the layernorm
    for (int i = global_index + local_index; i < global_index + dim; i += blockDim.x){
        float denominator = rsqrtf(variance + epsilon);
        output_tensor[i] = (input_tensor[i] - dim_avg) * denominator;
    }
}

__device__ float maxReduce(float local_max, float* shared){
    int local_index = threadIdx.x;
    int stride = blockDim.x / 2;
    
    // Load shared memory
    shared[local_index] = local_max;
    __syncthreads();

    while(stride > 0){
        if (local_index < stride){
            shared[local_index] = fmaxf(shared[local_index], shared[local_index + stride]);
        }
        stride /= 2;
        __syncthreads();
    }

    return shared[0];
}

__device__ float sumReduce(float local_sum, float* shared){
    int local_index = threadIdx.x;
    int stride = blockDim.x / 2;
    
    // Load shared memory
    shared[local_index] = local_sum;
    __syncthreads();

    while(stride > 0){
        if (local_index < stride){
            shared[local_index] += shared[local_index + stride];
        }
        stride /= 2;
        __syncthreads();
    }

    return shared[0];
}

__global__ void softmaxKernel(float* input_tensor, float* output_tensor, int dim){
    int batch_index = blockIdx.x * dim;
    extern __shared__ float shared[];

    // Step 1, find the max of each datapoint(each slice in the batch)
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < dim; i += blockDim.x){
        local_max = fmaxf(local_max, input_tensor[batch_index + i]);
    }
    float batch_max = maxReduce(local_max, shared);

    // Step 2
    float local_sum = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x){
        output_tensor[batch_index + i] = expf(input_tensor[batch_index + i] - batch_max);
        local_sum += output_tensor[batch_index + i];
    }
    float batch_sum = sumReduce(local_sum, shared);

    // Step 3
    for (int i = threadIdx.x; i < dim; i += blockDim.x){
        output_tensor[batch_index + i] /= batch_sum;
    }
}

__global__ void transpose(){
    
}

__global__ void transformer_layer(){

}

int main(){
    int hidden_dim = 512;
    int seq_len = 64;

    int num_element_input_tensor = seq_len * hidden_dim;
    int num_element_QKVO_tensor = hidden_dim * hidden_dim;
    int num_element_ffn_tensor = hidden_dim * hidden_dim * 4;
    size_t size_input_tensor = num_element_input_tensor * sizeof(float);
    size_t size_QKVO_tensor = num_element_QKVO_tensor * sizeof(float);
    size_t size_ffn_tensor = num_element_ffn_tensor * sizeof(float);

    std::vector<float> h_input_tensor(num_element_input_tensor);
    std::vector<float> h_Q_matrix(num_element_QKVO_tensor);
    std::vector<float> h_K_matrix(num_element_QKVO_tensor);
    std::vector<float> h_V_matrix(num_element_QKVO_tensor);
    std::vector<float> h_O_matrix(num_element_QKVO_tensor);
    std::vector<float> h_ffn_matrix_1(num_element_ffn_tensor);
    std::vector<float> h_ffn_matrix_2(num_element_ffn_tensor);

    random_init_tensor(h_input_tensor);
    random_init_tensor(h_Q_matrix);
    random_init_tensor(h_K_matrix);
    random_init_tensor(h_V_matrix);
    random_init_tensor(h_O_matrix);
    random_init_tensor(h_ffn_matrix_1);
    random_init_tensor(h_ffn_matrix_2);
    
    float *d_input_tensor, *d_Q_matrix, *d_K_matrix, *d_V_matrix, *d_O_matrix, *d_ffn_matrix_1, *d_ffn_matrix_2;

    CHECK_CUDA(cudaMalloc(&d_input_tensor, size_input_tensor));
    CHECK_CUDA(cudaMalloc(&d_Q_matrix, size_QKVO_tensor));
    CHECK_CUDA(cudaMalloc(&d_K_matrix, size_QKVO_tensor));
    CHECK_CUDA(cudaMalloc(&d_V_matrix, size_QKVO_tensor));
    CHECK_CUDA(cudaMalloc(&d_O_matrix, size_QKVO_tensor));
    CHECK_CUDA(cudaMalloc(&d_ffn_matrix_1, size_ffn_tensor));
    CHECK_CUDA(cudaMalloc(&d_ffn_matrix_2, size_ffn_tensor));

    CHECK_CUDA(cudaMemcpy(d_input_tensor, h_input_tensor.data(), size_input_tensor, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Q_matrix, h_Q_matrix.data(), size_QKVO_tensor, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K_matrix, h_K_matrix.data(), size_QKVO_tensor, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V_matrix, h_V_matrix.data(), size_QKVO_tensor, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_O_matrix, h_O_matrix.data(), size_QKVO_tensor, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ffn_matrix_1, h_ffn_matrix_1.data(), size_ffn_tensor, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ffn_matrix_2, h_ffn_matrix_2.data(), size_ffn_tensor, cudaMemcpyHostToDevice));


}