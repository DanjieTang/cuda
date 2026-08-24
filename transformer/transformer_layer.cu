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

int main(){
    int hidden_dim = 512;
    int seq_len = 51;

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


}