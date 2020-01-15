#include<random>
#include<iostream>

size_t n_blocks = 20;
size_t n_threads = 20;
size_t vec_size = n_blocks * n_threads;


void fill_random(double* a, size_t vec_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1000.0);
    for(size_t k = 0; k < vec_size; ++k) {
        *a++ = dis(gen);
    }
}

__global__ void add_elem(double* p_a, double* p_b, double* p_c, size_t vec_size)
{
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < vec_size) {
        p_c[index] = p_a[index] + p_b[index];
    }
}

int main(int argc, char** argv)
{

    // host vectors
    double *a, *b, *c;

    size_t vec_byte_size = vec_size * sizeof(double);

    a =(double*)malloc(vec_byte_size);
    b =(double*)malloc(vec_byte_size);
    c =(double*)malloc(vec_byte_size);

    fill_random(a, vec_size);
    fill_random(b, vec_size);

    // device vectors
    double *d_a, *d_b, *d_c;

    if (cudaMalloc((void**)(&d_a), vec_byte_size) != cudaSuccess) {
        std::cerr << "Cannot allocate memory on device, exiting" << std::endl;
        return 1;
    }

    if (cudaMalloc((void**)(&d_b), vec_byte_size) != cudaSuccess) {
        std::cerr << "Cannot allocate memory on device, exiting" << std::endl;
        return 1;
    }

    if (cudaMalloc((void**)(&d_c), vec_byte_size) != cudaSuccess) {
        std::cerr << "Cannot allocate memory on device, exiting" << std::endl;
        return 1;
    }

    cudaMemcpy((void*)d_a, (void*)a, vec_byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_b, (void*)b, vec_byte_size, cudaMemcpyHostToDevice);

    add_elem<<<n_blocks, n_threads>>>(d_a, d_b, d_c, vec_size);

    cudaMemcpy((void*)c, (void*)d_c, vec_byte_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for(size_t k = 0; k < vec_size; ++k) {
        if (a[k] + b[k] != c[k]) {
            std::cerr << "Error at index " << k << " " << a[k] << " + " << b[k] << " != " << c[k] << std::endl;
        }
    }

    cudaFree(&d_a);
    cudaFree(&d_b);
    cudaFree(&d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}
