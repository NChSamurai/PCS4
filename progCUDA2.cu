#include <iostream>
#include <array>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <iomanip>

double getTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    double seconds = std::chrono::duration<double>(duration).count();
    seconds = std::round(seconds * 1e5) / 1e5;
    return seconds;
}

void appendTimeToBinaryFile(double timeValue) {
    std::ofstream outfile("GeneralTime.bin", std::ios::binary | std::ios::app);
    if (!outfile) {
        std::cerr << "Ошибка открытия файла GeneralTime.bin" << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<const char*>(&timeValue), sizeof(double));
    if (!outfile.good()) {
        std::cerr << "Ошибка записи в файл" << std::endl;
    }
    outfile.close();
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void bitonicSortKernel(int* devArray, int j, int k, bool dir) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int ij = i ^ j;

    if (i < ij) {
        if ((i & k) == (dir ? 0 : k)) {
            if (devArray[i] > devArray[ij]) {
                int temp = devArray[i];
                devArray[i] = devArray[ij];
                devArray[ij] = temp;
            }
        }
    }
}

void parallelBitonicSort(int* array, int N) {
    int* devArray;
    size_t bytes = N * sizeof(int);

    CUDA_CHECK(cudaMalloc(&devArray, bytes));
    CUDA_CHECK(cudaMemcpy(devArray, array, bytes, cudaMemcpyHostToDevice));

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernel<<<gridSize, blockSize>>>(devArray, j, k, true);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaMemcpy(array, devArray, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(devArray));
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Использование: " << argv[0] << " <размер_массива>" << std::endl;
        return 1;
    }

    const int SIZE = atoi(argv[1]);
    if (SIZE <= 0) {
        std::cerr << "Размер массива должен быть положительным числом" << std::endl;
        return 1;
    }

    std::ifstream file("array.txt");
    if (!file.is_open()) {
        std::cerr << "Ошибка открытия файла :(" << std::endl;
        return 1;
    }

    int* array = new int[SIZE];
    int index = 0;

    while (index < SIZE && file >> array[index]) {
        index++;
    }

    file.close();

    double startTime = getTime();
    parallelBitonicSort(array, index);
    double endTime = getTime();

    std::ofstream out("arraySorted.txt");
    for (int i = 0; i < index; i++) {
        out << array[i] << '\n';
    }
    out.close();

    std::cout << endTime - startTime << "\n";
    appendTimeToBinaryFile(endTime - startTime);

    delete[] array;
    return 0;
}
