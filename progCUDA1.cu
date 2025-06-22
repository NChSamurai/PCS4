#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

double getTime(){
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

// Ядро CUDA для суммирования элементов массива (редукция)
__global__ void sumArray(int *array, int *result, int size) {
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (i < size) ? array[i] : 0;
    __syncthreads();

    // Редукция в shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    // Первый поток записывает результат для этого блока
    if (tid == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Использование: " << argv[0] << " <размер_массива> <размер_блока>" << std::endl;
        return 1;
    }

    const int SIZE = atoi(argv[1]);
    const int blockSize = atoi(argv[2]);
    if (SIZE <= 0 || blockSize <= 0) {
        std::cerr << "Размер массива и рамзер блока должны быть положительными числами" << std::endl;
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

    // Выделяем память на устройстве
    int *d_array, *d_partial_sums;
    int gridSize = (index + blockSize - 1) / blockSize;
    int *partial_sums = new int[gridSize];

    cudaMalloc(&d_array, index * sizeof(int));
    cudaMalloc(&d_partial_sums, gridSize * sizeof(int));

    // Копируем данные на устройство
    cudaMemcpy(d_array, array, index * sizeof(int), cudaMemcpyHostToDevice);

    // Запускаем ядро
    sumArray<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_array, d_partial_sums, index);

    // Копируем частичные суммы обратно
    cudaMemcpy(partial_sums, d_partial_sums, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Суммируем частичные суммы на хосте
    int summ = 0;
    for (int i = 0; i < gridSize; i++) {
        summ += partial_sums[i];
    }

    double endTime = getTime();
    std::cout << endTime - startTime << "\n";
    appendTimeToBinaryFile(double(endTime - startTime));

    // Освобождаем память
    delete[] array;
    delete[] partial_sums;
    cudaFree(d_array);
    cudaFree(d_partial_sums);

    return 0;
}
