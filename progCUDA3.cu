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
[chernousov.nikita2005.gmail.com@node2 Lab2]$ ^C
[chernousov.nikita2005.gmail.com@node2 Lab2]$ cat progCUDA2.cu
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
[chernousov.nikita2005.gmail.com@node2 Lab2]$ ^C
[chernousov.nikita2005.gmail.com@node2 Lab2]$ cat progCUDA3.cu
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <algorithm>
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

// Ядро CUDA для выполнения операций с массивами
__global__ void arrayOperations(const double* array1, const double* array2,
                               double* resultAdd, double* resultSub,
                               double* resultMul, double* resultDiv,
                               size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        resultAdd[i] = array1[i] + array2[i];
        resultSub[i] = array1[i] - array2[i];
        resultMul[i] = array1[i] * array2[i];
        // Используем собственное представление NaN вместо std::numeric_limits
        resultDiv[i] = (array2[i] != 0) ? (array1[i] / array2[i]) : __longlong_as_double(0x7ff8000000000000);
    }
}

// Функция для загрузки массива из файла
void loadArrayFromFile(const std::string& filename, double* arr, size_t size) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Не удалось открыть файл: " + filename);
    }

    for (size_t i = 0; i < size; ++i) {
        if (!(file >> arr[i])) {
            throw std::runtime_error("Ошибка чтения данных из файла: " + filename);
        }
    }
}

int main(int argc, char* argv[]) {
    // Получаем размер из переменной окружения
    const char* size_env = std::getenv("SIZE");
    if (!size_env) {
        std::cerr << "Ошибка: переменная окружения SIZE не установлена" << std::endl;
        return 1;
    }
    // Получаем размер блока из переменной окружения
    const char* block_size_env = std::getenv("BLOCKSIZE");
    if (!block_size_env) {
        std::cerr << "Ошибка: переменная окружения BLOCKSIZE не установлена" << std::endl;
        return 1;
    }

    const int blockSize = std::stoi(block_size_env);
    const size_t SIZE = std::stoul(size_env);
    if (SIZE <= 0) {
        std::cerr << "Размер массива должен быть положительным числом" << std::endl;
        return 1;
    }

    // Определяем размер блока (можно передавать как параметр)
    int gridSize = (SIZE + blockSize - 1) / blockSize;

    // Создаем массивы на хосте
    double* array1 = new double[SIZE];
    double* array2 = new double[SIZE];
    double* resultAdd = new double[SIZE];
    double* resultSub = new double[SIZE];
    double* resultMul = new double[SIZE];
    double* resultDiv = new double[SIZE];

    try {
        loadArrayFromFile("array.txt", array1, SIZE);
        loadArrayFromFile("array2.txt", array2, SIZE);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        delete[] array1;
        delete[] array2;
        delete[] resultAdd;
        delete[] resultSub;
        delete[] resultMul;
        delete[] resultDiv;
        return 1;
    }
double startTime = getTime();

    // Выделяем память на устройстве
    double *d_array1, *d_array2, *d_resultAdd, *d_resultSub, *d_resultMul, *d_resultDiv;

    cudaMalloc(&d_array1, SIZE * sizeof(double));
    cudaMalloc(&d_array2, SIZE * sizeof(double));
    cudaMalloc(&d_resultAdd, SIZE * sizeof(double));
    cudaMalloc(&d_resultSub, SIZE * sizeof(double));
    cudaMalloc(&d_resultMul, SIZE * sizeof(double));
    cudaMalloc(&d_resultDiv, SIZE * sizeof(double));

    // Копируем данные на устройство
    cudaMemcpy(d_array1, array1, SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, array2, SIZE * sizeof(double), cudaMemcpyHostToDevice);


    // Запускаем ядро
    arrayOperations<<<gridSize, blockSize>>>(d_array1, d_array2,
                                           d_resultAdd, d_resultSub,
                                           d_resultMul, d_resultDiv,
                                           SIZE);

    // Копируем результаты обратно на хост
    cudaMemcpy(resultAdd, d_resultAdd, SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultSub, d_resultSub, SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultMul, d_resultMul, SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultDiv, d_resultDiv, SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    double endTime = getTime();

    std::cout << endTime - startTime << "\n";
    appendTimeToBinaryFile(double(endTime - startTime));

    // Освобождаем память
    delete[] array1;
    delete[] array2;
    delete[] resultAdd;
    delete[] resultSub;
    delete[] resultMul;
    delete[] resultDiv;

    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(d_resultAdd);
    cudaFree(d_resultSub);
    cudaFree(d_resultMul);
    cudaFree(d_resultDiv);

    return 0;
}
