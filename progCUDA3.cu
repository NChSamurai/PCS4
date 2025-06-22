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
