#include <iostream>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

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
    outfile.close();
}

// Ядро CUDA для операций с матрицами
__global__ void matrixOperations(const double* matrix1, const double* matrix2,
                                double* resultAdd, double* resultSub,
                                double* resultMul, double* resultDiv,
                                size_t rows, size_t cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        int idx = i * cols + j;
        resultAdd[idx] = matrix1[idx] + matrix2[idx];
        resultSub[idx] = matrix1[idx] - matrix2[idx];
        resultMul[idx] = matrix1[idx] * matrix2[idx];
        resultDiv[idx] = (matrix2[idx] != 0) ? (matrix1[idx] / matrix2[idx])
                                            : __longlong_as_double(0x7ff8000000000000);
    }
}

void loadMatrixFromFile(const std::string& filename, double* matrix, size_t rows, size_t cols) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Не удалось открыть файл: " + filename);
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (!(file >> matrix[i * cols + j])) {
                throw std::runtime_error("Ошибка чтения данных из файла: " + filename);
            }
        }
    }
}

int main() {
    // Получаем размеры из переменных окружения
    const char* rows_env = std::getenv("ROWS");
    const char* cols_env = std::getenv("COLS");
    const char* block_size_env = std::getenv("BLOCKSIZE");

    if (!rows_env || !cols_env || !block_size_env) {
        std::cerr << "Ошибка: переменные окружения ROWS, COLS и BLOCKSIZE должны быть установлены" << std::endl;
        return 1;
    }

    const size_t ROWS = std::stoul(rows_env);
    const size_t COLS = std::stoul(cols_env);
    const int BLOCK_SIZE = std::stoi(block_size_env);

    if (ROWS == 0 || COLS == 0) {
        std::cerr << "Ошибка: размеры матрицы должны быть положительными числами" << std::endl;
        return 1;
    }

    if (BLOCK_SIZE <= 0 || BLOCK_SIZE > 32) {
        std::cerr << "Ошибка: размер блока должен быть в диапазоне 1-32" << std::endl;
        return 1;
    }

    // Выделяем память на хосте (единый блок памяти для каждой матрицы)
    double* matrix1 = new double[ROWS * COLS];
    double* matrix2 = new double[ROWS * COLS];
    double* resultAdd = new double[ROWS * COLS];
    double* resultSub = new double[ROWS * COLS];
    double* resultMul = new double[ROWS * COLS];
    double* resultDiv = new double[ROWS * COLS];

    try {
        loadMatrixFromFile("matrix.txt", matrix1, ROWS, COLS);
        loadMatrixFromFile("matrix2.txt", matrix2, ROWS, COLS);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        delete[] matrix1;
        delete[] matrix2;
        delete[] resultAdd;
        delete[] resultSub;
        delete[] resultMul;
        delete[] resultDiv;
        return 1;
    }
    double startTime = getTime();
    // Выделяем память на устройстве
    double *d_matrix1, *d_matrix2, *d_resultAdd, *d_resultSub, *d_resultMul, *d_resultDiv;

    cudaMalloc(&d_matrix1, ROWS * COLS * sizeof(double));
    cudaMalloc(&d_matrix2, ROWS * COLS * sizeof(double));
    cudaMalloc(&d_resultAdd, ROWS * COLS * sizeof(double));
    cudaMalloc(&d_resultSub, ROWS * COLS * sizeof(double));
    cudaMalloc(&d_resultMul, ROWS * COLS * sizeof(double));
    cudaMalloc(&d_resultDiv, ROWS * COLS * sizeof(double));

    // Копируем данные на устройство
    cudaMemcpy(d_matrix1, matrix1, ROWS * COLS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, ROWS * COLS * sizeof(double), cudaMemcpyHostToDevice);

    // Настраиваем размеры блоков и сетки
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((COLS + blockDim.x - 1) / blockDim.x,
                 (ROWS + blockDim.y - 1) / blockDim.y);



    // Запускаем ядро
    matrixOperations<<<gridDim, blockDim>>>(d_matrix1, d_matrix2,
                                          d_resultAdd, d_resultSub,
                                          d_resultMul, d_resultDiv,
                                          ROWS, COLS);

    // Копируем результаты обратно на хост
    cudaMemcpy(resultAdd, d_resultAdd, ROWS * COLS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultSub, d_resultSub, ROWS * COLS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultMul, d_resultMul, ROWS * COLS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultDiv, d_resultDiv, ROWS * COLS * sizeof(double), cudaMemcpyDeviceToHost);

    double endTime = getTime();

    std::cout << endTime - startTime << "\n";
    appendTimeToBinaryFile(double(endTime - startTime));

    // Освобождаем память
    delete[] matrix1;
    delete[] matrix2;
    delete[] resultAdd;
    delete[] resultSub;
    delete[] resultMul;
    delete[] resultDiv;

    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_resultAdd);
    cudaFree(d_resultSub);
    cudaFree(d_resultMul);
    cudaFree(d_resultDiv);

    return 0;
}
