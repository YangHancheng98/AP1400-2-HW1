#include "hw1.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

namespace algebra {
Matrix zeros(size_t n, size_t m) {
    Matrix matrix(n, std::vector<double>(m, 0));
    return matrix;
}

Matrix ones(size_t n, size_t m) {
    Matrix matrix(n, std::vector<double>(m, 1));
    return matrix;
}

Matrix random(size_t n, size_t m, double min, double max) {
    if (min >= max) {
        throw std::logic_error("Min value cannot be greater than max value.");
    }
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(
        min, std::nextafter(max, __DBL_MAX__));

    Matrix randomMatrix = zeros(n, m);
    for (auto& row : randomMatrix) {
        for (auto& elem : row) {
            elem = dist(mt);
        }
    }

    return randomMatrix;
}

void show(const Matrix& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            std::cout << std::setw(10) << std::setprecision(3) << elem << " ";
        }
        std::cout << endl;
    }
}

Matrix multiply(const Matrix& matrix, double c) {
    Matrix resultMatrix = zeros(matrix.size(), matrix[0].size());
    for (int col = 0; col < matrix[0].size(); col++) {
        for (int row = 0; row < matrix.size(); row++) {
            resultMatrix[row][col] = c * matrix[row][col];
        }
    }
    return resultMatrix;
}

Matrix multiply(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.size() == 0 || matrix2.size() == 0) {
        return Matrix{};
    }
    if (matrix1[0].size() != matrix2.size()) {
        throw std::logic_error("Invalid matrix format for multiplication.");
    }

    int resultRow = matrix2[0].size();
    int resultCol = matrix1.size();
    Matrix resultMatrix = zeros(resultCol, resultRow);
    for (int j = 0; j < resultRow; j++) {
        for (int i = 0; i < resultCol; i++) {
            double sum = 0;
            for (int m = 0; m < matrix1[0].size(); m++) {
                sum += matrix1[i][m] * matrix2[m][j];
            }
            resultMatrix[i][j] = sum;
        }
    }
    return resultMatrix;
}

Matrix sum(const Matrix& matrix, double c) {
    if (matrix.size() == 0) {
        return Matrix{};
    }
    Matrix resultMatrix = zeros(matrix.size(), matrix[0].size());
    for (int col = 0; col < matrix[0].size(); col++) {
        for (int row = 0; row < matrix.size(); row++) {
            resultMatrix[row][col] = c + matrix[row][col];
        }
    }
    return resultMatrix;
}

Matrix sum(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.size() != matrix2.size()) {
        throw std::logic_error("Invalid matrix format for sum.");
    }
    if (matrix1.size() == 0 || matrix2.size() == 0) {
        return Matrix{};
    }
    if (matrix1[0].size() != matrix2[0].size()) {
        throw std::logic_error("Invalid matrix format for sum.");
    }

    Matrix resultMatrix = zeros(matrix1.size(), matrix1[0].size());
    for (int col = 0; col < matrix1[0].size(); col++) {
        for (int row = 0; row < matrix1.size(); row++) {
            resultMatrix[row][col] = matrix1[row][col] + matrix2[row][col];
        }
    }
    return resultMatrix;
}

// From here, the variable naming is correct
Matrix transpose(const Matrix& matrix) {
    if (matrix.size() == 0) {
        return Matrix{};
    }
    int originRow = matrix[0].size();
    int originCol = matrix.size();
    Matrix resultMatrix = zeros(originRow, originCol);
    for (int row = 0; row < originRow; row++) {
        for (int col = 0; col < originCol; col++) {
            resultMatrix[row][col] = matrix[col][row];
        }
    }
    return resultMatrix;
}

Matrix minor(const Matrix& matrix, size_t rowToRemove, size_t colToRemove) {
    if (matrix.size() == 0 || matrix[0].size() == 0) {
        throw std::logic_error("Invalid matrix format for minor.");
    }
    if (rowToRemove >= matrix[0].size() || colToRemove >= matrix.size()) {
        throw std::logic_error("Invalid row/column index for minor.");
    }
    size_t n = matrix.size();
    size_t m = matrix[0].size();
    Matrix result;

    for (size_t i = 0; i < n; ++i) {
        if (i == rowToRemove) continue;
        std::vector<double> row;
        for (size_t j = 0; j < m; ++j) {
            if (j == colToRemove) continue;
            row.push_back(matrix[i][j]);
        }
        result.push_back(row);
    }
    return result;
}

double determinant(const Matrix& matrix) {
    if (matrix.size() == 0 || matrix[0].size() == 0) {
        return 1;
    }
    if (matrix.size() != matrix[0].size()) {
        throw std::logic_error("Determinant should apply on square matrix.");
    }
    if (matrix.size() == 2) {
        double a = matrix[0][0];
        double b = matrix[0][1];
        double c = matrix[1][0];
        double d = matrix[1][1];
        return a * d - b * c;
    }
    double result = 0;
    for (int col = 0; col < matrix[0].size(); col++) {
        double sign = (col % 2 == 0) ? 1 : -1;
        result +=
            pow(-1, col) * matrix[0][col] * determinant(minor(matrix, 0, col));
    }
    return result;
}

namespace {
Matrix getCofactorMatrix(const Matrix& matrix) {
    int row = matrix.size();
    int col = matrix[0].size();
    Matrix resultMatrix = zeros(row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            resultMatrix[i][j] =
                pow(-1, i + j) * determinant(minor(matrix, i, j));
        }
    }
    return resultMatrix;
}

Matrix getAdjMatrix(const Matrix& matrix) {
    return transpose(getCofactorMatrix(matrix));
}
}  // namespace

Matrix inverse(const Matrix& matrix) {
    if (matrix.size() == 0) {
        return Matrix{};
    }
    if (matrix.size() != matrix[0].size()) {
        throw std::logic_error("inverse should apply on square matrix.");
    }
    double det = determinant(matrix);
    if (det == 0) {
        throw std::logic_error("singular matrices have no inverse.");
    }

    return multiply(getAdjMatrix(matrix), 1.0 / det);
}

Matrix concatenate(const Matrix& matrix1, const Matrix& matrix2, int axis = 0) {
    int row1 = matrix1.size();
    int row2 = matrix2.size();
    int col1 = matrix1[0].size();
    int col2 = matrix2[0].size();

    if (axis == 0 && col1 != col2) {
        throw std::logic_error(
            "concatenate vertically, the column should be equal.");
    }
    if (axis == 1 && row1 != row2) {
        throw std::logic_error(
            "concatenate horizontally, the column should be equal.");
    }
    Matrix result;
    if (axis == 0) {
        result = matrix1;
        result.insert(result.end(), matrix2.begin(), matrix2.end());
        return result;
    } else if (axis == 1) {
        result = matrix1;
        for (int i = 0; i < result.size(); i++) {
            result[i].insert(result[i].end(), matrix2[i].begin(),
                             matrix2[i].end());
        }
        return result;
    } else {
        throw std::logic_error(
            "Invalid axis. Use 0 (vertical) or 1 (horizontal).");
    }
}

// Matrix ero_swap(const Matrix& matrix, size_t r1, size_t r2);
// Matrix ero_multiply(const Matrix& matrix, size_t r, double c);
// Matrix ero_sum(const Matrix& matrix, size_t r1, double c, size_t r2);
// Matrix upper_triangular(const Matrix& matrix);
}  // namespace algebra