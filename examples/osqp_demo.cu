//
// Created by plusai on 4/21/22.
//
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}


// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUSPARSE_CHECK(err)                                                                        \
    do {                                                                                           \
        cusparseStatus_t err_ = (err);                                                             \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                                     \
            printf("cusparse error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
        }                                                                                          \
    } while (0)

//int main(int argc, char *argv[]) {
//    cusolverDnHandle_t cusolverH = NULL;
//    cudaStream_t stream = NULL;
//
//    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
//    const int batchSize = 1;
//    const int nrhs = 1;
//    const int m = 5;
//    const int lda = m;
//    const int ldb = m;
//    /*
//     *      | 1     2     3 |
//     * A0 = | 2     5     5 | = L0 * L0**T
//     *      | 3     5    12 |
//     *
//     *            | 1.0000         0         0 |
//     * where L0 = | 2.0000    1.0000         0 |
//     *            | 3.0000   -1.0000    1.4142 |
//     *
//     *      | 1     2     3 |
//     * A1 = | 2     4     5 | is not s.p.d., failed at row 2
//     *      | 3     5    12 |
//     *
//     */
//
//    const std::vector<double> A0 = {1.000000, 0.353553, 0.706628, 0.998647, 0.000000,
//                                    0.353553, 1.000000, 0.999323, 0.000000, 0.999323,
//                                    0.706628, 0.999323, 10.000000, 0.000000, 0.000000,
//                                    0.998647, 0.000000, 0.000000, 10.000000, 0.000000,
//                                    0.000000, 0.999323, 0.000000, 0.000000, 10.000000, };
////    const std::vector<double> A1 = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 12.0};
//    const std::vector<double> B0 = {-0.5000000, 0.707000000, -0.000000, 0.000000, 0.000000};
//    std::vector<double> X0(m, 0);             /* X0 = A0\B0 */
//    std::vector<int> infoArray(batchSize, 0); /* host copy of error info */
//
//    std::vector<double> L0(lda * m); /* cholesky factor of A0 */
//
//    std::vector<double *> Aarray(batchSize, nullptr);
//    std::vector<double *> Barray(batchSize, nullptr);
//
//    double **d_Aarray = nullptr;
//    double **d_Barray = nullptr;
//    int *d_infoArray = nullptr;
//
//    std::printf("A0 = (matlab base-1)\n");
//    print_matrix(m, m, A0.data(), lda);
//    std::printf("=====\n");
//
////    std::printf("A1 = (matlab base-1)\n");
////    print_matrix(m, m, A1.data(), lda);
////    std::printf("=====\n");
//
//    std::printf("B0 = (matlab base-1)\n");
//    print_matrix(m, 1, B0.data(), ldb);
//    std::printf("=====\n");
//
//    /* step 1: create cusolver handle, bind a stream */
//    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
//
//    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
//
//    /* step 2: copy A to device */
//    for (int j = 0; j < batchSize; j++) {
//        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&Aarray[j]), sizeof(double) * lda * m));
//        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&Barray[j]), sizeof(double) * ldb * nrhs));
//    }
//    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_infoArray), sizeof(int) * infoArray.size()));
//
//    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Aarray), sizeof(double *) * Aarray.size()));
//    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Barray), sizeof(double *) * Barray.size()));
//
//    CUDA_CHECK(cudaMemcpyAsync(Aarray[0], A0.data(), sizeof(double) * A0.size(),
//                               cudaMemcpyHostToDevice, stream));
////    CUDA_CHECK(cudaMemcpyAsync(Aarray[1], A1.data(), sizeof(double) * A1.size(),
////                               cudaMemcpyHostToDevice, stream));
//
//    CUDA_CHECK(
//            cudaMemcpyAsync(Barray[0], B0.data(), sizeof(double) * B0.size(), cudaMemcpyHostToDevice, stream));
////    CUDA_CHECK(
////            cudaMemcpyAsync(Barray[1], B0.data(), sizeof(double) * B0.size(), cudaMemcpyHostToDevice, stream));
//
//    CUDA_CHECK(cudaMemcpyAsync(d_Aarray, Aarray.data(), sizeof(double) * Aarray.size(),
//                               cudaMemcpyHostToDevice, stream));
//    CUDA_CHECK(cudaMemcpyAsync(d_Barray, Barray.data(), sizeof(double) * Barray.size(),
//                               cudaMemcpyHostToDevice, stream));
//
//    /* step 3: Cholesky factorization */
//    CUSOLVER_CHECK(
//            cusolverDnDpotrfBatched(cusolverH, uplo, m, d_Aarray, lda, d_infoArray, batchSize));
//
//    CUDA_CHECK(cudaMemcpyAsync(infoArray.data(), d_infoArray, sizeof(int) * infoArray.size(),
//                               cudaMemcpyDeviceToHost, stream));
//    CUDA_CHECK(cudaMemcpyAsync(L0.data(), Aarray[0], sizeof(double) * lda * m,
//                               cudaMemcpyDeviceToHost, stream));
//
//    CUDA_CHECK(cudaStreamSynchronize(stream));
//
//    for (int j = 0; j < batchSize; j++) {
//        std::printf("info[%d] = %d\n", j, infoArray[j]);
//    }
//
//    assert(0 == infoArray[0]);
//    /* A1 is singular */
////    assert(2 == infoArray[1]);
//
//    std::printf("L = (matlab base-1), upper triangle is don't care \n");
//    print_matrix(m, m, L0.data(), lda);
//    std::printf("=====\n");
//
//    /*
//     * step 4: solve A0*X0 = B0
//     *        | 1 |        | 10.5 |
//     *   B0 = | 1 |,  X0 = | -2.5 |
//     *        | 1 |        | -1.5 |
//     */
//    CUSOLVER_CHECK(cusolverDnDpotrsBatched(cusolverH, uplo, m, nrhs, /* only support rhs = 1*/
//                                           d_Aarray, lda, d_Barray, ldb, d_infoArray, batchSize));
//
//    CUDA_CHECK(cudaMemcpyAsync(infoArray.data(), d_infoArray, sizeof(int), cudaMemcpyDeviceToHost,
//                               stream));
//    CUDA_CHECK(
//            cudaMemcpyAsync(X0.data(), Barray[0], sizeof(double) * X0.size(), cudaMemcpyDeviceToHost, stream));
//
//    CUDA_CHECK(cudaStreamSynchronize(stream));
//
//    std::printf("after potrsBatched: infoArray[0] = %d\n", infoArray[0]);
//    if (0 > infoArray[0]) {
//        std::printf("%d-th parameter is wrong \n", -infoArray[0]);
//        exit(1);
//    }
//
//    std::printf("X0 = (matlab base-1)\n");
//    print_matrix(m, 1, X0.data(), ldb);
//    std::printf("=====\n");
//
//    /* free resources */
//    CUDA_CHECK(cudaFree(d_Aarray));
//    CUDA_CHECK(cudaFree(d_Barray));
//    CUDA_CHECK(cudaFree(d_infoArray));
//    for (int j = 0; j < batchSize; j++) {
//        CUDA_CHECK(cudaFree(Aarray[j]));
//        CUDA_CHECK(cudaFree(Barray[j]));
//    }
//
//    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
//
//    CUDA_CHECK(cudaStreamDestroy(stream));
//
//    CUDA_CHECK(cudaDeviceReset());
//
//    return EXIT_SUCCESS;
//}