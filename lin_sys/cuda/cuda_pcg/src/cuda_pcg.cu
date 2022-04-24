/**
 *  Copyright (c) 2019-2021 ETH Zurich, Automatic Control Lab,
 *  Michel Schubiger, Goran Banjac.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "cuda_pcg.h"
#include "csr_type.h"
#include "cuda_configure.h"
#include "cuda_handler.h"
#include "cuda_malloc.h"
#include "cuda_lin_alg.h"
#include "cuda_wrapper.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */

#include <cusolverDn.h>
#include <stdio.h>
#include <vector>
#ifdef __cplusplus
extern "C" {extern CUDA_Handle_t *CUDA_handle;}
#endif

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


/*******************************************************************************
 *                              GPU Kernels                                    *
 *******************************************************************************/

__global__ void scalar_division_kernel(c_float       *res,
                                       const c_float *num,
                                       const c_float *den) {

  *res = (*num) / (*den);
}


/*******************************************************************************
 *                            Private Functions                                *
 *******************************************************************************/

/*
 * d_y = (P + sigma*I + A'*R*A) * d_x
 */
static void mat_vec_prod(cudapcg_solver *s,
                         c_float        *d_y,
                         const c_float  *d_x,
                         c_int           device) {

  c_float *sigma;
  c_float H_ZERO = 0.0;
  c_float H_ONE  = 1.0;
  c_int n = s->n;
  c_int m = s->m;
  csr *P  = s->P;
  csr *A  = s->A;
  csr *At = s->At;

  sigma = device ? s->d_sigma : s->h_sigma;

  /* d_y = d_x */
  checkCudaErrors(cudaMemcpy(d_y, d_x, n * sizeof(c_float), cudaMemcpyDeviceToDevice));

  /* d_y *= sigma */
  checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, sigma, d_y, 1));

  /* d_y += P * d_x */
  checkCudaErrors(cusparseCsrmvEx(CUDA_handle->cusparseHandle, P->alg,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  P->m, P->n, P->nnz, &H_ONE,
                                  CUDA_FLOAT, P->MatDescription, P->val,
                                  CUDA_FLOAT, P->row_ptr, P->col_ind, d_x,
                                  CUDA_FLOAT, &H_ONE, CUDA_FLOAT, d_y,
                                  CUDA_FLOAT, CUDA_FLOAT, P->buffer));

  if (m == 0) return;

  if (!s->d_rho_vec) {
    /* d_z = rho * A * d_x */
    checkCudaErrors(cusparseCsrmvEx(CUDA_handle->cusparseHandle, A->alg,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    A->m, A->n, A->nnz, s->h_rho,
                                    CUDA_FLOAT, A->MatDescription, A->val,
                                    CUDA_FLOAT, A->row_ptr, A->col_ind, d_x,
                                    CUDA_FLOAT, &H_ZERO, CUDA_FLOAT, s->d_z,
                                    CUDA_FLOAT, CUDA_FLOAT, A->buffer));
  }
  else {
    /* d_z = A * d_x */
    checkCudaErrors(cusparseCsrmvEx(CUDA_handle->cusparseHandle, A->alg,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    A->m, A->n, A->nnz, &H_ONE,
                                    CUDA_FLOAT, A->MatDescription, A->val,
                                    CUDA_FLOAT, A->row_ptr, A->col_ind, d_x,
                                    CUDA_FLOAT, &H_ZERO, CUDA_FLOAT, s->d_z,
                                    CUDA_FLOAT, CUDA_FLOAT, A->buffer));

    /* d_z = diag(d_rho_vec) * dz */
    cuda_vec_ew_prod(s->d_z, s->d_z, s->d_rho_vec, m);
  }

  /* d_y += A' * d_z */
  checkCudaErrors(cusparseCsrmvEx(CUDA_handle->cusparseHandle, At->alg,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  At->m, At->n, At->nnz, &H_ONE,
                                  CUDA_FLOAT, At->MatDescription, At->val,
                                  CUDA_FLOAT, At->row_ptr, At->col_ind, s->d_z,
                                  CUDA_FLOAT, &H_ONE, CUDA_FLOAT, d_y,
                                  CUDA_FLOAT, CUDA_FLOAT, A->buffer));
}

/**
 * For do not destroy the down-stream calculation, we copy the original csr matrix and convert it into coo format.
 * Then, we map these coo matricies to a huge dense matrix.
 * @param s
 */
c_float* csrMats2Dense(csr *s){
    csr* P = s;
    int m = s->m;
    int n = s->n;
    int nnz = s->nnz;
    c_float *ptr;
    c_float *vals = (c_float*)malloc(sizeof(c_float) * P->nnz);
    checkCudaErrors(cudaMalloc((void **)&ptr, sizeof(P->val) * P->nnz));
    checkCudaErrors(cudaMemcpy(ptr, P->val, P->nnz * sizeof(c_float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(vals, ptr, P->nnz * sizeof(c_float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(ptr));
//    for(int i = 0; i<P->nnz; i++){
//        printf("CSR val: %f ", vals[i]);
//    }
//    printf("\n");
    c_int *rowPtr, *colIdx;
    c_int row[nnz], col[nnz];
    checkCudaErrors(cudaMalloc((void**)&rowPtr, sizeof(c_int)*(P->nnz)));
    checkCudaErrors(cudaMalloc((void**)&colIdx, sizeof (c_int)*P->nnz));

    checkCudaErrors(cudaMemcpy(rowPtr, P->row_ind, (P->nnz) * sizeof(c_int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(&row, rowPtr, (P->nnz) * sizeof(c_int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(rowPtr));
//    for(int i = 0; i<P->nnz; i++){
//        printf("rowPtr: %d ", row[i]);
//    }
//    printf("\n");
    checkCudaErrors(cudaMemcpy(colIdx, P->col_ind, (P->nnz) * sizeof(c_int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(&col, colIdx, (P->nnz) * sizeof(c_int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(colIdx));
    //    for(int i = 0; i<P->nnz; i++){
//        printf("colIdx: %d ", col[i]);
//    }
//    printf("\n");
    c_float* mat = (c_float*)malloc(sizeof(c_float) * m * n);
    memset(mat, 0.0f, m * n * sizeof(c_float));
    for(int i = 0; i<nnz; i++){
        mat[row[i]*n+col[i]] = vals[i];
    }

//    for(int i = 0; i< m; i++){
//        for(int j=0; j<n;j++){
//            printf("P[%d][%d]: %f ", i, j, mat[i*n + j]);
//        }
//        printf("\n");
//    }
//    printf("\n");
    return mat;
}
/**
 * This function performs assignments from source matrix to target matrix
 * @param dst target matrix
 * @param src source matrix
 * @param dst_m rows of the target matrix
 * @param dst_n cols of the target matrix
 * @param m rows of the source matrix
 * @param n col of the source matrix
 * @param offset_x
 * @param offset_y
 */
c_int mergeMatrices(c_float* dst, c_float* src,int dst_m, int dst_n, int m, int n, int offset_x, int offset_y){
    int row = offset_x;
    int col = offset_y;
//    assert(dst_m >= m +offset_x && dst_n >= n + offset_y);
    for(int i = 0; i< m; i++){
        for(int j =0; j<n;j++){

            dst[row * dst_n + col + j] = src[i * n + j];
        }
        row++;
    }
    return 0;
}
void mergeScalar2Matrix(c_float* dst, c_float scalar, int dst_m, int dst_n, int n, int offset_x, int offset_y){
    int row = offset_x;
    int col = offset_y;

    for(int i = 0; i<n; i++){
        dst[row*dst_n + col + i] = scalar;
        row++;
    }
}

/*******************************************************************************
 *                              API Functions                                  *
 *******************************************************************************/
int linearSolverCHOL(cusolverDnHandle_t handle, c_int n, c_float *Acopy, int lda,
        c_float *b, c_float *x){

    int bufferSize = 0;
    int *info = NULL;
    c_float *buffer = NULL;
    c_float *A = NULL;
    int h_info = 0;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, uplo, n, (float*)Acopy, lda, &bufferSize));

    CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffer, sizeof(float)*bufferSize));
    CUDA_CHECK(cudaMalloc(&A, sizeof(float)*lda*n));


    // prepare a copy of A because potrf will overwrite A with L
    CUDA_CHECK(cudaMemcpy(A, Acopy, sizeof(c_float)*lda*n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(info, 0, sizeof(int)));


    CUSOLVER_CHECK(cusolverDnSpotrf(handle, uplo, n, (float*)A, lda, (float*)buffer, bufferSize, info));

    CUDA_CHECK(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: Cholesky factorization failed\n");
    }

    CUDA_CHECK(cudaMemcpy(x, b, sizeof(c_float)*n, cudaMemcpyDeviceToDevice));

    CUSOLVER_CHECK(cusolverDnSpotrs(handle, uplo, n, 1, (float*)A, lda, (float*)x, n, info));

    CUDA_CHECK(cudaDeviceSynchronize());


    if (info  ) { CUDA_CHECK(cudaFree(info)); }
    if (buffer) { CUDA_CHECK(cudaFree(buffer)); }
    if (A     ) { CUDA_CHECK(cudaFree(A)); }

    return 0;
}

int linearSolverLU(cusolverDnHandle_t handle,int n,const c_float *Acopy,int lda,
                   const c_float *b,c_float *x)
{
    int bufferSize = 0;
    int *info = NULL;
    c_float *buffer = NULL;
    c_float *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;

    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, n, n, (float*)Acopy, lda, &bufferSize));

    CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffer, sizeof(float)*bufferSize));
    CUDA_CHECK(cudaMalloc(&A, sizeof(float)*lda*n));
    CUDA_CHECK(cudaMalloc(&ipiv, sizeof(int)*n));


    // prepare a copy of A because getrf will overwrite A with L
    CUDA_CHECK(cudaMemcpy(A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(info, 0, sizeof(int)));


    CUSOLVER_CHECK(cusolverDnSgetrf(handle, n, n, (float*)A, lda, (float*)buffer, ipiv, info));
    CUDA_CHECK(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    CUDA_CHECK(cudaMemcpy(x, b, sizeof(float)*n, cudaMemcpyDeviceToDevice));
    CUSOLVER_CHECK(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, (float*)A, lda, ipiv, (float*)x, n, info));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (info  ) { CUDA_CHECK(cudaFree(info  )); }
    if (buffer) { CUDA_CHECK(cudaFree(buffer)); }
    if (A     ) { CUDA_CHECK(cudaFree(A)); }
    if (ipiv  ) { CUDA_CHECK(cudaFree(ipiv));}

    return 0;
}

int linearSolverQR(
        cusolverDnHandle_t handle,
        int n,
        const c_float *Acopy,
        int lda,
        const c_float *b,
        c_float *x)
{
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    int bufferSize = 0;
    int *info = NULL;
    c_float *buffer = NULL;
    c_float *A = NULL;
    c_float *tau = NULL;
    int h_info = 0;
    const float one = 1.0;

    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(handle, n, n, (float*)Acopy, lda, &bufferSize));

    CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffer, sizeof(c_float)*bufferSize));
    CUDA_CHECK(cudaMalloc(&A, sizeof(c_float)*lda*n));
    CUDA_CHECK(cudaMalloc ((void**)&tau, sizeof(c_float)*n));

// prepare a copy of A because getrf will overwrite A with L
    CUDA_CHECK(cudaMemcpy(A, Acopy, sizeof(c_float)*lda*n, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemset(info, 0, sizeof(int)));


// compute QR factorization
    CUSOLVER_CHECK(cusolverDnSgeqrf(handle, n, n, (float*)A, lda, (float*)tau, (float*)buffer, bufferSize, info));

    CUDA_CHECK(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    CUDA_CHECK(cudaMemcpy(x, b, sizeof(c_float)*n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    CUSOLVER_CHECK(cusolverDnSormqr(
            handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_OP_T,
            n,
            1,
            n,
            (float*)A,
            lda,
            (float*)tau,
            (float*)x,
            n,
            (float*)buffer,
            bufferSize,
            info));

    // x = R \ Q^T*b
    CUBLAS_CHECK(cublasStrsm(
            cublasHandle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            n,
            1,
            &one,
            (float*)A,
            lda,
            (float*)x,
            n));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (cublasHandle) { CUBLAS_CHECK(cublasDestroy(cublasHandle)); }
    if (info  ) { CUDA_CHECK(cudaFree(info  )); }
    if (buffer) { CUDA_CHECK(cudaFree(buffer)); }
    if (A     ) { CUDA_CHECK(cudaFree(A)); }
    if (tau   ) { CUDA_CHECK(cudaFree(tau)); }

    return 0;
}

c_int linearSolverLDL(cusolverDnHandle_t handle,
                      int n,
                      const c_float *Acopy,
                      int lda,
                      const c_float *b,
                      c_float *x){
    int bufferSize = 0;
    int *info = NULL;
    c_float *buffer = NULL;
    c_float *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;

    const cublasFillMode_t oplo = CUBLAS_FILL_MODE_LOWER;
    /*Malloc space for solving problems*/
    CUSOLVER_CHECK(cusolverDnSsytrf_bufferSize(handle, n, (float*)Acopy, lda, &bufferSize));

    CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffer, sizeof(c_float)* 2 * bufferSize));
    CUDA_CHECK(cudaMalloc(&A, sizeof(c_float)*lda*n));
    CUDA_CHECK(cudaMalloc(&ipiv, sizeof(int)*n));

    // prepare a copy of A because getrf will overwrite A with L
    CUDA_CHECK(cudaMemcpy(A, Acopy, sizeof(c_float)*lda*n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(info, 0, sizeof(int)));

    CUSOLVER_CHECK(cusolverDnSsytrf(handle, oplo, n, A, lda, ipiv, buffer, n, info));
    CUDA_CHECK(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LDL factorization failed\n");
    }
//    c_float *fack_news;
//    if(n ==5) {
//        c_float t[] = {
//                    1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
//                    0.0f, 2.0f, 0.0f, 0.0f, 0.0f,
//                    0.0f, 0.0f, 3.0f, 0.0f, 0.0f,
//                    0.0f, 0.0f, 0.0f, 4.0f, 0.0f,
//                    0.0f, 0.0f, 0.0f, 0.0f, 5.0f
//                    };
//        fack_news = t;
//    }else if(n ==4){
//        c_float t[] = {
//                1.0f, 0.0f, 0.0f, 0.0f,
//                0.0f, 2.0f, 0.0f, 0.0f,
//                0.0f, 0.0f, 3.0f, 0.0f,
//                0.0f, 0.0f, 0.0f, 4.0f,
//                };
//        fack_news = t;
//    }
//    CUDA_CHECK(cudaMemcpy(A, fack_news, sizeof(c_float) * n * n, cudaMemcpyHostToDevice));
    c_float* tmp, *test;
    CUDA_CHECK(cudaMalloc((void**)&tmp, sizeof(c_float) * n *n ));
    test = (c_float*)malloc(sizeof(c_float) * n *n);
    CUDA_CHECK(cudaMemcpy(tmp, A, sizeof(c_float) * n *n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(test, tmp, sizeof(c_float) * n *n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(tmp));
//    for(int i = 0; i <n ;i++){
//        for(int j = 0; j < n; j++){
//            printf("A[%d][%d]: %f", i, j, test[i*n+j]);
//        }
//        printf("\n");
//    }
    CUDA_CHECK(cudaMemcpy(x, b, sizeof(c_float)*n, cudaMemcpyDeviceToDevice));
    CUSOLVER_CHECK(cusolverDnSsytrs(handle, oplo, n, 1, A, lda, ipiv, x, n, buffer, bufferSize, info));// calculated but not calculated at all
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LDL factorization failed\n");
    }

    if (info  ) { CUDA_CHECK(cudaFree(info  )); }
    if (buffer) { CUDA_CHECK(cudaFree(buffer)); }
    if (A     ) { CUDA_CHECK(cudaFree(A)); }
    if (ipiv  ) { CUDA_CHECK(cudaFree(ipiv));}
}

/**
* This function performs solving the following linear system using LDLT method:
*  Ax = b
*  We assume the coefficient matrix is symmetrical.
* @param A
* @param b
* @return solved answer: x
*/
c_float* cuda_ldlt_solve(c_float *A, c_float* b, int m){
    cusolverDnHandle_t handle = NULL;
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    cudaStream_t stream = NULL;

    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

    c_float *x;
    cudaMalloc((void**)&x, sizeof(c_float)*m);
    linearSolverQR(handle, m, A, m, b, x);
//    linearSolverLDL(handle, m, A, m, b, x);
    return x;
    cudaFree(handle);
    cudaFree(cublasHandle);
    cudaFree(stream);
}

c_float* cuda_LDL_alg(cudapcg_solver *s, c_float *rhs){

    c_float *P_dev = csrMats2Dense(s->P);
    c_float* A_dev = csrMats2Dense(s->A);
    c_float* At_dev = csrMats2Dense(s->At);
    c_float rho = -1.0f/(*s->h_rho); //-(rho)^-1
    size_t mat_size = s->m + s->n;

    c_float *mat_dev = (c_float*)malloc(mat_size * mat_size * sizeof(c_float));
    memset(mat_dev, 0.0f, mat_size * mat_size * sizeof(c_float));

    mergeMatrices(mat_dev, P_dev, mat_size, mat_size, s->P->m, s->P->n, 0,0);
    mergeMatrices(mat_dev, A_dev, mat_size, mat_size, s->A->m, s->A->n, s->P->m, 0);
    mergeMatrices(mat_dev, At_dev, mat_size, mat_size, s->At->m, s->At->n, 0, s->P->n);
    mergeScalar2Matrix(mat_dev, rho, mat_size, mat_size, s->A->m, s->P->m, s->P->n);
//        for(int i = 0; i<mat_size; i++){
//            for(int j = 0; j<mat_size; j++){
//                printf("mat[%d][%d]: %f ", i, j, mat_dev[i * mat_size + j]);
//            }
//            printf("\n");
//        }
//        printf("\n");
    /* So far we have extract the A matrix of the linear system */
    /* The next step is to move the A matrix from cpu to gpu   */
    c_float *mat_a;
    checkCudaErrors(cudaMalloc((void**)&mat_a, mat_size * mat_size * sizeof(c_float)));
    checkCudaErrors(cudaMemcpy(mat_a, mat_dev, mat_size * mat_size * sizeof(c_float), cudaMemcpyHostToDevice));

    c_float *tmp, *rhs_;
    cudaMalloc((void**)&tmp, sizeof(c_float)*mat_size);
    rhs_ =  (c_float*)malloc(sizeof(c_float)*mat_size);
    cudaMemcpy(tmp, rhs, sizeof(c_float)*mat_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(rhs_, tmp, sizeof(c_float)*mat_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaFree(tmp));
//    for(int i = 0; i< mat_size; i++){
//        printf("RHS[%d]: %f, ", i, rhs_[i]);
//    }
//    printf("\n");
    /* Now, we start to solve the linear equation using cuSolver */
    /* Because the coefficient matrix is symmetrical, we use LDLT method to solve */
    c_float *x = cuda_ldlt_solve(mat_a, tmp, mat_size); //Calculated but not calculated at all
//    CUDA_CHECK(cudaMemcpy(rhs_, x, sizeof(c_float) * mat_size, cudaMemcpyDeviceToHost));
//    for(int i = 0; i< 2; i++){
//        printf("x[%d]: %f ", i, rhs_[i]);
//    }
//    printf("\n");
    cudaFree(mat_a);
    cudaFree(tmp);
    return x;
}

c_int cuda_pcg_alg(cudapcg_solver *s,
                   c_float         eps,
                   c_int           max_iter) {

  c_float *tmp;

  c_int iter = 0;
  c_int n    = s->n;
  c_float H_MINUS_ONE = -1.0;

  if (!s->warm_start) {
    /* d_x = 0 */
    checkCudaErrors(cudaMemset(s->d_x, 0, n * sizeof(c_float)));
  }

  /* d_p = 0 */
  checkCudaErrors(cudaMemset(s->d_p, 0, n * sizeof(c_float)));

  /* d_r = K * d_x */
  mat_vec_prod(s, s->d_r, s->d_x, 0);

  /* d_r -= d_rhs */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &H_MINUS_ONE, s->d_rhs, 1, s->d_r, 1));

  /* h_r_norm = |d_r| */
  s->vector_norm(s->d_r, n, s->h_r_norm);

  /* From here on cuBLAS is operating in device pointer mode */
  cublasSetPointerMode(CUDA_handle->cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

  if (s->precondition) {
    /* d_y = M \ d_r */
    cuda_vec_ew_prod(s->d_y, s->d_diag_precond_inv, s->d_r, n);
  }

  /* d_p = -d_y */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->D_MINUS_ONE, s->d_y, 1, s->d_p, 1));

  /* rTy = d_r' * d_y */
  checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, s->d_y, 1, s->d_r, 1, s->rTy));

  cudaDeviceSynchronize();

  /* Run the PCG algorithm */
  while ( *(s->h_r_norm) > eps && iter < max_iter ) {

    /* d_Kp = K * d_p */
    mat_vec_prod(s, s->d_Kp, s->d_p, 1);

    /* pKp = d_p' * d_Kp */
    checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, s->d_p, 1, s->d_Kp, 1, s->pKp));

    /* alpha = rTy / pKp */
    scalar_division_kernel<<<1,1>>>(s->alpha, s->rTy, s->pKp);

    /* d_x += alpha * d_p */
    checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->alpha, s->d_p, 1, s->d_x, 1));

    /* d_r += alpha * d_Kp */
    checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->alpha, s->d_Kp, 1, s->d_r, 1));

    if (s->precondition) {
      /* d_y = M \ d_r */
      cuda_vec_ew_prod(s->d_y, s->d_diag_precond_inv, s->d_r, n);
    }

    /* Swap pointers to rTy and rTy_prev */
    tmp = s->rTy_prev;
    s->rTy_prev = s->rTy;
    s->rTy = tmp;

    /* rTy = d_r' * d_y */
    checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, s->d_y, 1, s->d_r, 1, s->rTy));

    /* Update residual norm */
    s->vector_norm(s->d_r, n, s->d_r_norm);
    checkCudaErrors(cudaMemcpyAsync(s->h_r_norm, s->d_r_norm, sizeof(c_float), cudaMemcpyDeviceToHost));

    /* beta = rTy / rTy_prev */
    scalar_division_kernel<<<1,1>>>(s->beta, s->rTy, s->rTy_prev);

    /* d_p *= beta */
    checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, s->beta, s->d_p, 1));

    /* d_p -= d_y */
    checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->D_MINUS_ONE, s->d_y, 1, s->d_p, 1));

    cudaDeviceSynchronize();
    iter++;

  } /* End of the PCG algorithm */

  /* From here on cuBLAS is operating in host pointer mode again */
  cublasSetPointerMode(CUDA_handle->cublasHandle, CUBLAS_POINTER_MODE_HOST);

  return iter;
}


void cuda_pcg_update_precond(cudapcg_solver *s,
                             c_int           P_updated,
                             c_int           A_updated,
                             c_int           R_updated) {

  void    *buffer;
  c_float *tmp;
  c_int    n  = s->n;
  csr     *At = s->At;

  size_t Buffer_size_in_bytes = n * (sizeof(c_float) + sizeof(c_int));

  if (!P_updated && !A_updated && !R_updated) return;

  if (P_updated) {
    /* Update d_P_diag_val */
    cuda_vec_gather(n, s->P->val, s->d_P_diag_val, s->d_P_diag_ind);
  }

  if (A_updated || R_updated) {
    /* Allocate memory */
    cuda_malloc((void **) &tmp, At->nnz * sizeof(c_float));
    cuda_malloc((void **) &buffer, Buffer_size_in_bytes);

    /* Update d_AtRA_diag_val */
    if (!s->d_rho_vec) {  /* R = rho*I  -->  A'*R*A = rho * A'*A */

      if (A_updated) {
        /* Update d_AtA_diag_val */
        cuda_vec_ew_prod(tmp, At->val, At->val, At->nnz);
        cuda_vec_segmented_sum(tmp, At->row_ind, s->d_AtA_diag_val, buffer, n, At->nnz);
      }

      /* d_AtRA_diag_val = rho * d_AtA_diag_val */
      cuda_vec_add_scaled(s->d_AtRA_diag_val, s->d_AtA_diag_val, NULL, *s->h_rho, 0.0, n);
    }
    else {    /* R = diag(d_rho_vec)  -->  A'*R*A = A' * diag(d_rho_vec) * A */
      cuda_mat_rmult_diag_new(At, tmp, s->d_rho_vec);   /* tmp = A' * R */
      cuda_vec_ew_prod(tmp, tmp, At->val, At->nnz);     /* tmp = tmp * A */
      cuda_vec_segmented_sum(tmp, At->row_ind, s->d_AtRA_diag_val, buffer, n, At->nnz);
    }

    /* Free memory */
    cuda_free((void **) &tmp);
    cuda_free((void **) &buffer);
  }

  /* d_diag_precond = sigma */
  cuda_vec_set_sc(s->d_diag_precond, *s->h_sigma, n);

  /* d_diag_precond += d_P_diag_val + d_AtRA_diag_val */
  cuda_vec_add_scaled3(s->d_diag_precond, s->d_diag_precond, s->d_P_diag_val, s->d_AtRA_diag_val, 1.0, 1.0, 1.0, n);

  /* d_diag_precond_inv = 1 / d_diag_precond */
  cuda_vec_reciprocal(s->d_diag_precond_inv, s->d_diag_precond, n);
}
