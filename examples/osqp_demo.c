#include "osqp.h"
#include <stdlib.h>
#include "../algebra/cuda/include/csr_type.h"
#include "../algebra/cuda/include/cuda_csr.h"
#include "util.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <cublas_api.h>

#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
        }                                                                                          \
    }while(0)

#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUSPARSE_CHECK(err)                                                                        \
    do {                                                                                           \
        cusparseStatus_t err_ = (err);                                                             \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                                     \
            printf("cusparse error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusparse error");                                            \
        }                                                                                          \
    } while (0)

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main(void) {
    c_int exitflag;

  /* Load problem data */
  c_float P_x[3] = { 4.0, 1.0, 2.0, };
  c_int   P_nnz  = 3;
  c_int   P_i[3] = { 0, 0, 1, };
  c_int   P_p[3] = { 0, 1, 3, };
  c_float q[2]   = { 1.0, 1.0, };
  c_float A_x[4] = { 1.0, 1.0, 1.0, 1.0, };
  c_int   A_nnz  = 4;
  c_int   A_i[4] = { 0, 1, 0, 2, };
  c_int   A_p[3] = { 0, 2, 4, };
  c_float l[3]   = { 1.0, 0.0, 0.0, };
  c_float u[3]   = { 1.0, 0.7, 0.7, };
  c_int n = 2;
  c_int m = 3;

  /* Exitflag */
    clock_t start, end;
    double dur;
    start = clock();
for(int i = 0; i< 1000; i++) {
  /* Workspace, settings, matrices */
  printf("---------------%d------------\n", i);
  OSQPSolver   *solver;
  OSQPSettings *settings;
  csc *P = malloc(sizeof(csc));
  csc *A = malloc(sizeof(csc));

  /* Populate matrices */

      csc_set_data(A, m, n, A_nnz, A_x, A_i, A_p);
      csc_set_data(P, n, n, P_nnz, P_x, P_i, P_p);

      /* Set default settings */
      settings = (OSQPSettings *) malloc(sizeof(OSQPSettings));
      if (settings) osqp_set_default_settings(settings);
      settings->polish = 1;

      /* Setup workspace */

      exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);
//      end = clock();

//      printf("Setup Timecost: %f\n ", dur);
      /* Solve Problem */

//      start = clock();
      osqp_solve(solver);
//      end = clock();
//      dur = (double) (end - start) / CLOCKS_PER_SEC;
//      printf("Solve Time cost: %f\n ", dur);

    for(int i =0; i<2;i++){
        printf("%f ", solver->solution->x[i]);
    }
    printf("\n");
      /* Clean workspace */
//      start = clock();
      osqp_cleanup(solver);
      free(A);
      free(P);
      free(settings);
//      end = clock();
  }
    end = clock();
    dur = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Test time cost: %f\n ", dur);
  return exitflag;
}