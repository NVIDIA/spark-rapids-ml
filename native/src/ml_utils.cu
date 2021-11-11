#include <cudf/column/column_factories.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/eig.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/matrix/math.cuh>
#include <rmm/exec_policy.hpp>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

cublasOperation_t convertToCublasOpEnum(int int_type) {
  switch(int_type) {
    case 0: return CUBLAS_OP_N;
    case 1: return CUBLAS_OP_T;
    case 2: return CUBLAS_OP_C;
    case 3: return CUBLAS_OP_CONJG;
    default:
      throw "Invalid type enum: " + std::to_string(int_type);
      break;
  }
}

void signFlip(
  double* input, int n_rows, int n_cols, double* components, int n_cols_comp, cudaStream_t stream) {
  auto counting = thrust::make_counting_iterator(0);
  auto m        = n_rows;

  thrust::for_each(rmm::exec_policy(stream), counting, counting + n_cols, [=] __device__(int idx) {
    int d_i = idx * m;
    int end = d_i + m;

    double max    = 0.0;
    int max_index = 0;
    for (int i = d_i; i < end; i++) {
      double val = input[i];
      if (val < 0.0) { val = -val; }
      if (val > max) {
        max       = val;
        max_index = i;
      }
    }

    if (input[max_index] < 0.0) {
      for (int i = d_i; i < end; i++) {
        input[i] = -input[i];
      }
    }
  });
}

long dgemmWithColumnViewPtr(int transa, int transb, int m, int n,
                            int k, double alpha, double* A, int size_A, int lda, long B,
                            int ldb, double beta, int ldc, int deviceID) {
    cudaSetDevice(deviceID);
    raft::handle_t raft_handle;
    cudaStream_t stream = raft_handle.get_stream();
    auto const *B_cv_ptr = reinterpret_cast<cudf::lists_column_view const *>(B);
    auto const child_column_view = B_cv_ptr->child();
    // init cuda stream view from rmm
    auto c_stream = rmm::cuda_stream_view(stream);

    rmm::device_buffer dev_buff_A = rmm::device_buffer(A, size_A * sizeof(double), c_stream);

    auto size_C = m * n;
    //create child column that will own the computation result
    auto child_column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64}, size_C);
    auto child_mutable_view = child_column->mutable_view();
    auto status = raft::linalg::cublasgemm(raft_handle.get_cublas_handle(), convertToCublasOpEnum(transa), convertToCublasOpEnum(transb),
                                           m, n, k, &alpha, (double const *)dev_buff_A.data(), lda, child_column_view.data<double>(),
                                           ldb, &beta, child_mutable_view.data<double>(), ldc, stream);
    // create offset column
    auto zero = cudf::numeric_scalar<int32_t>(0, true, c_stream);
    auto step = cudf::numeric_scalar<int32_t>(m, true, c_stream);
    std::unique_ptr<cudf::column> offset_column = cudf::sequence(n + 1, zero, step, rmm::mr::get_current_device_resource());

    auto target_column = cudf::make_lists_column(n, std::move(offset_column), std::move(child_column), 0, rmm::device_buffer());

    return reinterpret_cast<long>(target_column.release());
}
