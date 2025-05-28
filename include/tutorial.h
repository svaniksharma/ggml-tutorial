#include <type_traits>
#ifndef TUTORIAL_GGML_H

#include <vector>
#include <iostream>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"
#include "ggml-cuda.h"

class TutorialRegression {
public:
  TutorialRegression() {
    struct ggml_init_params params {
      16 * 1024 * 1024,
      nullptr,
      false
    };
    _ctx = ggml_init(params); 
    _a = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, 1);
    _b = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, 1);
    _x = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, 1);
    ggml_set_param(_x);
    struct ggml_tensor *ax = ggml_mul(_ctx, _a, _x);
    _result = ggml_add(_ctx, ax, _b);
    ggml_set_output(_result);
  }
  void set_params(const float a, const float b);
  float forward(const float x);
  ~TutorialRegression() {
    ggml_free(_ctx);
  }
private:
  struct ggml_tensor *_a;
  struct ggml_tensor *_b;
  struct ggml_tensor *_x;
  struct ggml_context *_ctx;
  struct ggml_tensor *_result;
};

template<typename T>
class DataLoader {
public:
  DataLoader(const float matrix[][2], const size_t N) {
    enum ggml_type dataset_type;
    if (std::is_same<T, float>::value)
      dataset_type = GGML_TYPE_F32;
    else if (std::is_same<T, double>::value)
      dataset_type = GGML_TYPE_F64;
    else
      GGML_ASSERT(false);
    _dataset = ggml_opt_dataset_init(dataset_type, dataset_type, 1, 1, N, 1);
    struct ggml_tensor *data = ggml_opt_dataset_data(_dataset);
    struct ggml_tensor *labels = ggml_opt_dataset_labels(_dataset);
    T *data_buf = static_cast<T *>(ggml_get_data(data));
    T *labels_buf = static_cast<T *>(ggml_get_data(labels));
    for (int i = 0; i < N; i++) {
      data_buf[i] = matrix[i][0];
      labels_buf[i] = matrix[i][1];
    }
  }
  ggml_opt_dataset_t get_dataset() const { return _dataset; }
  ~DataLoader() {
    ggml_opt_dataset_free(_dataset);
  }
private:
  ggml_opt_dataset_t _dataset;
};

template <typename T>
class BackendRegression {
public:
  BackendRegression() {
    struct ggml_init_params params = {
      1024 * ggml_tensor_overhead(),
      nullptr,
      true
    };
    enum ggml_type tensor_type;
    if (std::is_same<T, float>::value)
      tensor_type = GGML_TYPE_F32;
    else if (std::is_same<T, double>::value)
      tensor_type = GGML_TYPE_F64;
    else
      GGML_ASSERT(false);
    _ctx_static = ggml_init(params);
    _a = ggml_new_tensor_1d(_ctx_static, tensor_type, 1);
    _b = ggml_new_tensor_1d(_ctx_static, tensor_type, 1);
    _x = ggml_new_tensor_1d(_ctx_static, tensor_type, 1);
    ggml_set_input(_x);
    ggml_set_param(_a);
    ggml_set_param(_b);
    _backend = ggml_backend_cuda_init(0);
    _backend_buffer = ggml_backend_alloc_ctx_tensors(_ctx_static, _backend);
    params = {
      ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + 3 * ggml_graph_overhead(),
      nullptr,
      true
    };
    _ctx_compute = ggml_init(params);
    struct ggml_tensor *ax = ggml_mul(_ctx_compute, _a, _x);
    _result = ggml_add(_ctx_compute, ax, _b);
    ggml_set_output(_result);
    std::vector<ggml_backend_t> backends;
    backends.push_back(_backend);
    backends.push_back(ggml_backend_cpu_init());
    _backend_sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);
    std::cout << "Using " << ggml_backend_name(_backend) << " as backend\n";
    _gf = ggml_new_graph(_ctx_compute);
    ggml_build_forward_expand(_gf, _result);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(_backend));
    ggml_gallocr_alloc_graph(allocr, _gf); 
  }
  void set_params(const T a, const T b);
  float forward(const T x);
  void train(const DataLoader<T> &dl);
  void print_params() const;
  ~BackendRegression() {
    ggml_free(_ctx_static);
    ggml_free(_ctx_compute);
    ggml_backend_free(_backend);
    ggml_backend_buffer_free(_backend_buffer);
    ggml_backend_sched_free(_backend_sched);
    ggml_gallocr_free(_allocr);
  }
private:
  struct ggml_tensor *_a;
  struct ggml_tensor *_b;
  struct ggml_tensor *_x;
  struct ggml_tensor *_result;
  struct ggml_context *_ctx_static;
  struct ggml_context *_ctx_compute;
  struct ggml_backend *_backend;
  ggml_backend_buffer_t _backend_buffer;
  ggml_backend_sched_t _backend_sched;
  struct ggml_cgraph *_gf;
  ggml_gallocr_t _allocr;
};

#endif // !TUTORIAL_GGML_H


