#ifndef TUTORIAL_GGML_H

#include <string>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-cpu.h"
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

class BackendRegression {
public:
  BackendRegression() {
    struct ggml_init_params params = {
      1024 * ggml_tensor_overhead(),
      nullptr,
      true
    };
    _ctx_static = ggml_init(params);
    _a = ggml_new_tensor_1d(_ctx_static, GGML_TYPE_F32, 1);
    _b = ggml_new_tensor_1d(_ctx_static, GGML_TYPE_F32, 1);
    _x = ggml_new_tensor_1d(_ctx_static, GGML_TYPE_F32, 1);
    ggml_set_param(_x);
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
  }
  void set_params(const float a, const float b);
  float forward(const float x);
  ~BackendRegression() {
    ggml_free(_ctx_static);
    ggml_free(_ctx_compute);
    ggml_backend_free(_backend);
    ggml_backend_buffer_free(_backend_buffer);
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
};

#endif // !TUTORIAL_GGML_H


