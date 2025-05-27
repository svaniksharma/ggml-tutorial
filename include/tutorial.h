#ifndef TUTORIAL_GGML_H
#include "ggml.h"

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

#endif // !TUTORIAL_GGML_H


