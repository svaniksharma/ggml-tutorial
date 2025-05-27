#include "tutorial.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <iostream>

void TutorialRegression::set_params(const float a, const float b) {
  ggml_set_f32(_a, a);
  ggml_set_f32(_b, b);
}

float TutorialRegression::forward(const float x) {
  ggml_set_f32(_x, x);
  struct ggml_cgraph *cf = ggml_new_graph(_ctx);
  ggml_build_forward_expand(cf, _result);
  ggml_graph_compute_with_ctx(_ctx, cf, 1);
  return ggml_get_f32_1d(_result, 0);
}

void BackendRegression::set_params(const float a, const float b) {
  ggml_set_f32(_a, a);
  ggml_set_f32(_b, b);
}

float BackendRegression::forward(const float x) {
  ggml_set_f32(_x, x);
  struct ggml_cgraph *cf = ggml_new_graph(_ctx_compute);
  ggml_build_forward_expand(cf, _result);
  ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(_backend));
  ggml_gallocr_alloc_graph(allocr, cf);
  ggml_backend_graph_compute(_backend, cf);
  float result = ggml_get_f32_1d(_result, 0);
  ggml_gallocr_free(allocr);
  return result;
}

int main (int argc, char *argv[]) {
  TutorialRegression regressor;
  regressor.set_params(3.0f, 4.0f);
  float result = regressor.forward(5.0f);
  std::cout << "Tutorial Result: " << result << "\n";
  BackendRegression backend_regressor;
  backend_regressor.set_params(3.0f, 4.0f);
  result = regressor.forward(5.0f);
  std::cout << "Backend result: " << result << "\n";
  return 0;
}
