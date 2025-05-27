#include "tutorial.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"
#include "ggml.h"
#include <cstring>
#include <iostream>
#include <random>

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
  ggml_backend_tensor_set(_a, &a, 0, ggml_nbytes(_a));
  ggml_backend_tensor_set(_b, &b, 0, ggml_nbytes(_b));
}

float BackendRegression::forward(const float x) {
  ggml_backend_tensor_set(_x, &x, 0, ggml_nbytes(_x));
  struct ggml_cgraph *cf = ggml_new_graph(_ctx_compute);
  ggml_build_forward_expand(cf, _result);
  ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(_backend));
  ggml_gallocr_alloc_graph(allocr, cf);
  ggml_backend_graph_compute(_backend, cf);
  float result = ggml_get_f32_1d(_result, 0);
  ggml_gallocr_free(allocr);
  return result;
}

void BackendRegression::train(const DataLoader &dl) {
  ggml_opt_fit(_backend_sched, _ctx_compute, _x, _result, dl.get_dataset(), GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, ggml_opt_get_default_optimizer_params, 5, 1, 0.2f, true);
}

void BackendRegression::print_params() const {
  float a = 0;
  float b = 0;
  ggml_backend_tensor_get(_a, &a, 0, ggml_nbytes(_a));
  ggml_backend_tensor_get(_b, &b, 0, ggml_nbytes(_b));
  std::cout << "a: " << a << "\n";
  std::cout << "b: " << b << "\n";
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
  const int N = 10000;
  float matrix[N][2];
  std::memset(matrix, 0, 20 * sizeof(float));
  std::uniform_real_distribution<float> unif(1, 10);
  std::default_random_engine re;
  double a = unif(re);
  double b = unif(re);
  std::cout << "Parameters to recover: a=" << a << "; b=" << b << "\n";
  for (int i = 0; i < N; i++) {
    matrix[i][0] = static_cast<float>(i+1);
    matrix[i][1] = a * matrix[i][0] + b;
  }
  DataLoader dl(matrix, N);
  backend_regressor.train(dl);
  std::cout << "Recovered parameters\n---------------\n";
  backend_regressor.print_params();
  return 0;
}
