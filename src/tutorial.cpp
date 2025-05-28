#include "tutorial.h"
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

template<typename T>
void BackendRegression<T>::set_params(const T a, const T b) {
  ggml_backend_tensor_set(_a, &a, 0, ggml_nbytes(_a));
  ggml_backend_tensor_set(_b, &b, 0, ggml_nbytes(_b));
}

template <typename T>
float BackendRegression<T>::forward(const T x) {
  ggml_backend_tensor_set(_x, &x, 0, ggml_nbytes(_x));
  ggml_backend_graph_compute(_backend, _gf);
  struct ggml_tensor *result = ggml_graph_node(_gf, -1);
  float result_data = 0;
  ggml_backend_tensor_get(result, &result_data, 0, ggml_nbytes(result));
  return result_data;
}

template<typename T>
void BackendRegression<T>::train(const DataLoader<T> &dl) {
  ggml_opt_fit(_backend_sched, _ctx_compute, _x, _result, dl.get_dataset(), GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, ggml_opt_get_default_optimizer_params, 5, 1, 0.2f, true);
}

template<typename T>
void BackendRegression<T>::print_params() const {
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
  BackendRegression<float> backend_regressor;
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
  DataLoader<float> dl(matrix, N);
  backend_regressor.train(dl);
  std::cout << "Recovered parameters\n---------------\n";
  backend_regressor.print_params();
  std::cout << "Evaluation on test data\n------------\n";
  float test_x[] = { 15000.0f, 20000.0f, 30000.0f };
  for (int i = 0; i < sizeof(test_x) / sizeof(float); i++) {
    auto x = test_x[i];
    float y = a * x + b;
    float y_pred = backend_regressor.forward(x);
    std::cout << "x = " << x << "\n";
    std::cout << "y: " << y << "\n";
    std::cout << "y pred: " << y_pred << "\n";
  }
  return 0;
}
