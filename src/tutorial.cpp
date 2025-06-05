#include "tutorial.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"
#include "ggml.h"
#include <cstring>
#include <iostream>
#include <random>

void TutorialRegression::set_params(const float a, const float b) {
  // This just sets the values of `_a` and `_b`. This is used once we are 
  // ready to perform the computation (see `forward`).
  ggml_set_f32(_a, a);
  ggml_set_f32(_b, b);
}

float TutorialRegression::forward(const float x) {
  // Set the input tensor `_x`
  ggml_set_f32(_x, x);
  // Create a new graph using the context
  struct ggml_cgraph *cf = ggml_new_graph(_ctx);
  // Use the new graph and the output tensor to build the graph outwards
  ggml_build_forward_expand(cf, _result);
  // Use the graph and context to compute the result (use 1 thread)
  ggml_graph_compute_with_ctx(_ctx, cf, 1);
  // get the result from the tensor (since this tensor holds 1 value, we pass index 0).
  return ggml_get_f32_1d(_result, 0);
}

template<typename T>
void BackendRegression<T>::set_params(const T a, const T b) {
  // Similar to set_params for TutorialRegression. But now, we are using a backend.
  // We have to use ggml_backend_tensor_set since ggml_set_f32 is used specifically for the CPU backend. 
  ggml_backend_tensor_set(_a, &a, 0, ggml_nbytes(_a));
  ggml_backend_tensor_set(_b, &b, 0, ggml_nbytes(_b));
}

template <typename T>
T BackendRegression<T>::forward(const T x) {
  // Again, we use ggml_backend_tensor_set instead of ggml_set_f32
  ggml_backend_tensor_set(_x, &x, 0, ggml_nbytes(_x));
  // We already built and allocated the graph in the constructor. 
  // Now we just have to do the computation using the backend.
  ggml_backend_graph_compute(_backend, _gf);
  // The result is stored in _result, but this shows an alternate way to get it.
  // We know that the last node in the graph is the result, we we fetch the result node.
  struct ggml_tensor *result = ggml_graph_node(_gf, -1);
  // Now, we use the backend to get the data.
  T result_data = 0;
  ggml_backend_tensor_get(result, &result_data, 0, ggml_nbytes(result));
  return result_data;
}

template<typename T>
void BackendRegression<T>::train(const DataLoader<T> &dl) {
  /* We train the model on the dataset. Here are the parameters:
   * backend_sched: The backend scheduler (we made this in the constructor) 
   * ctx_compute: The compute context (we made this in the constructor, too)
   * inputs: In this case, it is the tensor, `_x`. 
   * outputs: In this case, it is the tensor, `_result`. 
   * dataset: A ggml_opt_dataset_t. We made this inside of the DataLoader `dl`.
   * loss_type: A loss type (in this case, we use mean squared error). 
   * get_opt_pars: A callback that returns the ggml_opt_get_optimizer_params. In this case, we use 
   * the default parameters.
   * nepoch: The number of epochs.
   * nbatch_logical: How many values per batch.
   * val_split: What percentage of the data should we use for validation? We don't really need this in this example. 
   * silent: do not print diagnostic output to stderr */
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

int main() { 
  /* TutorialRegression: Compute a * x + b where a = 3, b = 4, x = 5, (end result should be 19) */
  TutorialRegression regressor;
  regressor.set_params(3.0f, 4.0f);
  float result = regressor.forward(5.0f);
  std::cout << "Tutorial Result: " << result << "\n";
  /* Do the same thing as TutorialRegression, but with a backend (end result should be 19) */
  BackendRegression<float> backend_regressor;
  backend_regressor.set_params(3.0f, 4.0f);
  result = regressor.forward(5.0f);
  std::cout << "Backend result: " << result << "\n";
  // This part does not run using WASM.
#ifndef EMSCRIPTEN
  /* Create 10000 datapoints; first column is x, second column is y. This is our "dataset" */
  const int N = 10000;
  float matrix[N][2];
  // Randomly generate the parameters a and b.
  std::memset(matrix, 0, 2 * N * sizeof(float));
  std::uniform_real_distribution<float> unif(1, 10);
  std::default_random_engine re;
  double a = unif(re);
  double b = unif(re);
  std::cout << "Parameters to recover: a=" << a << "; b=" << b << "\n";
  // Compute a * x + b for integer x in the interval [1, N].
  for (int i = 0; i < N; i++) {
    matrix[i][0] = static_cast<float>(i+1);
    matrix[i][1] = a * matrix[i][0] + b;
  }
  // Use the DataLoader on the matrix to create a GGML dataset.
  DataLoader<float> dl(matrix, N);
  // Train the backend regressor on the dataset.
  backend_regressor.train(dl);
  // Print the results, and evaluate at the points x = 15000, x = 20000, x = 30000
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
#endif
  return 0;
}
