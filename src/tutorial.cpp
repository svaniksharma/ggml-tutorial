#include "tutorial.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <iostream>

void TutorialRegression::set_params(float a, float b) {
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

int main (int argc, char *argv[]) {
  TutorialRegression regressor;
  regressor.set_params(3.0f, 4.0f);
  float result = regressor.forward(5.0f);
  std::cout << "Result: " << result << "\n";
  return 0;
}
