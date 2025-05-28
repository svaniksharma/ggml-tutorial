#ifndef TUTORIAL_GGML_H

#include <type_traits>
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
      /* .mem_size = */ 1024 * ggml_tensor_overhead(), 
      /* .mem_buffer = */ nullptr,
      /* .no_alloc = */ false
    };
    /* We initialize a context. 
     * The context keeps track of the tensors and operations between them. 
     * In this case, we are also using it to perform the computation. */
    _ctx = ggml_init(params); 
    // Using the context, we construct the computational graph for y = a * x + b.
    // Note that no computation is actually being performed.
    // This is only done to build the graph. 
    _a = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, 1);
    _b = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, 1);
    _x = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor *ax = ggml_mul(_ctx, _a, _x);
    _result = ggml_add(_ctx, ax, _b);
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
  DataLoader(const T matrix[][2], const size_t N) {
    // This can work for either double or float.
    enum ggml_type dataset_type;
    if (std::is_same<T, float>::value)
      dataset_type = GGML_TYPE_F32;
    else if (std::is_same<T, double>::value)
      dataset_type = GGML_TYPE_F64;
    else
      GGML_ASSERT(false);
    /* This creates a dataset. The arguments are as follows:
     * type_data: We set this above: 32-bit float or 64-bit float.
     * type_label: Same as above. In this case, it's the same type.
     * ne_datapoint: Number of elements per datapoint (i.e, how many features per datapoint) 
     * ne_label: Number of elements per label (i.e, how many outputs/targets/dependent variables do you have)
     * ndata: The number of datapoints and labels
     * ndata_shard: A shard is the unit along which a datapoint is shuffled. This is the number of points per shard. */
    _dataset = ggml_opt_dataset_init(dataset_type, dataset_type, 1, 1, N, 1);
    // Once the dataset is created, the underlying tensors are allocated for you based on the arguments passed above. ^^^
    // The following code gets the underlying tensors, and then uses ggml_get_data to get the actual buffer inside of the tensor. We then use the matrix passed in the constructor (first column datapoint, second column labels) to set these underlying buffers.
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
    // This can work for either double or float.
    // Notice that .no_alloc is true here. We want to allocate memory explicitly.
    struct ggml_init_params params = {
      /* .mem_size = */ 1024 * ggml_tensor_overhead(),
      /* .mem_buffer = */ nullptr,
      /* .no_alloc = */ true
    };
    // Like before, we determine whether we are dealing with float or double.
    enum ggml_type tensor_type;
    if (std::is_same<T, float>::value)
      tensor_type = GGML_TYPE_F32;
    else if (std::is_same<T, double>::value)
      tensor_type = GGML_TYPE_F64;
    else
      GGML_ASSERT(false);
    // We now allocate a context, but we will be using a backend for computations. 
    // In this case, the only purpose of the *static* context is to create the tensor metadata.
    _ctx_static = ggml_init(params);
    _a = ggml_new_tensor_1d(_ctx_static, tensor_type, 1);
    _b = ggml_new_tensor_1d(_ctx_static, tensor_type, 1);
    _x = ggml_new_tensor_1d(_ctx_static, tensor_type, 1);
    // Since we are going to be using this for both training and inference, we need to specify the 
    // model inputs, outputs, and parameters. In y = a * x + b, a and b are parameters, x is an input.
    ggml_set_input(_x);
    ggml_set_param(_a);
    ggml_set_param(_b);
    // Now we initialize the backend. In this case, we use CUDA as the backend.
    // The backend buffer is returned after we allocate the tensors using the static context + backend.
    // We will need to keep the backend buffer to free it later.
    _backend = ggml_backend_cuda_init(0);
    _backend_buffer = ggml_backend_alloc_ctx_tensors(_ctx_static, _backend);
    // Now, we create the *compute* context. This is what does inference and training. 
    // Again, .no_alloc = true because we will explicitly allocate the graph. Calculating the memory needed is easy
    // and we don't have to allocate a sufficiently large amount or check with ggml_mem.
    // By default, GGML allocates 2048 nodes (GGML_DEFAULT_GRAPH_SIZE) when allocating a graph.
    // Each of the 2048 nodes carries overhead since they are essentially tensors. Since we are 
    // doing inference and training, we need to allocate 1 graph for the forward pass, 1 for the backward pass.
    params = {
      ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + 2 * ggml_graph_overhead(),
      nullptr,
      true
    };
    // This time, we use the *compute* context in order to construct the computational graph.
    // Note that after we get `_result`, we use ggml_set_output to mark _result as the output tensor.
    _ctx_compute = ggml_init(params);
    struct ggml_tensor *ax = ggml_mul(_ctx_compute, _a, _x);
    _result = ggml_add(_ctx_compute, ax, _b);
    ggml_set_output(_result);
    // To do training, we need a backend scheduler. The backend scheduler allows us to manage several backends 
    // at once for inference and training. In this case, we really only need it to fit the model. We push the CPU
    // backend as well since it is required as a fallback.
    std::vector<ggml_backend_t> backends;
    backends.push_back(_backend);
    backends.push_back(ggml_backend_cpu_init());
    _backend_sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);
    std::cout << "Using " << ggml_backend_name(_backend) << " as backend\n";
    // After constructing the computational graph, we need to allocate the graph.
    // ggml_gallocr_new needs to know the backend buffer type. In this case, we 
    // find the backend buffer type using ggml_backend_get_default_buffer_type.
    _gf = ggml_new_graph(_ctx_compute);
    ggml_build_forward_expand(_gf, _result);
    _allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(_backend));
    ggml_gallocr_alloc_graph(_allocr, _gf); 
  }
  void set_params(const T a, const T b);
  T forward(const T x);
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


