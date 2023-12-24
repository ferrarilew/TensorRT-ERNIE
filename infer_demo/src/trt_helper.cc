#include "trt_helper.h"

#include <string>
#include <fstream>
#include <sstream>

#include "NvInferPlugin.h"

using namespace std;

// BEGIN_LIB_NAMESPACE {

TrtLogger::TrtLogger(nvinfer1::ILogger::Severity level) : level_(level) {}

nvinfer1::ILogger& TrtLogger::getTRTLogger() { return *this; }

// trt logger
void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity > level_) {
    return;
  }

  switch (severity) {
    case Severity::kINTERNAL_ERROR:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kERROR:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kWARNING:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kINFO:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kVERBOSE:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
  }
}


TrtHepler::TrtHepler(std::string model_param, int dev_id)
    : _dev_id(dev_id), _model_param(model_param) {
  { // read model, deserializeCudaEngine and createExecutionContext
    std::ifstream t(_model_param);  // string pth
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string contents(buffer.str());

    CUDA_CHECK(cudaSetDevice(_dev_id));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));

    TrtLogger trt_logger;
    initLibNvInferPlugins(&trt_logger.getTRTLogger(), "");
    auto runtime = MakeUnique(nvinfer1::createInferRuntime(trt_logger.getTRTLogger()));
    auto e = runtime->deserializeCudaEngine((void*)contents.c_str(),
                                            contents.size(), nullptr);
    engine_ = MakeShared(e);
    context_ = MakeShared(engine_->createExecutionContext());
    context_->setOptimizationProfile(0);

    int max_input_size = 3 * 10 * 128 * sizeof(int) +
                         1 * 10 * 128 * sizeof(float) +
                         8 * 10 * 1 * sizeof(int);
    int max_output_size = 1 * 10 * 1 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_buffer_, max_input_size + max_output_size));
    CUDA_CHECK(cudaMallocHost(&h_buffer_, max_input_size + max_output_size));
  }

}

int TrtHepler::Forward(sample& s) {
  cudaSetDevice(_dev_id);

  int first_data_size = 1;
  for (int i = 0; i < s.shape_info_0.size(); i++) {
    first_data_size *= s.shape_info_0[i];
  }
  int second_data_size = first_data_size * sizeof(float);
  first_data_size *= sizeof(int);
  int third_data_size = 1;
  for (int i = 0; i < s.shape_info_4.size(); i++) {
    third_data_size *= s.shape_info_4[i];
  }
  int fourth_data_size = third_data_size * sizeof(float);
  third_data_size *= sizeof(int);
  int whole_input_size = 3 * first_data_size +
                         1 * second_data_size +
                         8 * third_data_size;
  int whole_output_size = 1 * fourth_data_size;
  // char *h_buffer_;
  // char *d_buffer_;
  // CUDA_CHECK(cudaMalloc(&d_buffer_, whole_input_size + whole_output_size));
  // CUDA_CHECK(cudaMallocHost(&h_buffer_, whole_input_size + whole_output_size));

  void *host_bindings_[13];
  void *device_bindings_[13];

  int b_i = 0;
  auto h_buffer_ptr = h_buffer_;
  auto d_buffer_ptr = d_buffer_;
  // 0 - 2
  while (b_i < 3) {
    host_bindings_[b_i] = h_buffer_ptr;
    h_buffer_ptr += first_data_size;
    device_bindings_[b_i] = d_buffer_ptr;
    d_buffer_ptr += first_data_size;
    b_i++;
  }
  // 3
  host_bindings_[b_i] = h_buffer_ptr;
  h_buffer_ptr += second_data_size;
  device_bindings_[b_i]= d_buffer_ptr;
  d_buffer_ptr += second_data_size;
  b_i++;
  // 4 - 11
  while (b_i < 12) {
    host_bindings_[b_i] = h_buffer_ptr;
    h_buffer_ptr += third_data_size;
    device_bindings_[b_i] = d_buffer_ptr;
    d_buffer_ptr += third_data_size;
    b_i++;
  }
  // 13 - output
  host_bindings_[b_i] = h_buffer_ptr;
  // h_buffer_ptr += fourth_data_size;
  device_bindings_[b_i]= d_buffer_ptr;
  // d_buffer_ptr += fourth_data_size;
  // b_i++;

  // memcpy
  memcpy(host_bindings_[0], s.i0.data(), first_data_size);
  memcpy(host_bindings_[1], s.i1.data(), first_data_size);
  memcpy(host_bindings_[2], s.i2.data(), first_data_size);
  memcpy(host_bindings_[3], s.i3.data(), second_data_size);

  memcpy(host_bindings_[4], s.i4.data(), third_data_size);
  memcpy(host_bindings_[5], s.i5.data(), third_data_size);
  memcpy(host_bindings_[6], s.i6.data(), third_data_size);
  memcpy(host_bindings_[7], s.i7.data(), third_data_size);
  memcpy(host_bindings_[8], s.i8.data(), third_data_size);
  memcpy(host_bindings_[9], s.i9.data(), third_data_size);
  memcpy(host_bindings_[10], s.i10.data(), third_data_size);
  memcpy(host_bindings_[11], s.i11.data(), third_data_size);

  CUDA_CHECK(cudaMemcpy(d_buffer_, h_buffer_, whole_input_size,
                        cudaMemcpyHostToDevice));

  // cudaEvent_t start, stop;
  // float elapsed_time = 0.0;

  int binding_idx = 0;
  std::vector<std::vector<int>> input_dims = {s.shape_info_0, s.shape_info_1, s.shape_info_2, s.shape_info_3,
                                              s.shape_info_4, s.shape_info_5, s.shape_info_6, s.shape_info_7,
                                              s.shape_info_8, s.shape_info_9, s.shape_info_10, s.shape_info_11};
  // set device_bindings_ and setBindingDimensions
  for (size_t i = 0; i < input_dims.size(); i++) {
    std::vector<int> dims_vec = input_dims[i];
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = static_cast<int>(dims_vec.size());
    memcpy(trt_dims.d, dims_vec.data(), sizeof(int) * trt_dims.nbDims);
    context_->setBindingDimensions(binding_idx, trt_dims);
    binding_idx++;
  }

  if (!context_->allInputDimensionsSpecified()) {
    //gLogFatal << "context_->allInputDimensionsSpecified() error";
    std::cout << ("context_->allInputDimensionsSpecified() error") << std::endl;
    assert(0);
  }

  bool ret = context_->enqueueV2(device_bindings_, cuda_stream_, nullptr);
  if (!ret) {
    std::cout << ("context_->enqueueV2 failed!") << std::endl;
    return -100;
  }

  CUDA_CHECK(cudaMemcpy(s.out_data.data(), device_bindings_[12], fourth_data_size, cudaMemcpyDeviceToHost));
  cudaStreamSynchronize(cuda_stream_);
  struct timeval tv;
  gettimeofday(&tv, NULL);
  s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;

  return 0;
}

TrtHepler::~TrtHepler() {
  CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
  CUDA_CHECK(cudaFree(d_buffer_));
}

// } // BEGIN_LIB_NAMESPACE
