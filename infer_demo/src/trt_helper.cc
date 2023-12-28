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

TrtEngine::TrtEngine(std::string model_param, int dev_id)
    : dev_id_(dev_id) {
  // read model, deserializeCudaEngine and createExecutionContext
  std::ifstream t(model_param);  // string pth
  std::stringstream buffer;
  buffer << t.rdbuf();
  std::string contents(buffer.str());

  CUDA_CHECK(cudaSetDevice(dev_id_));

  initLibNvInferPlugins(&trt_logger.getTRTLogger(), "");
  auto runtime = MakeUnique(nvinfer1::createInferRuntime(trt_logger.getTRTLogger()));
  auto e = runtime->deserializeCudaEngine((void*)contents.c_str(),
                                          contents.size(), nullptr);
  engine_ = MakeShared(e);

  std::cout << "getNbIOTensors:" << engine_->getNbIOTensors() << std::endl;
}

constexpr size_t kAlignment = 128;
constexpr int AlignTo(int a, int b = kAlignment) {
  return (a + b - 1) / b * b;
}

std::vector<void*> TrtContext::global_device_bindings_;

TrtContext::TrtContext(TrtEngine *trt_engine, int profile_idx) {
  profile_idx_ = profile_idx;
  engine_ = trt_engine->engine_;
  dev_id_ = trt_engine->dev_id_;
  CUDA_CHECK(cudaSetDevice(dev_id_));
  CUDA_CHECK(cudaStreamCreate(&cuda_stream_));

  context_ = MakeShared(engine_->createExecutionContext());
  context_->setOptimizationProfile(profile_idx_);

  start_binding_idx_ = profile_idx * engine_->getNbBindings() /
                       engine_->getNbOptimizationProfiles();
  auto max_profile = engine_->getProfileDimensions(
      start_binding_idx_, profile_idx, nvinfer1::OptProfileSelector::kMAX);

  max_batch_ = max_profile.d[0];
  max_seq_len_ = max_profile.d[1];

  // 4 inputs:[B，S]
  align_input_type1_bytes_ = AlignTo(max_batch_ * max_seq_len_ * sizeof(int));
  align_input_type2_bytes_ = AlignTo(max_batch_ * max_seq_len_ * sizeof(float));
  // 8 aside inputs and 1 output:[B]
  align_aside_input_type3_bytes_ = AlignTo(max_batch_ * sizeof(int));
  align_output_bytes_ = AlignTo(max_batch_ * sizeof(float));

  whole_bytes_ = align_input_type1_bytes_ * 3 + align_input_type2_bytes_ * 1 +
                 align_aside_input_type3_bytes_ * 8 + align_output_bytes_ * 1;

  CUDA_CHECK(cudaMalloc(&d_buffer_, whole_bytes_));
  CUDA_CHECK(cudaMallocHost(&h_buffer_, whole_bytes_));

  auto h_buffer_ptr = h_buffer_;
  auto d_buffer_ptr = d_buffer_;

  if (global_device_bindings_.empty()) {
    global_device_bindings_.resize(engine_->getNbBindings());
  }
  // for (size_t i= 0; i < global_device_bindings_.size(); i++) {
  //   global_device_bindings_[i] = d_buffer_ptr;
  // }

  int b_i = 0;
  // 0 - 2
  while (b_i < 3) {
    global_device_bindings_[start_binding_idx_ + b_i]= d_buffer_ptr;
    host_bindings_.push_back(h_buffer_ptr);
    h_buffer_ptr += align_input_type1_bytes_;
    d_buffer_ptr += align_input_type1_bytes_;
    b_i++;
  }
  // 3
  global_device_bindings_[start_binding_idx_ + b_i]= d_buffer_ptr;
  host_bindings_.push_back(h_buffer_ptr);
  h_buffer_ptr += align_input_type2_bytes_;
  d_buffer_ptr += align_input_type2_bytes_;
  b_i++;
  // 4 - 11
  while (b_i < 12) {
    global_device_bindings_[start_binding_idx_ + b_i] = d_buffer_ptr;
    host_bindings_.push_back(h_buffer_ptr);
    h_buffer_ptr += align_aside_input_type3_bytes_;
    d_buffer_ptr += align_aside_input_type3_bytes_;
    b_i++;
  }
  // 12
  global_device_bindings_[start_binding_idx_ + b_i] = d_buffer_ptr;
  host_bindings_.push_back(h_buffer_ptr);

  std::vector<int> input_dim = {max_batch_, max_seq_len_, 1};
  std::vector<int> aside_input_dim = {max_batch_, 1, 1};

  int binding_idx = start_binding_idx_;
  std::vector<std::vector<int>> input_dims = {
    input_dim, input_dim, input_dim, input_dim,
    aside_input_dim, aside_input_dim, aside_input_dim, aside_input_dim,
    aside_input_dim, aside_input_dim, aside_input_dim, aside_input_dim
    };

  // set device bindings_ and setBindingDimensions
  for (size_t i = 0; i < input_dims.size(); i++) {
    std::vector<int> dims_vec = input_dims[i];
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = static_cast<int>(dims_vec.size());
    memcpy(trt_dims.d, dims_vec.data(), sizeof(int) * trt_dims.nbDims);
    context_->setBindingDimensions(binding_idx, trt_dims);
    binding_idx++;
  }

  if (!context_->allInputDimensionsSpecified()) {
    std::cout << ("context->allInputDimensionsSpecified() error") << std::endl;
    assert(0);
  }

  // warmup
  CUDA_CHECK(cudaMemcpyAsync(d_buffer_, h_buffer_, whole_bytes_,
                             cudaMemcpyHostToDevice, cuda_stream_));
  cudaStreamSynchronize(cuda_stream_);
}

int TrtContext::Forward(sample& s) {
  cudaSetDevice(dev_id_);

  auto batch = s.shape_info_0[0];
  auto seq_len = s.shape_info_0[1];
  auto input_type1_bytes = batch * seq_len * sizeof(int);
  auto input_type2_bytes = batch * seq_len * sizeof(float);
  auto aside_input_type3_bytes = batch * sizeof(int);
  auto aside_output_bytes = batch * sizeof(float);

  // memcpy
  memcpy(host_bindings_[0], s.i0.data(), input_type1_bytes);
  memcpy(host_bindings_[1], s.i1.data(), input_type1_bytes);
  memcpy(host_bindings_[2], s.i2.data(), input_type1_bytes);
  memcpy(host_bindings_[3], s.i3.data(), input_type2_bytes);

  memcpy(host_bindings_[4], s.i4.data(), aside_input_type3_bytes);
  memcpy(host_bindings_[5], s.i5.data(), aside_input_type3_bytes);
  memcpy(host_bindings_[6], s.i6.data(), aside_input_type3_bytes);
  memcpy(host_bindings_[7], s.i7.data(), aside_input_type3_bytes);
  memcpy(host_bindings_[8], s.i8.data(), aside_input_type3_bytes);
  memcpy(host_bindings_[9], s.i9.data(), aside_input_type3_bytes);
  memcpy(host_bindings_[10], s.i10.data(), aside_input_type3_bytes);
  memcpy(host_bindings_[11], s.i11.data(), aside_input_type3_bytes);

  CUDA_CHECK(cudaMemcpyAsync(d_buffer_, h_buffer_, whole_bytes_ - align_output_bytes_,
                        cudaMemcpyHostToDevice, cuda_stream_));
  cudaStreamSynchronize(cuda_stream_);

  bool ret = context_->enqueueV2((void**)global_device_bindings_.data(), cuda_stream_, nullptr);
  if (!ret) {
    std::cout << "context_->enqueueV2 failed!" << std::endl;
    return -100;
  }

  // s.out_data.resize(batch);
  CUDA_CHECK(cudaMemcpyAsync(s.out_data.data(), global_device_bindings_[start_binding_idx_ + 12], aside_output_bytes,
                             cudaMemcpyDeviceToHost, cuda_stream_));
  cudaStreamSynchronize(cuda_stream_);

  struct timeval tv;
  gettimeofday(&tv, NULL);
  s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;

  return 0;
}

TrtContext::~TrtContext() {
  CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
  CUDA_CHECK(cudaFree(d_buffer_));
  CUDA_CHECK(cudaFreeHost(h_buffer_));
}

// } // BEGIN_LIB_NAMESPACE
