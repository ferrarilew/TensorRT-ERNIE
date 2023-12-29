#ifndef TRT_HEPLER_
#define TRT_HEPLER_

#include <sys/time.h>
#include <vector>
#include <string>
#include <string.h>
#include <memory>
#include <cassert>
#include <iostream>

#include "NvInfer.h"

#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#  define CUDA_CHECK(status)                                                      \
    if (status != cudaSuccess) {                                                  \
      std::cout << "Cuda failure! Error=" << cudaGetErrorString(status)           \
                << " at " << __FILE__ << " " << __LINE__ << std::endl;            \
    }
#endif

struct TrtDestroyer {
  template <typename T>
  void operator()(T *obj) const {
    if (obj) obj->destroy();
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer>;

template <typename T>
inline TrtUniquePtr<T> MakeUnique(T *t) {
  return TrtUniquePtr<T>{t};
}

template <typename T>
inline std::shared_ptr<T> MakeShared(T *t) {
  return std::shared_ptr<T>(t, TrtDestroyer());
}

struct sample{
    std::string qid;
    std::string label;
    std::vector<int> shape_info_0;
    std::vector<int> i0;
    std::vector<int> shape_info_1;
    std::vector<int> i1;
    std::vector<int> shape_info_2;
    std::vector<int> i2;
    std::vector<int> shape_info_3;
    std::vector<float> i3;
    std::vector<int> shape_info_4;
    std::vector<int> i4;
    std::vector<int> shape_info_5;
    std::vector<int> i5;
    std::vector<int> shape_info_6;
    std::vector<int> i6;
    std::vector<int> shape_info_7;
    std::vector<int> i7;
    std::vector<int> shape_info_8;
    std::vector<int> i8;
    std::vector<int> shape_info_9;
    std::vector<int> i9;
    std::vector<int> shape_info_10;
    std::vector<int> i10;
    std::vector<int> shape_info_11;
    std::vector<int> i11;
    std::vector<float> out_data;
    uint64_t timestamp;
};

/**
 * \brief Trt TrtLogger 日志类，全局对象
 */
class TrtLogger : public nvinfer1::ILogger {
  using Severity = nvinfer1::ILogger::Severity;

 public:
  explicit TrtLogger(Severity level = Severity::kINFO);

  ~TrtLogger() = default;

  nvinfer1::ILogger& getTRTLogger();

  void log(Severity severity, const char* msg) noexcept override;

 private:
  Severity level_;
};

class TrtEngine {
 public:
  TrtEngine(std::string model_param, int dev_id);

  ~TrtEngine() {};

  int dev_id_;

  std::shared_ptr<nvinfer1::ICudaEngine> engine_;

  TrtLogger trt_logger;
};

class TrtContext {
 public:
  TrtContext(TrtEngine* engine, int profile_idx);

  int CaptureCudaGraph();

  int Forward(sample& s);

  ~TrtContext();

 private:
  int dev_id_;
  // NS_PROTO::ModelParam *_model_param_ptr;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t cuda_stream_;

  static std::vector<void*> global_device_bindings_;
  // std::vector<void*> device_bindings_;
  std::vector<void*> host_bindings_;

  char *h_buffer_;
  char *d_buffer_;

  int profile_idx_;

  int max_batch_;
  int max_seq_len_;
  int start_binding_idx_;

  int align_input_type1_bytes_;
  int align_input_type2_bytes_;
  int align_aside_input_type3_bytes_;
  int align_output_bytes_;
  int whole_bytes_;

  cudaGraph_t graph_;
  cudaGraphExec_t instance_;
  bool graph_created_ = false;
};

#endif // TRT_HEPLER_
