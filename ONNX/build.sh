#!/bin/bash

cd `dirname $0`
model_type=$1

mkdir -p ../trt_models

if [[ "$model_type" == "fp32" ]]; then
  python3 onnx2trt.py -m=../../sti2_data/model/onnx_infer_model/model.onnx -o=../trt_models/ernie_onnx_fp32.engine
elif [[ "$model_type" == "fp16" ]]; then
  python3 onnx2trt.py -f -m=../../sti2_data/model/onnx_infer_model/model.onnx -o=../trt_models/ernie_onnx_fp16.engine
else
  echo "no this model type, only support fp32 and fp16 now!"
fi
