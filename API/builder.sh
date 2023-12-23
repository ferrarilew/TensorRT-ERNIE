#!/bin/bash

cd `dirname $0`
model_type=$1

mkdir -p ../trt_models

if [[ "$model_type" == "fp32" ]]; then
  #python3 API/builder.py -p ../sti2_data/model/paddle_infer_model -o trt_models/ernie_api_fp32.engine
  python3 builder.py -p ../../sti2_data/model/paddle_infer_model -o ../trt_models/ernie_api_fp32.engine
  #python3 API/builder_base.py -p ../sti2_data/model/paddle_infer_model -o trt_models/ernie_api_fp32.engine
elif [[ "$model_type" == "fp16" ]]; then
  # python3 API/builder.py --fp16 -p ../sti2_data/model/paddle_infer_model -o trt_models/ernie_api_fp16.engine
  python3 builder.py -f -p ../../sti2_data/model/paddle_infer_model -o ../trt_models/ernie_api_fp16.engine
# elif [[ "$model_type" == "int8" ]]; then
  # int8 build
  # python builder.py --strict --int8 -p ../model/paddle_infer_model -o ../model/trt_model/ernie_int8.engine -c /home/ubuntu/baidu_sti/model/calib_data/
else
  echo "no this model type, only support fp32 and fp16 now!"
fi
