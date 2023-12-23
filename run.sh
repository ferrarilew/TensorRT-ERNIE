#!/bin/bash

cd `dirname $0`

bash ONNX/build.sh fp32 > /tmp/build_onnx_fp32.log 2>/tmp/build_onnx_fp32.err
bash ONNX/build.sh fp16 > /tmp/build_onnx_fp16.log 2>/tmp/build_onnx_fp16.err
bash API/builder.sh fp32 > /tmp/build_api_fp32.log 2>/tmp/build_api_fp32.err
bash API/builder.sh fp16 > /tmp/build_api_fp16.log 2>/tmp/build_api_fp16.err

cd infer_demo/
make

# onnx fp32
echo "# onnx fp32"
./ernie_infer_demo ../trt_models/ernie_onnx_fp32.engine ../data/label.test.txt label_onnx_fp32.res.txt
python3 ../local_evaluate.py label_onnx_fp32.res.txt
./ernie_infer_demo ../trt_models/ernie_onnx_fp32.engine ../data/perf.test.txt perf_onnx_fp32.res.txt
python3 ../local_evaluate.py perf_onnx_fp32.res.txt

# onnx fp16
echo "# onnx fp16"
./ernie_infer_demo ../trt_models/ernie_onnx_fp16.engine ../data/label.test.txt label_onnx_fp16.res.txt
python3 ../local_evaluate.py label_onnx_fp16.res.txt
./ernie_infer_demo ../trt_models/ernie_onnx_fp16.engine ../data/perf.test.txt perf_onnx_fp16.res.txt
python3 ../local_evaluate.py perf_onnx_fp16.res.txt

# api fp32
echo "# api fp32"
./ernie_infer_demo ../trt_models/ernie_api_fp32.engine ../data/label.test.txt label_api_fp32.res.txt
python3 ../local_evaluate.py label_api_fp32.res.txt
./ernie_infer_demo ../trt_models/ernie_api_fp32.engine ../data/perf.test.txt perf_api_fp32.res.txt
python3 ../local_evaluate.py perf_api_fp32.res.txt

# api fp16
echo "# api fp16"
./ernie_infer_demo ../trt_models/ernie_api_fp16.engine ../data/label.test.txt label_api_fp16.res.txt
python3 ../local_evaluate.py label_api_fp16.res.txt
./ernie_infer_demo ../trt_models/ernie_api_fp16.engine ../data/perf.test.txt perf_api_fp16.res.txt
python3 ../local_evaluate.py perf_api_fp16.res.txt
