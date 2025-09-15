#!/bin/bash
# ./startApp.sh /home/modelscope/ai/model/Qwen2.5-VL-7B-Instruct-lora Qwen2.5-VL-7B-Instruct-lora 8001
#export HSA_OVERRIDE_GFX_VERSION=11.0.0
#启用 ROCm 的 HIPBLASLT计算引擎
#export ROCBLAS_USE_HIPBLASLT=1
#--modelPath 模型路径
#--modelName 模型名称
#--port 端口
#--dtype 精度 bfloat16
modelPath1="/home/modelscope/ai/model/Qwen2.5-VL-7B-Instruct-lora"
modelPath2="/home/modelscope/ai/model/Qwen2.5-VL-7B-Instruct"
modelName1="Qwen2.5-VL-7B-Instruct-lora"
modelName2="Qwen2.5-VL-7B-Instruct"
port1=8001
port2=8002
dtype1="bfloat16"
dtype2="bfloat16"
# 在第一个终端中运行模型服务
echo "启动模型服务终端..."
gnome-terminal --title="模型服务 - $modelName1" -- bash -c "echo '正在启动模型服务...'; export HSA_OVERRIDE_GFX_VERSION=11.0.0; export ROCBLAS_USE_HIPBLASLT=1; python runModel.py --modelPath \"$modelPath1\" --modelName \"$modelName1\" --port \"$port1\" --dtype \"$dtype1\"; exec bash"
sleep 3
gnome-terminal --title="模型服务 - $modelName2" -- bash -c "echo '正在启动模型服务...'; export HSA_OVERRIDE_GFX_VERSION=11.0.0; export ROCBLAS_USE_HIPBLASLT=1; python runModel.py --modelPath \"$modelPath2\" --modelName \"$modelName2\" --port \"$port2\" --dtype \"$dtype2\"; exec bash"
# 等待几秒让模型服务启动
sleep 8

# 在第二个终端中运行Web应用
echo "启动Web应用终端..."
gnome-terminal --title="Web应用" -- bash -c "echo '正在启动Web应用...'; python app.py; exec bash"

read -p "执行完毕，按回车键退出..."