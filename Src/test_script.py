# coding=utf-8
import sys, os, math, time, copy
import numpy as np
from test_profiling import ETOpt

algorithm = "subgraph"

# # 全map到dla就很好, 目前算法不能处理
# model_name = "densenet-12"
# ETOpt(model_name, algorithm)

# 再测测看
model_name = "googlenet-12"
ETOpt(model_name, algorithm)

# model_name = "mobilenetv2-7"
# ETOpt(model_name, algorithm)

# model_name = "vgg16"
# ETOpt(model_name, algorithm)

# model_name = "vgg19"
# ETOpt(model_name, algorithm)

# model_name = "yolov4"
# ETOpt(model_name, algorithm)

# model_name = "retinanet-9"
# ETOpt(model_name, algorithm)

