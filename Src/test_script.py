# coding=utf-8
import sys, os, math, time, copy
import numpy as np
from test_profiling import ETOpt
from JDIMO import JDIMO

algorithm = "subgraph"

model_name = "vgg16"
ETOpt(model_name, algorithm)

# model_name = "vgg19"
# ETOpt(model_name, algorithm)

# model_name = "mobilenetv2-7"
# ETOpt(model_name, algorithm)

# model_name = "retinanet-9"
# ETOpt(model_name, algorithm)

# model_name = "yolov4"
# ETOpt(model_name, algorithm)

# vgg16 相关路径
model_folder_dir = os.path.join("home", "user_name", "onnx_model_zoo", "vgg16")
model_file_name = "vgg16.onnx"
model_file_dir = os.path.join(model_folder_dir, model_file_name)
BatchSize = 1
listUnknownDims = []
listInputName = ["data"]
listInputShape = [[1,3,224,224]]
listRange = [[0, 255]]
useless_prefix = "vgg0_"

dictInputTensor = {}
dictInputTensor["data"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
listInputTensor = [dictInputTensor["data"]]
listInputShape = [[1,3,224,224]]
listRange = [[0, 255]]
useless_prefix = "vgg0_"

JDIMO(model_file_dir, BatchSize, listUnknownDims, listRange, useless_prefix)


