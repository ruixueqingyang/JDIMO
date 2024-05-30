# coding=utf-8
import sys, os
sys.path.append("../Src")
from JDIMO import JDIMO

# yolov4
model_folder_dir = os.path.join("home", "user_name", "onnx_model_zoo", "yolov4")
model_file_name = "yolov4.onnx"
model_file_dir = os.path.join(model_folder_dir, model_file_name)
BatchSize = 1

# Unknown dimensions of different input variables
dictUnknownShape = {}
dictUnknownShape["input_1:0"] = [BatchSize,416,416,3]
dictUnknownShape["Identity:0"] = [1, 52, 52, 3, 85]
dictUnknownShape["Identity_1:0"] = [1, 26, 26, 3, 85]
dictUnknownShape["Identity_2:0"] = [1, 13, 13, 3, 85]

# The range of values for input variables
dictRange = {}
dictRange["input_1:0"] = [0, 255]

useless_prefix = "StatefulPartitionedCall/model/"

JDIMO(model_file_dir, BatchSize, dictUnknownShape, dictRange, useless_prefix)


