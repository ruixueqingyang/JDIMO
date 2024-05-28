##############################################################################################
# 移动平台 CNN 能效-温度 优化 主文件
##############################################################################################

import os, math
from time import time
# from queue import Queue
from multiprocessing import Queue, Process, Manager, Value, shared_memory
# from threading import Thread
from typing import Any
import onnx
import onnx.version_converter
import onnxruntime
import sys, getopt
import numpy as np
import torch
import copy
import tensorrt as trt
from trt_engine_memory import get_engine, allocate_buffers, allocate_input_buffers, allocate_output_buffers
import network_type
from tmp_engine import tmp_measure_network_energy

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
import pycuda.driver as cuda
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

from jetson_measuring import orin_nx_measuring
from profiling import profile_network, measure_network_energy, measure_engine_pipeline
from runtime import ENGINE_PIPELINE
from mapping import map_network_manual1, map_network_manual, map_network_subgraph
from calibrator import VOID_CALIBRATOR

if __name__ == "__main__":

    WorkDir = "/home/user_name/work_space"

    # model_name = "vgg16"
    # model_name = "vgg19"
    # model_name = "mobilenetv2-7"
    model_name = "retinanet-9"
    # model_name = "yolov4"

    BatchSize = 1
    listUnknownDims = []
    dictInputTensor = {}
    dictTensorShape = {}
    listInputTensor = []
    useless_prefix = ""
    origin_stdout = sys.stdout
    origin_stderr = sys.stderr
    std2file = False

    if model_name == "retinanet-9":
        # retinanet-9 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "retinanet-9.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "retinanet-9_profiling_manual.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["input"] = np.random.random([1,3,480,640]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["input"]]
        listInputShape = [[1,3,480,640]]
        listRange = [[0, 255]]

        useless_prefix = ""

    elif model_name == "mobilenetv2-7":
        # mobilenetv2-7 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "mobilenetv2-7")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "mobilenetv2-7", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "mobilenetv2-7.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "mobilenetv2-7_profiling_manual.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = [BatchSize]
        dictInputTensor["input"] = np.random.random([BatchSize,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["input"]]
        listInputShape = [[BatchSize,3,224,224]]
        listRange = [[0, 255]]

        useless_prefix = ""

    elif model_name == "yolov4":
        # yolov4 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov4")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov4", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "yolov4.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "yolov4_profiling_manual.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = [BatchSize]
        dictInputTensor["input_1:0"] = np.random.random([1,416,416,3]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["input_1:0"]]
        listInputShape = [[1,416,416,3]]
        listRange = [[0, 255]]

        dictTensorShape["Identity:0"] = [1, 52, 52, 3, 85]
        dictTensorShape["Identity_1:0"] = [1, 26, 26, 3, 85]
        dictTensorShape["Identity_2:0"] = [1, 13, 13, 3, 85]
        useless_prefix = "StatefulPartitionedCall/model/"

    elif model_name == "vgg19":
        # vgg19 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "vgg19.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "vgg19_profiling_manual.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]
        useless_prefix = "vgg0_"

    elif model_name == "vgg16":
        # vgg16 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg16")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg16", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "vgg16.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "vgg16_profiling_manual.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]
        useless_prefix = "vgg0_"

    else:
        exit(0)

    # 输入 onnx 文件路径
    # 加载模型, 提取并结构化模型信息, 构造网络拓扑图
    NetworkMap = network_type.network_map(onnx_file_dir, listUnknownDims, useless_prefix)

    # # 进行 profiling 获得 融合节点在 GPU/DLA 上的 运行时间/能耗
    # idle_power = 6.0 # 系统空载功率 5.6 W
    # # idle_power = -1.0 # 系统空载功率 -1.0 W, 表示要进行实际测量来确定空载功率
    # profile_network(NetworkMap, dictInputTensor, dictTensorShape, idle_power)

    # 配置 int8 校准
    cache_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name+"_calibration.cache")
    print("cache_file_dir = {}".format(cache_file_dir), flush=True)
    NumBatches = 8
    my_calibrator = VOID_CALIBRATOR(listInputShape, listRange, BatchSize, NumBatches, cache_file_dir)

    engine_pipeline = map_network_manual1(NetworkMap, BatchSize, my_calibrator)