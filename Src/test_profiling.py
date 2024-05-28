# coding=utf-8
##############################################################################################
# 移动平台 CNN 能效-温度 优化 主文件
##############################################################################################

import sys, os, math, time
# from queue import Queue
# from multiprocessing import Queue, Process, Manager, Value, shared_memory
# from threading import Thread
# from typing import Any
# import onnx
# import onnx.version_converter
# import onnxruntime
# import sys, getopt
import numpy as np
import torch
import copy
# from enum import Enum
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
from runtime import ENGINE_PIPELINE, ENGINE_STREAM
from mapping import map_network_subgraph
from calibrator import VOID_CALIBRATOR

def ETOpt(model_name, algorithm = "subgraph"):

    time_start = time.time()

    if len(model_name) == 0 or len(algorithm) == 0:
        return

    time_begin = time.time()
    
    WorkDir = "/home/user_name/work_space"

    origin_stdout = sys.stdout
    origin_stderr = sys.stderr
    std2file = True

    # model_name = "vgg16"
    # model_name = "vgg19"
    # model_name = "mobilenetv2-7"
    # model_name = "retinanet-9"
    # model_name = "yolov4"

    BatchSize = 1
    listUnknownDims = []
    dictInputTensor = {}
    dictTensorShape = {}
    listInputTensor = []
    useless_prefix = ""

    if model_name == "retinanet-9":
        # retinanet-9 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "retinanet-9.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "retinanet-9_profiling.log"), "w", encoding='utf-8')
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
            sys.stdout = open(os.path.join(input_folder_dir, "mobilenetv2-7_profiling.log"), "w", encoding='utf-8')
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
            sys.stdout = open(os.path.join(input_folder_dir, "yolov4_profiling.log"), "w", encoding='utf-8')
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
            sys.stdout = open(os.path.join(input_folder_dir, "vgg19_profiling.log"), "w", encoding='utf-8')
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
            sys.stdout = open(os.path.join(input_folder_dir, "vgg16_profiling.log"), "w", encoding='utf-8')
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

    # 进行 profiling 获得 融合节点在 GPU/DLA 上的 运行时间/能耗
    # idle_power = 5.5 # 系统空载功率 5.6 W
    idle_power = -1.0 # 系统空载功率 -1.0 W, 表示要进行实际测量来确定空载功率
    profile_network(NetworkMap, dictInputTensor, dictTensorShape, idle_power)
    time_end = time.time()
    print("\nprofiling duration = {} s\n".format(time_end-time_begin), flush=True)

    # exit(0)


    time_begin = time.time()
    # 配置 int8 校准
    cache_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name+"_calibration.cache")
    print("cache_file_dir = {}".format(cache_file_dir), flush=True)
    NumBatches = 8
    my_calibrator = VOID_CALIBRATOR(listInputShape, listRange, BatchSize, NumBatches, cache_file_dir)
    # 生成优化的映射和 trt engine
    opt_engine_file_dir = map_network_subgraph(NetworkMap, BatchSize, my_calibrator)
    time_end = time.time()
    print("\nmapping duration = {} s".format(time_end-time_begin), flush=True)


    time_begin = time.time()
    stream_number_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name+"_NumStreams.log")
    OptNumCUDAStreams = find_optimal_stream_number(stream_number_file_dir, opt_engine_file_dir, BatchSize, listInputTensor)
    time_end = time.time()
    print("\nfinding number of streams duration = {} s".format(time_end-time_begin), flush=True)


    print("optimization duration = {} s\n".format(time_end-time_start), flush=True)

    # exit(0)
    # return

    sys.stdout = origin_stdout
    sys.stderr = origin_stderr

# 找到最优 CUDA stream 数量
def find_optimal_stream_number(stream_number_file_dir, engine_file_dir, BatchSize, listInputTensor):

    OptNumCUDAStreams = 0
    NumDLAs = 2
    RingLen = 2

    trt_engine_gpu_dla0 = get_engine("", engine_file_dir, BatchSize, trt.DeviceType.GPU, {}, 0)
    trt_engine_gpu_dla1 = get_engine("", engine_file_dir, BatchSize, trt.DeviceType.GPU, {}, 1)
    list_engine = [trt_engine_gpu_dla0, trt_engine_gpu_dla1]

    qps_prev = -1
    while True:
        with open(stream_number_file_dir, "w") as tmp_file:
            tmpStr = "OptNumCUDAStreams = {}\n".format(OptNumCUDAStreams)
            tmp_file.write(tmpStr)
        OptNumCUDAStreams += 1

        trt_engine_gpu_dla0 = get_engine("", engine_file_dir, BatchSize, trt.DeviceType.GPU, {}, 0)
        trt_engine_gpu_dla1 = get_engine("", engine_file_dir, BatchSize, trt.DeviceType.GPU, {}, 1)
        list_engine = [trt_engine_gpu_dla0, trt_engine_gpu_dla1]

        list_pipeline_stage_hybrid_engine = [[]]
        for i in range(OptNumCUDAStreams):
            # sub_engine = get_engine("", trt_engine_gpu_dla_file_dir, 1, trt.DeviceType.GPU, {}, i%NumDLAs)
            list_pipeline_stage_hybrid_engine[0].append(ENGINE_STREAM(list_engine[i%NumDLAs]))

        engine_pipeline_hybrid = ENGINE_PIPELINE(list_pipeline_stage_hybrid_engine, RingLen)
        # 初始化流水线输入
        list_input_nparray = []
        for _ in range(engine_pipeline_hybrid.ring_buf.ring_len.value):
            list_input_nparray.append(copy.deepcopy(listInputTensor))
        # 第一个维度是buf的idx, 第二个维度是输入的idx
        engine_pipeline_hybrid.fillInputBuf(list_input_nparray)


        required_measurement_time = 4 # 测量 4s 左右
        with orin_nx_measuring(0.05) as OrinNXMeasuring:

            # 测 trt engine pipeline 的 能耗-性能
            NumExec = measure_engine_pipeline(OrinNXMeasuring, engine_pipeline_hybrid, required_measurement_time)
            qps = 1/(OrinNXMeasuring.measurement_duration.value / NumExec)
            print("\nNumCUDAStreams = {}".format(OptNumCUDAStreams))
            print("time = {:.4e} s".format(OrinNXMeasuring.measurement_duration.value / NumExec))
            print("power = {:.4e} W".format(OrinNXMeasuring.power_vdd.value))
            print("energy = {:.4e} J".format(OrinNXMeasuring.energy_vdd.value / NumExec))
            print("qps = {:.3f} q/s".format(qps))
            print("")

        del engine_pipeline_hybrid, list_pipeline_stage_hybrid_engine, list_engine, trt_engine_gpu_dla0, trt_engine_gpu_dla1

        if qps_prev > qps:
            OptNumCUDAStreams -= 1
            print("\nOptNumCUDAStreams = {}\n".format(OptNumCUDAStreams))
            break
        else:
            qps_prev = qps

    with open(stream_number_file_dir, "w") as tmp_file:
        tmpStr = "OptNumCUDAStreams = {}\n".format(OptNumCUDAStreams)
        tmp_file.write(tmpStr)

    return OptNumCUDAStreams

