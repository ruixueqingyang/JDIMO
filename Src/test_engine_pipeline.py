# coding=utf-8
##############################################################################################
# 移动平台 CNN 能效-温度 优化 主文件
##############################################################################################

import os, math, time, json
import numpy as np
import torch
import copy
import onnx
import tensorrt as trt
from trt_engine_memory import get_engine, allocate_buffers, allocate_input_buffers, allocate_output_buffers
from mapping import ENGINE_CONFIG, loadMappingConfig
# from tmp_engine import tmp_measure_network_energy

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

if __name__=="__main__":
    
    if os.path.exists("D:\\cloud\\study\\Coding"):
        WorkDir = "D:\\cloud\\study\\Coding"
    elif os.path.exists("/home/wfr/Coding"):
        WorkDir = "/home/wfr/Coding"
    else:
        print("预设工作路径不存在!")
        exit(1)

    # model_name = "vgg16"
    # model_name = "vgg19"
    # model_name = "mobilenetv2-7"
    model_name = "retinanet-9"
    # model_name = "yolov4"
    # model_name = "googlenet-12"

    BatchSize = 1
    listUnknownDims = []
    dictInputTensor = {}
    dictTensorShape = {}
    listInputTensor = []
    list_input_nparray = []

    if model_name == "googlenet-12":
        # googlenet-12 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "googlenet-12")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "googlenet-12", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "googlenet-12.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data_0"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data_0"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "googlenet-12", "googlenet-12_gpu.trt")

    elif model_name == "retinanet-9":
        # retinanet-9 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9")
        onnx_file_dir = os.path.join(input_folder_dir, "retinanet-9.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["input"] = np.random.random([1,3,480,640]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["input"]]
        listInputShape = [[1,3,480,640]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9", "retinanet-9_gpu.trt")

    elif model_name == "mobilenetv2-7":
        # mobilenetv2-7 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "mobilenetv2-7")
        onnx_file_dir = os.path.join(input_folder_dir, "mobilenetv2-7.onnx")
        BatchSize = 1
        listUnknownDims = [BatchSize]
        dictInputTensor["input"] = np.random.random([BatchSize,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["input"]]
        listInputShape = [[BatchSize,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "mobilenetv2-7", "mobilenetv2-7_gpu_dla_subgraph.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "mobilenetv2-7", "mobilenetv2-7_gpu.trt")

    elif model_name == "yolov4":
        # yolov4 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov4")
        onnx_file_dir = os.path.join(input_folder_dir, "yolov4.onnx")
        BatchSize = 1
        listUnknownDims = [BatchSize]
        dictInputTensor["input_1:0"] = np.random.random([1,416,416,3]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["input_1:0"]]
        listInputShape = [[1,416,416,3]]
        listRange = [[0, 255]]

        dictTensorShape["Identity:0"] = [1, 52, 52, 3, 85]
        dictTensorShape["Identity_1:0"] = [1, 26, 26, 3, 85]
        dictTensorShape["Identity_2:0"] = [1, 13, 13, 3, 85]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov4", "yolov4_gpu_dla_subgraph.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov4", "yolov4_gpu.trt")

    elif model_name == "vgg19":
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19")
        onnx_file_dir = os.path.join(input_folder_dir, "vgg19.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19", "vgg19_gpu_dla_subgraph.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19", "vgg19_gpu.trt")

    elif model_name == "vgg16":
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg16")
        onnx_file_dir = os.path.join(input_folder_dir, "vgg16.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg16", "vgg16_gpu_dla_subgraph.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg16", "vgg16_gpu.trt")

    else:
        exit(0)


    #########################################################################
    # 加载 engine pipeline ###################################################
    # vgg19_[0-88]_dla.trt
    # vgg19_[89-104]_gpu.trt

    # engine_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", model_name, "JEDI_Origin")
    # engine_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", model_name, "JEDI_onnx")
    # engine_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", model_name, "manual_opt")
    # engine_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", model_name, "manual_0")
    engine_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", model_name, "manual_JEDI")
    mapping_file_dir = os.path.join(engine_folder_dir, model_name + "_mapping_manual.json")

    # engine_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", model_name)
    # mapping_file_dir = os.path.join(engine_folder_dir, model_name + "_mapping_subgraph.json")

    print("test_engine_pipeline: mapping_file_dir: {}".format(mapping_file_dir))
    
    listEngineConfig = loadMappingConfig(mapping_file_dir)

    isUseDualDLA = True
    if isUseDualDLA == True:
        NumDLAs = 2
    else:
        NumDLAs = 1
    default_device_type = trt.DeviceType.GPU
    list_pipeline_stage_hybrid_engine = []
    for EngineConfig in listEngineConfig:
        device_name = EngineConfig.device
        listLayerID = EngineConfig.listLayerID
        NumStreams = EngineConfig.NumStreams

        isConsecutive = False
        strHyphen = "..."
        if len(EngineConfig.listLayerID) - 1 == EngineConfig.listLayerID[-1] - EngineConfig.listLayerID[0]:
            isConsecutive = True
            strHyphen = "-"

        strSuffix = "_gpu"
        if device_name[:3] == "DLA":
            strSuffix = "_dla"

        sub_engine_file_dir = os.path.join(engine_folder_dir, "{}_[{}{}{}]{}.trt".format(model_name, listLayerID[0], strHyphen, listLayerID[-1], strSuffix))

        print("test_engine_pipeline: sub_engine_file_dir: {}".format(sub_engine_file_dir))

        pipeline_stage_hybrid_engine = []
        for i in range(NumStreams):
            if device_name == "DLA0":
                sub_engine = get_engine("", sub_engine_file_dir, 1, default_device_type, {}, 0)
                pipeline_stage_hybrid_engine.append(ENGINE_STREAM(sub_engine))
            elif device_name == "DLA1":
                sub_engine = get_engine("", sub_engine_file_dir, 1, default_device_type, {}, 1)
                pipeline_stage_hybrid_engine.append(ENGINE_STREAM(sub_engine))
            else:
                sub_engine = get_engine("", sub_engine_file_dir, 1, default_device_type, {}, i%NumDLAs)
                pipeline_stage_hybrid_engine.append(ENGINE_STREAM(sub_engine))

        list_pipeline_stage_hybrid_engine.append(pipeline_stage_hybrid_engine)

    print("list_pipeline_stage_hybrid_engine:")
    for pipeline_stage_hybrid_engine in list_pipeline_stage_hybrid_engine:
        print("pipeline_stage_hybrid_engine: ", end="")
        for engine_stream in pipeline_stage_hybrid_engine:
            print("{}, ".format(engine_stream.trt_engine.num_layers), end="")
        print("")
    print("")

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
        print("\n{} engine_pipeline_hybrid:".format(model_name))
        print("time = {:.4e} s".format(OrinNXMeasuring.measurement_duration.value / NumExec))
        print("power = {:.4e} W".format(OrinNXMeasuring.power_vdd.value))
        print("energy = {:.4e} J".format(OrinNXMeasuring.energy_vdd.value / NumExec))
        print("qps = {:.3f} q/s".format(1/(OrinNXMeasuring.measurement_duration.value / NumExec)))
        print("")

    del engine_pipeline_hybrid, list_pipeline_stage_hybrid_engine

    exit(0)



    #########################################################################
    # 2GPU: 只使用 GPU #######################################################
    print("load trt enigne: {}".format(trt_engine_gpu_file_dir))
    trt_engine_gpu = get_engine("", trt_engine_gpu_file_dir, 1, trt.DeviceType.GPU, {}, 0)
    
    # list_pipeline_stage_gpu_engine = [[ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu)]]
    # list_pipeline_stage_gpu_engine = [[ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu)]]
    list_pipeline_stage_gpu_engine = [[ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu)]]
    # list_pipeline_stage_gpu_engine = [[ENGINE_STREAM(trt_engine_gpu)]]
    
    engine_pipeline_gpu = ENGINE_PIPELINE(list_pipeline_stage_gpu_engine, RingLen)
    # 初始化流水线输入
    list_input_nparray = []
    for _ in range(engine_pipeline_gpu.ring_buf.ring_len.value):
        list_input_nparray.append(copy.deepcopy(listInputTensor))
    # 第一个维度是buf的idx, 第二个维度是输入的idx
    engine_pipeline_gpu.fillInputBuf(list_input_nparray)


    required_measurement_time = 4 # 测量 4s 左右
    with orin_nx_measuring(0.05) as OrinNXMeasuring:

        # 测 trt engine pipeline 的 能耗-性能
        NumExec = measure_engine_pipeline(OrinNXMeasuring, engine_pipeline_gpu, required_measurement_time)
        print("\n{} engine_pipeline_gpu:".format(model_name))
        print("time = {:.4e} s".format(OrinNXMeasuring.measurement_duration.value / NumExec))
        print("power = {:.4e} W".format(OrinNXMeasuring.power_vdd.value))
        print("energy = {:.4e} J".format(OrinNXMeasuring.energy_vdd.value / NumExec))
        print("qps = {:.3f} q/s".format(1/(OrinNXMeasuring.measurement_duration.value / NumExec)))
        print("")

    del engine_pipeline_gpu, list_pipeline_stage_gpu_engine, trt_engine_gpu
