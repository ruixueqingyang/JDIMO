# coding=utf-8
##############################################################################################
# 移动平台 CNN 能效-温度 优化 主文件
##############################################################################################

import os, math, time
import numpy as np
import torch
import copy
import tensorrt as trt
from trt_engine_memory import get_engine, allocate_buffers, allocate_input_buffers, allocate_output_buffers
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

    # print("tmp_measure_network_energy: 0")
    # tmp_measure_network_energy("/home/wfr/work/DLA/onnx_model_zoo/yolov4/Gather__1875_gpu.trt")

    # time_begin = time.time()
    
    if os.path.exists("D:\\cloud\\study\\Coding"):
        WorkDir = "D:\\cloud\\study\\Coding"
    elif os.path.exists("/home/wfr/Coding"):
        WorkDir = "/home/wfr/Coding"
    else:
        print("预设工作路径不存在!")
        exit(1)

    model_name = "vgg16"
    # model_name = "vgg19"
    # model_name = "mobilenetv2-7"
    # model_name = "retinanet-9"
    # model_name = "yolov4"

    # model_name = "resnet18-v2-7"
    # model_name = "resnet50-v2-7"
    # model_name = "googlenet-12"
    # model_name = "bvlcalexnet-12"
    # model_name = "densenet-12"
    # model_name = "inception-v2-9"
    # model_name = "shufflenet-v2-10"
    # model_name = "squeezenet1.1"

    BatchSize = 1
    listUnknownDims = []
    dictInputTensor = {}
    dictTensorShape = {}
    listInputTensor = []
    list_input_nparray = []

    if model_name == "densenet-12":
        # densenet-12 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "densenet-12")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "densenet-12", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "densenet-12.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data_0"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data_0"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "densenet-12", "densenet-12_gpu_dla_sliding_window.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "densenet-12", "densenet-12_gpu.trt")

    elif model_name == "shufflenet-v2-10":
        # shufflenet-v2-10 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "shufflenet-v2-10")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "shufflenet-v2-10", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "shufflenet-v2-10.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["input"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["input"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "shufflenet-v2-10", "shufflenet-v2-10_gpu_dla_sliding_window.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "shufflenet-v2-10", "shufflenet-v2-10_gpu.trt")

    elif model_name == "retinanet-9":
        # retinanet-9 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "retinanet-9.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["input"] = np.random.random([1,3,480,640]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["input"]]
        listInputShape = [[1,3,480,640]]
        listRange = [[0, 255]]

        RingLen = 2

        # trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9", "retinanet-9_gpu_dla_sliding_window.trt")
        # trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9", "retinanet-9_gpu_dla_manual.trt")
        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9", "retinanet-9_gpu_dla_subgraph.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "retinanet-9", "retinanet-9_gpu.trt")

    elif model_name == "inception-v2-9":
        # inception-v2-9 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "inception-v2-9")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "inception-v2-9", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "inception-v2-9.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data_0"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data_0"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "inception-v2-9", "inception-v2-9_gpu_dla_sliding_window.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "inception-v2-9", "inception-v2-9_gpu.trt")

    elif model_name == "squeezenet1.1":
        # squeezenet1.1 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "squeezenet1.1")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "squeezenet1.1", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "squeezenet1.1.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "squeezenet1.1", "squeezenet1.1_gpu_dla_sliding_window.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "squeezenet1.1", "squeezenet1.1_gpu.trt")

    elif model_name == "mobilenetv2-7":
        # mobilenetv2-7 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "mobilenetv2-7")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "mobilenetv2-7", "opt")
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

    elif model_name == "bvlcalexnet-12":
        # bvlcalexnet-12 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "bvlcalexnet-12")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "bvlcalexnet-12", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "bvlcalexnet-12.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data_0"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data_0"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "bvlcalexnet-12", "bvlcalexnet-12_gpu_dla_sliding_window.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "bvlcalexnet-12", "bvlcalexnet-12_gpu.trt")

    elif model_name == "googlenet-12":
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

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "googlenet-12", "googlenet-12_gpu_dla_subgraph.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "googlenet-12", "googlenet-12_gpu.trt")

    elif model_name == "resnet50-v2-7":
        # resnet50-v2-7 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet50-v2-7")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet50-v2-7", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "resnet50-v2-7.onnx")
        BatchSize = 1
        listUnknownDims = [BatchSize]
        dictInputTensor["data"] = np.random.random([BatchSize,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[BatchSize,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet50-v2-7", "resnet50-v2-7_gpu_dla_sliding_window.trt")
        # trt_engine_gpu_dla1_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet50-v2-7", "resnet50-v2-7_gpu_dla1_sliding_window.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet50-v2-7", "resnet50-v2-7_gpu.trt")

    elif model_name == "yolov3-tiny":
        # tiny-yolov3 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "tiny-yolov3")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "tiny-yolov3", "19_dla_layers")
        onnx_file_dir = os.path.join(input_folder_dir, "yolov3-tiny.onnx")
        # "D:\cloud\study\Coding\DLA\onnx_model_zoo\tiny-yolov3\yolov3-tiny.onnx"
        BatchSize = 1
        ImageSize = [128, 128]
        listUnknownDims = [BatchSize, *ImageSize, BatchSize]
        dictInputTensor["input_1"] = np.random.random([1,3,128,128]).astype(np.float32) * 255
        dictInputTensor["image_shape"] = np.array([128,128]).astype(np.float32).reshape([1,2])
        listInputTensor = [dictInputTensor["input_1"], dictInputTensor["image_shape"]]
        listInputShape = [[1,3,128,128],[1,2]]
        listRange = [[0, 255], np.array([128, 128])]

        dictTensorShape["yolonms_layer_1"] = [1, 240, 4]
        dictTensorShape["yolonms_layer_1:1"] = [1, 80, 240]
        dictTensorShape["yolonms_layer_1:2"] = [1, 32, 3]

        RingLen = 2

    elif model_name == "yolov3":
        # yolov3 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov3")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov3", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "yolov3.onnx")
        BatchSize = 1
        ImageSize = [416, 416]
        listUnknownDims = [BatchSize, *ImageSize, BatchSize]
        dictInputTensor["input_1"] = np.random.random([1,3,416,416]).astype(np.float32) * 255
        dictInputTensor["image_shape"] = np.array([416,416]).astype(np.float32).reshape([1,2])
        listInputTensor = [dictInputTensor["input_1"], dictInputTensor["image_shape"]]
        listInputShape = [[1,3,416,416], [1,2]]
        listRange = [[0, 255], np.array([416, 416])]

        dictTensorShape["yolonms_layer_1/ExpandDims_1:0"] = [1, 16000, 4]
        dictTensorShape["yolonms_layer_1/ExpandDims_3:0"] = [1, 80, 16000]
        dictTensorShape["yolonms_layer_1/concat_2:0"] = [64, 8]

        RingLen = 2

    elif model_name == "yolov4":
        # yolov4 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov4")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov4", "opt")
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

        # trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov4", "yolov4_gpu_dla_sliding_window.trt")
        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov4", "yolov4_gpu_dla_subgraph.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov4", "yolov4_gpu.trt")

    elif model_name == "resnet18-v2-7":
        # resnet18-v2-7 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet18-v2-7")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet18-v2-7", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "resnet18-v2-7.onnx")
        BatchSize = 1
        listUnknownDims = [BatchSize]
        dictInputTensor["data"] = np.random.random([BatchSize,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[BatchSize,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet18-v2-7", "resnet18-v2-7_gpu_dla_sliding_window.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet18-v2-7", "resnet18-v2-7_gpu.trt")

    elif model_name == "vgg19":
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "vgg19.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19", "vgg19_gpu_dla_subgraph.trt")
        # trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19", "vgg19_gpu_dla_sliding_window.trt")
        # trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19", "vgg19_dla.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19", "vgg19_gpu.trt")

        # trt_engine_sub0_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19", "19_dla_layers", "vgg19_sub0_dla.trt")
        # trt_engine_sub0_dla1_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19", "19_dla_layers", "vgg19_sub0_dla1.trt")
        # trt_engine_sub1_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg19", "19_dla_layers", "vgg19_sub1_gpu.trt")

    elif model_name == "vgg16":
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg16")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg16", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "vgg16.onnx")
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        RingLen = 2

        # trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg16", "vgg16_gpu_dla_sliding_window.trt")
        # trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg16", "vgg16_gpu_dla_manual.trt")
        trt_engine_gpu_dla_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg16", "vgg16_gpu_dla_subgraph.trt")
        trt_engine_gpu_file_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "vgg16", "vgg16_gpu.trt")

    else:
        exit(0)

    # # trt_engine_gpu_dla_file_dir = os.path.join("/home/wfr/Coding/DLA.bak/onnx_model_zoo/yolov4/result_15/yolov4_gpu_dla_subgraph.trt")
    # 2(GPU-DLA): tensorrt 生成的 DLA-GPU 混合使用的 trt engine
    trt_engine_gpu_dla0 = get_engine("", trt_engine_gpu_dla_file_dir, 1, trt.DeviceType.GPU, {}, 0)
    trt_engine_gpu_dla1 = get_engine("", trt_engine_gpu_dla_file_dir, 1, trt.DeviceType.GPU, {}, 1)
    list_engine = [trt_engine_gpu_dla0, trt_engine_gpu_dla1]

    NumDLAs = 2
    NumStreams = 4
    list_pipeline_stage_hybrid_engine = [[]]
    for i in range(NumStreams):
        # sub_engine = get_engine("", trt_engine_gpu_dla_file_dir, 1, trt.DeviceType.GPU, {}, i%NumDLAs)
        list_pipeline_stage_hybrid_engine[0].append(ENGINE_STREAM(list_engine[i%NumDLAs]))
    

    # 4-stream
    # list_pipeline_stage_hybrid_engine = [[ENGINE_STREAM(trt_engine_gpu_dla0), ENGINE_STREAM(trt_engine_gpu_dla1), ENGINE_STREAM(trt_engine_gpu_dla0), ENGINE_STREAM(trt_engine_gpu_dla1)]]
    # 3-stream
    # list_pipeline_stage_hybrid_engine = [[ENGINE_STREAM(trt_engine_gpu_dla0), ENGINE_STREAM(trt_engine_gpu_dla1), ENGINE_STREAM(trt_engine_gpu_dla0)]]
    # 2-stream
    # list_pipeline_stage_hybrid_engine = [[ENGINE_STREAM(trt_engine_gpu_dla0), ENGINE_STREAM(trt_engine_gpu_dla1)]]
    # 1-stream
    # list_pipeline_stage_hybrid_engine = [[ENGINE_STREAM(trt_engine_gpu_dla1)]]

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

    del engine_pipeline_hybrid, list_pipeline_stage_hybrid_engine, trt_engine_gpu_dla0, trt_engine_gpu_dla1

    exit(0)

    # RingLen = 2
    # 2GPU: 只使用 GPU
    print("load trt enigne: {}".format(trt_engine_gpu_file_dir))
    trt_engine_gpu = get_engine("", trt_engine_gpu_file_dir, 1, trt.DeviceType.GPU, {}, 0)
    
    # list_pipeline_stage_gpu_engine = [[ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu)]]
    # list_pipeline_stage_gpu_engine = [[ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu)]]
    # list_pipeline_stage_gpu_engine = [[ENGINE_STREAM(trt_engine_gpu), ENGINE_STREAM(trt_engine_gpu)]]
    # list_pipeline_stage_gpu_engine = [[ENGINE_STREAM(trt_engine_gpu)]]

    NumStreams = 3
    list_pipeline_stage_gpu_engine = [[]]
    for i in range(NumStreams):
        # sub_engine = get_engine("", trt_engine_gpu_file_dir, 1, trt.DeviceType.GPU, {}, 0)
        list_pipeline_stage_gpu_engine[0].append(ENGINE_STREAM(trt_engine_gpu))
    
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
