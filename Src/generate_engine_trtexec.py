# coding=utf-8
import os, time, pickle
from timeit import default_timer as timer # 实际上是调用了 time.perf_counter() 返回高精度时间
import onnx
import onnx.version_converter
import onnxruntime
import sys, getopt
import numpy as np
# import torch
import copy
import json, re
# import joblib
from trt_engine_memory import get_engine, allocate_buffers
import tensorrt as trt
# TRT_LOGGER_PROFILING = trt.Logger(trt.Logger.VERBOSE)
# TRT_LOGGER_PROFILING = trt.Logger(trt.Logger.WARNING)
# TRT_LOGGER_PROFILING = trt.Logger(trt.Logger.ERROR)
TRT_LOGGER_PROFILING = trt.Logger()

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
import pycuda.driver as cuda
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

from jetson_measuring import orin_nx_measuring
from profiling import measure_engine_pipeline
from runtime import ENGINE_PIPELINE, ENGINE_STREAM

if __name__ == "__main__":

    if os.path.exists("D:\\cloud\\study\\Coding"):
        WorkDir = "D:\\cloud\\study\\Coding"
    elif os.path.exists("/home/wfr/Coding"):
        WorkDir = "/home/wfr/Coding"
    else:
        print("预设工作路径不存在!")
        exit(1)
    engine_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "trtexec_engine")

    std2file = False
    if std2file == True:
        sys.stdout = open(os.path.join(engine_folder_dir, "generate_engine_trtexec.log"), "w", encoding='utf-8')
        sys.stderr = sys.stdout

    list_model_name = ["retinanet-9"]
    # list_model_name = ["vgg16", "vgg19", "mobilenetv2-7", "retinanet-9", "yolov4", "googlenet-12"]

    dictInputTensorShapes = {"vgg16": [[1,3,224,224]], "vgg19": [[1,3,224,224]], "mobilenetv2-7": [[1,3,224,224]], "retinanet-9": [[1,3,480,640]], "yolov4": [[1,416,416,3]], "googlenet-12": [[1,3,224,224]]}
    
    dictNumStreamsDLA = {"vgg16": 4, "vgg19": 4, "mobilenetv2-7": 2, "retinanet-9": 4, "yolov4": 1, "googlenet-12": 2}
    dictNumStreamsGPU = {"vgg16": 3, "vgg19": 3, "mobilenetv2-7": 2, "retinanet-9": 2, "yolov4": 3, "googlenet-12": 2}
    
    for model_name in list_model_name:

        print("#############################################################")
        print("{}:".format(model_name), flush=True)


        onnx_model_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", model_name, model_name+"_infered.onnx")

        trt_gpu_log_dir = os.path.join(engine_folder_dir, model_name+"_trtexec_gpu.log")
        trt_gpu_engine_file_dir = os.path.join(engine_folder_dir, model_name+"_trtexec_gpu.trt")

        trt_dla_log_dir = os.path.join(engine_folder_dir, model_name+"_trtexec_dla.log")
        trt_dla_engine_file_dir = os.path.join(engine_folder_dir, model_name+"_trtexec_dla.trt")
        
        if not os.path.exists(trt_gpu_engine_file_dir) or not os.path.exists(trt_gpu_log_dir):
            strCMD = "trtexec"
            strCMD += " --onnx=" + onnx_model_dir
            strCMD += " --saveEngine=" + trt_gpu_engine_file_dir
            strCMD += " --best" # --threads --useManagedMemory" # --useManagedMemory
            strCMD += " 1>" + trt_gpu_log_dir + " 2>&1"
            print(strCMD, flush=True)
            time_begin = time.time()
            os.system(strCMD)
            time_duration = time.time() - time_begin
            print("generating GPU-only engine time: {}".format(time_duration), flush=True)

        if not os.path.exists(trt_dla_engine_file_dir) or not os.path.exists(trt_dla_log_dir):
            strCMD = "trtexec"
            strCMD += " --onnx=" + onnx_model_dir
            strCMD += " --saveEngine=" + trt_dla_engine_file_dir
            strCMD += " --useDLACore=0 --allowGPUFallback"
            strCMD += " --best" # --threads --useManagedMemory" # --useManagedMemory
            strCMD += " 1>" + trt_dla_log_dir + " 2>&1"
            print(strCMD)
            time_begin = time.time()
            os.system(strCMD)
            time_duration = time.time() - time_begin
            print("generating DLA-possible engine time: {}".format(time_duration), flush=True)

        print("")
        RingLen = 2
        listInputTensor = []
        for InputTensorShape in dictInputTensorShapes[model_name]:
            tmpInputTensor = np.random.random(InputTensorShape).astype(np.float32) * 255
            listInputTensor.append(tmpInputTensor)


        ######################################################################
        list_engine_gpu_dla = []
        trt_engine_gpu_dla0 = get_engine("", trt_dla_engine_file_dir, 1, trt.DeviceType.GPU, {}, 0)
        list_engine_gpu_dla.append(trt_engine_gpu_dla0)
        NumDLAsUsed = 1
        if dictNumStreamsDLA[model_name] > 1:
            trt_engine_gpu_dla1 = get_engine("", trt_dla_engine_file_dir, 1, trt.DeviceType.GPU, {}, 1)
            list_engine_gpu_dla.append(trt_engine_gpu_dla1)
            NumDLAsUsed = 2

        list_pipeline_stage_hybrid_engine = [[]]
        for i in range(dictNumStreamsDLA[model_name]):
            list_pipeline_stage_hybrid_engine[0].append(ENGINE_STREAM(list_engine_gpu_dla[i%NumDLAsUsed]))

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
            print("\nrun DLA-possible engine:".format(model_name))
            print("time = {:.4e} s".format(OrinNXMeasuring.measurement_duration.value / NumExec))
            print("power = {:.4e} W".format(OrinNXMeasuring.power_vdd.value))
            print("energy = {:.4e} J".format(OrinNXMeasuring.energy_vdd.value / NumExec))
            print("qps = {} q/s".format(1/(OrinNXMeasuring.measurement_duration.value / NumExec)))
            print("", flush=True)

        del engine_pipeline_hybrid, list_pipeline_stage_hybrid_engine, list_engine_gpu_dla, trt_engine_gpu_dla0
        if NumDLAsUsed > 1:
            del trt_engine_gpu_dla1



        ######################################################################
        trt_engine_gpu = get_engine("", trt_gpu_engine_file_dir, 1, trt.DeviceType.GPU, {}, 0)
        list_pipeline_stage_gpu_engine = [[]]
        for i in range(dictNumStreamsGPU[model_name]):
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
            print("\nrun GPU-only engine:".format(model_name))
            print("time = {:.4e} s".format(OrinNXMeasuring.measurement_duration.value / NumExec))
            print("power = {:.4e} W".format(OrinNXMeasuring.power_vdd.value))
            print("energy = {:.4e} J".format(OrinNXMeasuring.energy_vdd.value / NumExec))
            print("qps = {} q/s".format(1/(OrinNXMeasuring.measurement_duration.value / NumExec)))
            print("", flush=True)

        del engine_pipeline_gpu, list_pipeline_stage_gpu_engine, trt_engine_gpu