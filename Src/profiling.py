# coding=utf-8
##############################################################################################
# 测量融合节点在 GPU/DLA 上的 性能/能耗
# 实验结果: 
##############################################################################################

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
import network_type
import extract_layer


def profile_network(NetworkMap, dictInputTensor = {}, dictTensorShape = {}, idle_power = -1.0):

    if idle_power <= 0.0:
        idle_power = measure_idle_power()
    NetworkMap.idle_power = idle_power
    print("\nprofile_network: idle_power = {:.4e} W\n".format(idle_power), flush=True)
    
    trt_gpu_engine_verbose_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name+"_verbose_gpu.trt")
    trt_gpu_layer_information_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name+"_layer_information_gpu.json")
    trt_gpu_log_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_trtexec_verbose_gpu.log")
    trt_gpu_engine_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name+"_gpu.trt")
    trt_gpu_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_trtexec_perf_gpu.log")

    trt_dla_engine_verbose_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name+"_verbose_dla.trt")
    trt_dla_layer_information_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name+"_layer_information_dla.json")
    trt_dla_log_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_trtexec_verbose_dla.log")
    trt_dla_engine_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name+"_dla.trt")
    trt_dla_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_trtexec_perf_dla.log")

    # 命令行调用 trtexec, 设置 GPU 为默认设备, 生成 trt 模型, 输出 layer_information.json
    # 只使用GPU: trtexec --onnx=./resnet18-v2-7.onnx --saveEngine=resnet_engine_cli.trt --shapes=data:4x3x224x224 --useManagedMemory --threads --exportProfile=./layer_information.json
    # 尽可能使用DLA: trtexec --onnx=./resnet18-v2-7.onnx --saveEngine=resnet_engine_cli.trt --shapes=data:4x3x224x224 --useDLACore=0 --allowGPUFallback --useManagedMemory --threads --exportProfile=./layer_information.json
    # 设置输入数据 shape
    strShape = ""
    # strShape = " --shapes=" # "input0:1x3x256x256,input1:1x3x128x128"
    # for input in NetworkMap.listModelInput:
    #     strShape += input.name + ":"
    #     for dim in input.type.tensor_type.shape.dim:
    #         # if len(dim.dim_param) != 0 or dim.dim_value == 0:
    #         #     strShape += str(BatchSize) + "x"
    #         # else:
    #         strShape += str(dim.dim_value) + "x"
    #     strShape = strShape[:-1] + ","
    # strShape = strShape[:-1]

    # 先将 onnx 模型 整体全部映射到 GPU / DLA0 上 获取相关测量数据
    # trtexec 生成 GPU 上 单stream engine 及 节点/层之间融合 配置情况
    if not os.path.exists(trt_gpu_engine_file_dir) or not os.path.exists(trt_gpu_layer_information_file_dir) or \
        not os.path.exists(trt_gpu_log_dir): # 这个没有使用
        strCMD = "trtexec"
        strCMD += " --onnx=" + NetworkMap.infered_file_dir
        strCMD += " --saveEngine=" + trt_gpu_engine_file_dir
        strCMD += strShape
        strCMD += " --best" # --threads" # --profilingVerbosity=detailed" --useManagedMemory
        strCMD += " --exportProfile=" + trt_gpu_layer_information_file_dir
        strCMD += " 1>" + trt_gpu_log_dir + " 2>&1"
        print(strCMD, flush=True)
        time_begin = time.time()
        os.system(strCMD)
        time_duration = time.time() - time_begin
        print("time_duration = {}".format(time_duration), flush=True)

    # trtexec --loadEngine=/home/wfr/work/DLA/onnx_model_zoo/vgg19/vgg19_gpu_dla_manual.trt --useDLACore=1 --allowGPUFallback --iterations=20 --warmUp=500 --duration=8 #--useManagedMemory
    # trtexec 生成 GPU 上 单stream engine 及 性能(qps 每秒推理次数)
    if not os.path.exists(trt_gpu_perf_log_dir):
        strCMD = "trtexec"
        strCMD += " --loadEngine=" + trt_gpu_engine_file_dir
        strCMD += strShape
        strCMD += " --iterations=20 --warmUp=1000" # --useManagedMemory"
        strCMD += " 1>" + trt_gpu_perf_log_dir + " 2>&1"
        print(strCMD, flush=True)
        time_begin = time.time()
        os.system(strCMD)
        time_duration = time.time() - time_begin
        print("time_duration = {}".format(time_duration), flush=True)

    # # trtexec 生成 DLA 上 单stream engine 及 层之间聚合配置情况
    # if not os.path.exists(trt_dla_engine_file_dir) or not os.path.exists(trt_dla_layer_information_file_dir) or \
    #     not os.path.exists(trt_dla_log_dir): # 这个没有使用
    #     strCMD = "trtexec"
    #     strCMD += " --onnx=" + NetworkMap.infered_file_dir
    #     strCMD += " --saveEngine=" + trt_dla_engine_file_dir
    #     strCMD += strShape
    #     strCMD += " --useDLACore=1 --allowGPUFallback"
    #     strCMD += " --best --useManagedMemory --threads" # --profilingVerbosity=detailed"
    #     strCMD += " --exportProfile=" + trt_dla_layer_information_file_dir
    #     strCMD += " 1>" + trt_dla_log_dir + " 2>&1"
    #     print(strCMD)
    #     time_begin = time.time()
    #     os.system(strCMD)
    #     time_duration = time.time() - time_begin
    #     print("time_duration = {}".format(time_duration))

    # # trtexec 生成 DLA 上 单stream engine 及 性能(qps 每秒推理次数)
    # if not os.path.exists(trt_dla_perf_log_dir):
    #     strCMD = "trtexec"
    #     strCMD += " --loadEngine=" + trt_dla_engine_file_dir
    #     strCMD += strShape
    #     strCMD += " --useDLACore=1 --iterations=20 --warmUp=1000 --useManagedMemory"
    #     strCMD += " 1>" + trt_dla_perf_log_dir + " 2>&1"
    #     print(strCMD)
    #     time_begin = time.time()
    #     os.system(strCMD)
    #     time_duration = time.time() - time_begin
    #     print("time_duration = {}".format(time_duration))


    ################ 上边对模型整体进行 trtexec, 收集各层融合情况信息 ################
    
    # 创建辅助模型: 将 有未知维度 的 输出 都添加为 模型输出
    NetworkMap.createAllOutputModel()
    # 在CPU上运行辅助模型, 根据输出, 修补未知维度 并 保存输出到文件
    get_all_output(NetworkMap, dictInputTensor, dictTensorShape)

    # exit(0)

    # 生成 trt engine 时 / 测量时, 使用保存的输入数据

    # 读取 xxxxx_trtexec_perf_gpu.log, 提取 节点融合信息及执行时间
    # 相信 NVIDIA TensorRT 的节点融合 能够 提高性能和能效, 因此将融合节点整体处理, 不再拆分
    if os.path.exists(trt_gpu_log_dir) and os.path.exists(trt_gpu_layer_information_file_dir):
        # listDLALayerNames, listGPULayerNames = extract_layer.extrat_layer_from_log(trt_gpu_log_dir)

        listDLALayerNames, listDLALayerTime, listGPULayerNames, listGPULayerTime = extract_layer.extrat_layer_from_json(trt_gpu_layer_information_file_dir)

        for i in range(len(listGPULayerNames)):
            tmpListGPULayerNames, tmpListGPULayerTime = splitLayers(NetworkMap, listGPULayerNames[i], listGPULayerTime[i])
            for j in range(len(tmpListGPULayerNames)):
                flag = NetworkMap.fuseNode(tmpListGPULayerNames[j])
                if flag == True:
                    NetworkMap.dictFusedNode[tmpListGPULayerNames[j][0]].gpu_profiling.avg_exe_time = tmpListGPULayerTime[j]

    # 按融合节点对模型进行拆分, 一个融合节点独立构造一个 onnx模型
    # 将单节点 onnx模型 使用 trtexec 转换, 设置尽量使用 DLA, 解析 layer_information.json
    # 判断节点是否可以运行在 GPU/DLA
    # print("Fused NN layers can run on DLA/GPU: ")
    # 获得 FusedNode 的无重复项 list, 且按照 id 升序排列
    listFusedNode = NetworkMap.getListFusedNode()
    for tmpFusedNode in listFusedNode:
        print("FusedNode: {}".format([onnx_node.name for onnx_node in tmpFusedNode.list_onnx_node]), flush=True)
    print("", flush=True)

    # 对所有节点进行 trtexec, 判断是否能运行在 GPU/DLA, 并进行能耗测量
    for FusedNode in listFusedNode:

        SubModelName = FusedNode.getName(NetworkMap.useless_prefix)
        print("\nSubModelName: {}".format(SubModelName), flush=True)

        # 先在 dictOps_listFused 中搜索是否有所有op完全相同的 FusedNode, 如有则直接读取已有的测量结果 并 存入当前 FusedNode, 否则将当前 FusedNode 存入 dictOps_listFused
        FusedNode_SameOps = NetworkMap.findFusedNode_SameOps(FusedNode)
        if FusedNode_SameOps == None:
            NetworkMap.saveFusedNode_SameOps(FusedNode)
        else:
            print("使用: {}".format(FusedNode_SameOps.getName(NetworkMap.useless_prefix)), flush=True)
            FusedNode.copyProfilingData(FusedNode_SameOps)
            continue
        
        sub_onnx_model_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + ".onnx")
        sub_trt_dla_engine_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_dla.trt")
        sub_trt_dla_layer_information_file_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_layer_information_dla.json")
        sub_trt_dla_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_verbose_dla.log")
        sub_trt_dla_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_perf_dla.log")

        sub_trt_gpu_engine_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_gpu.trt")
        sub_trt_gpu_layer_information_file_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_layer_information_gpu.json")
        sub_trt_gpu_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_verbose_gpu.log")
        sub_trt_gpu_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_perf_gpu.log")

        # 尝试构造子模型, 不成功则跳过
        # list_onnx_node = [4,5,6,7,8,9,10,11]
        # list_onnx_node = ["resnetv22_pool1_fwd"]
        if not os.path.exists(sub_onnx_model_dir):
            SubModel, isValidModel = NetworkMap.createSubOnnxModel(FusedNode.list_onnx_node, SubModelName)
            # onnx.save_model(SubModel, sub_onnx_model_dir)
            if isValidModel == True:
                onnx.save_model(SubModel, sub_onnx_model_dir)
            else:
                tmp_dir = re.sub("\.onnx", "_error.onnx", sub_onnx_model_dir)
                onnx.save_model(SubModel, tmp_dir)
                print("跳过子节点: {}".format(SubModelName), flush=True)
                continue
        
        # 这里判断融合节点中各个 layer 是否都在 NetworkMap.setDLALayerName
        isCanUseDLA = FusedNode.canUseDLA(NetworkMap.setDLALayerName)
        if isCanUseDLA == False:
            continue # 跳过不能在 DLA 上运行的 层, 不生成 engine, 不进行测量

        SubNetworkMap = network_type.network_map(sub_onnx_model_dir)

        # 设置输入数据 shape 和 加载静态输入数据
        strShape = ""
        strInput = ""
        # strShape = " --shapes=\"TFNodes/yolo_evaluation_layer_1/Reshape__10:0\":1x1x1x3x2"
        # strInput = " --loadInputs=\"TFNodes/yolo_evaluation_layer_1/Reshape__10:0\":./input_tensor.dat"

        # 先判断当前节点是否需要加载静态输入 (有时静态输入才是合法输入)
        # 如果需要 则 构造加载文件命令, 指定维度命令应该是不需要了
        # 因为 此时 所有维度应该都确定了 没有未知维度了
        # 当输入数据会影响输出维度的时候应该需要加载静态输入, 这个不好判断, 因为和节点类型有关

        # 使用简单逻辑, 如果存在静态输入数据就使用
        for input in SubNetworkMap.listModelInput:
            # 静态文件名
            data_name = get_legal_name(input.name, NetworkMap.useless_prefix)
            # 静态文件路径
            data_dir = os.path.join(NetworkMap.onnx_folder_dir, data_name+".dat")
            if not os.path.exists(data_dir):
                continue
            strInput += "\"" + input.name + "\":" + data_dir + ","
            # " --loadInputs=\"TFNodes/yolo_evaluation_layer_1/Reshape__10:0\":./input_tensor.dat"
        if len(strInput) > 0:
            strInput = strInput[:-1]
            strInput = " --loadInputs=" + strInput

        # 下边对 融合节点 进行 trtexec
        if not os.path.exists(sub_trt_dla_engine_dir) or not os.path.exists(sub_trt_dla_log_dir) or \
            not os.path.exists(sub_trt_dla_layer_information_file_dir):
            # 尽可能使用DLA: trtexec --onnx=./resnet18-v2-7.onnx --saveEngine=resnet_engine_cli.trt --shapes=data:4x3x224x224 --useDLACore=0 --allowGPUFallback --useManagedMemory --threads --exportProfile=./layer_information.json
            strCMD = "trtexec"
            strCMD += " --onnx=\"" + sub_onnx_model_dir + "\""
            strCMD += " --saveEngine=\"" + sub_trt_dla_engine_dir + "\""
            strCMD += strInput
            strCMD += " --useDLACore=1 --best --allowGPUFallback" # --useManagedMemory --threads" # --profilingVerbosity=detailed"
            strCMD += " --exportProfile=\"" + sub_trt_dla_layer_information_file_dir + "\""
            strCMD += " 1>\"" + sub_trt_dla_log_dir + "\"" + " 2>&1"
            # print(strCMD, flush=True)
            os.system(strCMD)
        # trtexec --loadEngine=/home/wfr/work/DLA/onnx_model_zoo/vgg19/vgg19_gpu_dla_manual.trt --useDLACore=1 --iterations=20 --warmUp=500 --duration=8 #--useManagedMemory
        # if not os.path.exists(sub_trt_dla_perf_log_dir) and os.path.exists(sub_trt_dla_engine_dir): # 这个没有使用
        #     strCMD = "trtexec"
        #     strCMD += " --loadEngine=\"" + sub_trt_dla_engine_dir + "\""
        #     strCMD += strInput
        #     strCMD += " --useDLACore=1 --iterations=20 --warmUp=1000 --useManagedMemory"
        #     # strCMD += " --useDLACore=0 --allowGPUFallback --best --useManagedMemory --threads"
        #     strCMD += " 1>\"" + sub_trt_dla_perf_log_dir + "\"" + " 2>&1"
        #     # print(strCMD)
        #     os.system(strCMD)

        if not os.path.exists(sub_trt_gpu_engine_dir) or not os.path.exists(sub_trt_gpu_log_dir) or \
            not os.path.exists(sub_trt_gpu_layer_information_file_dir):
            strCMD = "trtexec"
            strCMD += " --onnx=\"" + sub_onnx_model_dir + "\""
            strCMD += " --saveEngine=\"" + sub_trt_gpu_engine_dir + "\""
            strCMD += strInput
            strCMD += " --best" # --useManagedMemory --threads" # --profilingVerbosity=detailed"
            strCMD += " --exportProfile=\"" + sub_trt_gpu_layer_information_file_dir + "\""
            strCMD += " 1>\"" + sub_trt_gpu_log_dir + "\"" + " 2>&1"
            # print(strCMD, flush=True)
            os.system(strCMD)
        # if not os.path.exists(sub_trt_gpu_perf_log_dir) and os.path.exists(sub_trt_gpu_engine_dir): # 这个没有使用
        #     strCMD = "trtexec"
        #     strCMD += " --loadEngine=\"" + sub_trt_gpu_engine_dir + "\""
        #     strCMD += strInput
        #     strCMD += " --useDLACore=1 --iterations=20 --warmUp=1000 --useManagedMemory"
        #     # strCMD += " --best --useManagedMemory --threads"
        #     strCMD += " 1>\"" + sub_trt_gpu_perf_log_dir + "\"" + " 2>&1"
        #     # print(strCMD)
        #     os.system(strCMD)

        # 这里使用 sub_trt_dla_engine_dir 和 sub_trt_dla_log_dir 文件来判断 融合节点是否能运行在 DLA
        # 相信 NVIDIA TensorRT 的节点融合 能够 提高性能和能效, 因此将融合节点整体处理, 不再拆分
        if os.path.exists(sub_trt_dla_engine_dir):
            # 通过 sub_trt_dla_log_dir 文件判断是否所有节点都跑在 DLA 上
            listDLALayerNames, listGPULayerNames = extract_layer.extrat_layer_from_log(sub_trt_dla_log_dir) # sub_trt_dla_perf_log_dir sub_trt_dla_log_dir
            NumGPULayers = NetworkMap.countListLayers(listGPULayerNames)
            NumDLALayers = NetworkMap.countListLayers(listDLALayerNames)
            countForeignNode = extract_layer.count_ForeignNode(sub_trt_dla_layer_information_file_dir)

            print("listDLALayerNames: {}".format(listDLALayerNames), flush=True)
            print("listGPULayerNames: {}".format(listGPULayerNames), flush=True)
            print("countForeignNode: {}".format(countForeignNode), flush=True)
            
            # 如果没有层运行在 GPU 上, 即全部层都运行在 DLA 上, 且 只有个一 DLA 融合节点
            if NumGPULayers == 0 and NumDLALayers > 0 and countForeignNode == 1:
                FusedNode.isCanUseDLA = True
            else:
                FusedNode.isCanUseDLA = False
        
        # 如果没有生成 sub_trt_dla_engine_dir, 即当前融合节点不能在 DLA 上运行
        else:
            FusedNode.isCanUseDLA = False
        print("{}: isCanUseDLA = {}".format(SubModelName, FusedNode.isCanUseDLA), flush=True)
            
        # 这里使用 sub_trt_gpu_engine_dir 和 sub_trt_gpu_log_dir 文件来判断 融合节点是否能运行在 GPU
        if os.path.exists(sub_trt_gpu_engine_dir):
            # 通过 sub_trt_gpu_log_dir 文件判断是否所有节点都跑在 GPU 上
            listDLALayerNames, listGPULayerNames = extract_layer.extrat_layer_from_log(sub_trt_gpu_log_dir) # sub_trt_gpu_perf_log_dir sub_trt_gpu_log_dir
            NumDLALayers = NetworkMap.countListLayers(listDLALayerNames)

            # if NumDLALayers == 0: # 如果没有层运行在 DLA 上, 即全部层都运行在 GPU 上
            #     FusedNode.isCanUseGPU = True
            # else:
            #     FusedNode.isCanUseGPU = False
        
        # 如果没有生成 sub_trt_gpu_engine_dir, 即当前融合节点不能在 GPU 上运行
        else:
            pass
            # FusedNode.isCanUseGPU = False
            # 注意好像有的层不能运行在GPU上, 好像见过一次这种情况
        print("{}: isCanUseGPU = {}".format(SubModelName, FusedNode.isCanUseGPU), flush=True)

    ##########################################################################################
    # 下边进行能耗测量

    # 只对能在 DLA 上运行的融合节点进行测量
    print("\n\n开始能耗测量: ", flush=True)
    energy_measurement_output_log_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_eng-perf_all.log")
    with orin_nx_measuring(0.05) as OrinNXMeasuring, \
        open(energy_measurement_output_log_dir, "w") as energy_file:
        MeasurementDuration = 8

        # 测量模型整体的能耗
        eng_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_eng-perf.log")
        isUseDLA, isUseGPU, tmpStrAccu = measure_save_network_eng_perf(eng_perf_log_dir, NetworkMap, NetworkMap.idle_power, "", OrinNXMeasuring, trt_dla_engine_file_dir, trt_dla_perf_log_dir, trt_gpu_engine_file_dir, trt_gpu_perf_log_dir, MeasurementDuration, False, dictInputTensor, dictTensorShape)
        energy_file.write(tmpStrAccu)
        energy_file.flush()

        # exit(0)

        # 测量各个 融合节点在 GPU/DLA 上的能耗/性能
        print("\nlen(listFusedNode) = {}".format(len(listFusedNode)), flush=True)
        MeasurementDuration = 3
        dla_sum_exe_time, dla_sum_energy = 0, 0
        gpu_sum_exe_time, gpu_sum_energy = 0, 0
        dla_dla_count, dla_gpu_count = 0, 0
        gpu_dla_count, gpu_gpu_count = 0, 0
        for FusedNode in listFusedNode:

            SubModelName = FusedNode.getName(NetworkMap.useless_prefix)

            sub_onnx_model_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + ".onnx")
            sub_trt_gpu_engine_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_gpu.trt")
            sub_trt_dla_engine_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_dla.trt")
            sub_trt_gpu_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_perf_gpu.log")
            sub_trt_dla_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_perf_dla.log")
            sub_trt_gpu_layer_information_file_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_layer_information_gpu.json")
            sub_trt_dla_layer_information_file_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_layer_information_dla.json")

            # 这里使用文件保存测量结果, 起到保存测量进度的作用
            sub_eng_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_eng-perf.log")

            isUseDLA, isUseGPU = False, False

            # 先在 dictOps_listFused 中搜索是否有所有op完全相同的 FusedNode, 如有则直接读取已有的测量结果 并 存入当前 FusedNode, 否则将当前 FusedNode 存入 dictOps_listFused
            FusedNode_SameOps = NetworkMap.findFusedNode_SameOps(FusedNode)
            if FusedNode_SameOps != None and (FusedNode_SameOps.gpu_profiling.avg_energy > 0.0 or FusedNode_SameOps.dla_profiling.avg_energy > 0.0):
                # 使用已经测量的测量数据设置 FusedNode
                print("使用: {}".format(FusedNode_SameOps.getName(NetworkMap.useless_prefix)), flush=True)
                FusedNode.copyProfilingData(FusedNode_SameOps)
                # 打印及保存数据
                isUseDLA, isUseGPU, tmpStrAccu = save_eng_perf_log_dir(sub_eng_perf_log_dir, FusedNode, sub_trt_dla_engine_dir, sub_trt_gpu_engine_dir, NetworkMap.useless_prefix)
                energy_file.write(tmpStrAccu)
                energy_file.flush()
            else:

                # 如果不是 GPU / DLA 上都能运行, 就跳过
                if (not os.path.exists(sub_trt_dla_engine_dir)) or (not os.path.exists(sub_trt_gpu_engine_dir)):
                    continue

                # 使用简单逻辑, 如果存在静态输入数据就使用
                SubNetworkMap = network_type.network_map(sub_onnx_model_dir)
                tmpDictInputTensor, tmpDictTensorShape = {}, {}
                listTmp = SubNetworkMap.listModelInput # + SubNetworkMap.listModelOutput
                for input in listTmp:
                    # 静态文件名
                    data_name = get_legal_name(input.name, NetworkMap.useless_prefix)
                    # 静态文件路径
                    data_dir = os.path.join(NetworkMap.onnx_folder_dir, data_name+".dat")
                    if not os.path.exists(data_dir):
                        continue
                    with open(data_dir, "rb") as data_file:
                        tmpDictInputTensor[input.name] = pickle.load(data_file)
                        tmpDictTensorShape[input.name] = tmpDictInputTensor[input.name].shape

                isUseDLA, isUseGPU, tmpStrAccu = measure_save_network_eng_perf(sub_eng_perf_log_dir, FusedNode, NetworkMap.idle_power, NetworkMap.useless_prefix, OrinNXMeasuring, sub_trt_dla_engine_dir, sub_trt_dla_layer_information_file_dir, sub_trt_gpu_engine_dir, sub_trt_gpu_layer_information_file_dir, MeasurementDuration, False, tmpDictInputTensor, tmpDictTensorShape)
                energy_file.write(tmpStrAccu)
                energy_file.flush()
            
            if isUseDLA == True and isUseGPU == True:
                dla_sum_exe_time += FusedNode.dla_profiling.avg_exe_time
                dla_sum_energy += FusedNode.dla_profiling.avg_energy
                dla_dla_count += 1

                gpu_sum_exe_time += FusedNode.gpu_profiling.avg_exe_time
                gpu_sum_energy += FusedNode.gpu_profiling.avg_energy
                gpu_gpu_count += 1

            elif isUseDLA == True and isUseGPU == False:
                dla_sum_exe_time += FusedNode.dla_profiling.avg_exe_time
                dla_sum_energy += FusedNode.dla_profiling.avg_energy
                gpu_sum_exe_time += FusedNode.dla_profiling.avg_exe_time
                gpu_sum_energy += FusedNode.dla_profiling.avg_energy
                dla_dla_count += 1
                gpu_dla_count += 1

            elif isUseDLA == False and isUseGPU == True:
                dla_sum_exe_time += FusedNode.gpu_profiling.avg_exe_time
                dla_sum_energy += FusedNode.gpu_profiling.avg_energy
                gpu_sum_exe_time += FusedNode.gpu_profiling.avg_exe_time
                gpu_sum_energy += FusedNode.gpu_profiling.avg_energy
                dla_gpu_count += 1
                gpu_gpu_count += 1

        dla_qps, dla_power, gpu_qps, gpu_power = 0.0, 0.0, 0.0, 0.0
        if dla_sum_exe_time > 0:
            dla_power = dla_sum_energy/dla_sum_exe_time
            dla_qps = 1/dla_sum_exe_time
        if gpu_sum_exe_time > 0:
            gpu_power = gpu_sum_energy/gpu_sum_exe_time
            gpu_qps = 1/gpu_sum_exe_time
        tmpStr = "\nsum_results:\n"
        tmpStr += "dla_energy: {:.4e} J; dla_time: {:.4e} s; dla_power: {:.4e} W; dla_qps: {:.2f} q/s\n".format(dla_sum_energy, dla_sum_exe_time, dla_power, dla_qps)
        tmpStr += "gpu_energy: {:.4e} J; gpu_time: {:.4e} s; gpu_power: {:.4e} W; gpu_qps: {:.2f} q/s\n".format(gpu_sum_energy, gpu_sum_exe_time, gpu_power, gpu_qps)
        if gpu_sum_energy > dla_sum_energy and gpu_sum_energy > 0:
            tmpStr += "DLA 节能 {:.4%}\n".format((gpu_sum_energy - dla_sum_energy)/gpu_sum_energy)
        print(tmpStr, end="", flush=True)
        energy_file.write(tmpStr)
        energy_file.flush()

        # 两组测量(尽量使用DLA/GPU)中 使用 DLA/GPU 的数量
        tmpArray = np.array([dla_dla_count, dla_gpu_count])
        NetworkMap.array_dla_gpu_count = tmpArray
        tmpArray = np.array([gpu_dla_count, gpu_gpu_count])
        NetworkMap.array_dla_gpu_count = np.vstack((NetworkMap.array_dla_gpu_count, tmpArray))

        # 两组测量(尽量使用DLA/GPU)中 各层加和时间 减去 整体测量时间
        tmpArray = np.array([dla_sum_exe_time - NetworkMap.dla_profiling.avg_exe_time])
        NetworkMap.array_delta_time = tmpArray
        tmpArray = np.array([gpu_sum_exe_time - NetworkMap.gpu_profiling.avg_exe_time])
        NetworkMap.array_delta_time = np.vstack((NetworkMap.array_delta_time, tmpArray))

        # tmpNetworkMap = NetworkMap
        # 保存测量数据
        # print("dump energy_measurement_data_file")
        # joblib.dump(NetworkMap, energy_measurement_data_file_dir)

    # 推导 纯GPU engine 的 运行时间
    for FusedNode in listFusedNode:
        NetworkMap.sumTimeGPU += FusedNode.gpu_profiling.avg_exe_time
        NetworkMap.sumEnergyGPU += FusedNode.dla_profiling.avg_energy

    # 下边找出 SingleIOMap 并进行补充测量
    NetworkMap.getAllSingleIOMap()
    # exit(0)

    # 需要封装的功能: 生成 onnx 子模型, 生成 trt 模型, 测量性能-能耗

    energy_file = open(energy_measurement_output_log_dir, "a")
    energy_file.write("\n\nSingleIOMaps:\n")
    energy_file.flush()

    for tmpName, tmpSingleIOMap in NetworkMap.dictSingleIOMap.items():

        if isinstance(tmpName, str) != True:
            continue

        print("Measure SingleIOMap: {}".format(tmpSingleIOMap.name))

        SubModelName = tmpSingleIOMap.name

        sub_onnx_model_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + ".onnx")
        sub_trt_dla_engine_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_dla.trt")
        sub_trt_dla_layer_information_file_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_layer_information_dla.json")
        sub_trt_dla_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_verbose_dla.log")
        sub_trt_dla_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_perf_dla.log")

        sub_trt_gpu_engine_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_gpu.trt")
        sub_trt_gpu_layer_information_file_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_layer_information_gpu.json")
        sub_trt_gpu_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_verbose_gpu.log")
        sub_trt_gpu_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_perf_gpu.log")

        # 生成 子onnx 所需输入: NetworkMap, 模型名称, 保存文件路径
        tmpFlag = NetworkMap.generateSubOnnxFile(tmpSingleIOMap.list_network_node_id, SubModelName, NetworkMap.onnx_folder_dir)
        if tmpFlag == False:
            continue

        generateTRTEngine(sub_onnx_model_dir, NetworkMap.onnx_folder_dir, SubModelName)

        # 测量 仅GPU / 尽量DLA engine 的 性能-能耗, 并 保存到文件
        eng_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, "{}_eng-perf.log".format(SubModelName))
        MeasurementDuration = 3
        isUseDLA, isUseGPU, tmpStrAccu = measureTRTEngine(eng_perf_log_dir, tmpSingleIOMap, sub_trt_gpu_engine_dir, sub_trt_dla_engine_dir, NetworkMap.idle_power, NetworkMap.useless_prefix, MeasurementDuration, dictInputTensor, dictTensorShape)
        energy_file.write(tmpStrAccu)
        energy_file.flush()

        tmpSingleIOMap.accumulateFusedNodeData(True)
        tmpSingleIOMap.calculateBubbleTime()

        print("tmpSingleIOMap.list_entry_fused_node:")
        for tmpFusedNode in tmpSingleIOMap.list_entry_fused_node:
            print("{}".format(tmpFusedNode.getName(tmpSingleIOMap.NetworkMap.useless_prefix)))
        print("tmpSingleIOMap.list_exit_fused_node:")
        for tmpFusedNode in tmpSingleIOMap.list_exit_fused_node:
            print("{}".format(tmpFusedNode.getName(tmpSingleIOMap.NetworkMap.useless_prefix)))
        print("")


    # end for
    energy_file.close()

    # exit(0)

    return

# SubNetwork 可以是: network_map / fused_node / SUB_GRAPH
def measureTRTEngine(eng_perf_log_dir, SubNetwork, trt_gpu_engine_dir, trt_dla_engine_dir, idle_power, useless_prefix, MeasurementDuration = 2,dictInputTensor = {}, dictTensorShape = {}):
    
    tmpStrAccu = "\n"
    tmpStr = SubNetwork.getName(useless_prefix) + ":\n"
    print("\nmeasureTRTEngine: {}".format(tmpStr), end="", flush=True)
    tmpStrAccu += tmpStr

    print("measureTRTEngine: eng_perf_log_dir: {}".format(eng_perf_log_dir), flush=True)
    print("measureTRTEngine: trt_gpu_engine_dir: {}".format(trt_gpu_engine_dir), flush=True)
    print("measureTRTEngine: trt_dla_engine_dir: {}".format(trt_dla_engine_dir), flush=True)

    isUseDLA, isUseGPU = False, False

    # if False:
    if os.path.exists(eng_perf_log_dir):
        isUseDLA, isUseGPU = extractEngPerf(SubNetwork, eng_perf_log_dir, idle_power)

        print("dla_energy = {} J".format(SubNetwork.dla_profiling.avg_energy), flush=True)
        print("dla_time = {} s".format(SubNetwork.dla_profiling.avg_exe_time), flush=True)
        dla_qps = 0.0
        if SubNetwork.dla_profiling.avg_exe_time > 0:
            dla_qps = 1/SubNetwork.dla_profiling.avg_exe_time
        print("dla_qps = {} q/s".format(dla_qps), flush=True)

        if SubNetwork.dla_profiling.avg_energy > 0:
            ModelName = os.path.split(trt_dla_engine_dir)[1]
            ModelName = os.path.splitext(ModelName)[0]
            if not isinstance(SubNetwork, network_type.network_map):
                tmpStr = "sub_model: {}\n".format(ModelName)
            else:
                tmpStr = ModelName + ":\n"
            tmpStrAccu += tmpStr
            tmpStr = "dla_energy: {:.4e} J; dla_time: {:.4e} s; dla_power: {:.4e} W; dla_qps: {:.2f} q/s; dla_input_time: {:.4e} s; dla_output_time: {:.4e} s\n".format(SubNetwork.dla_profiling.avg_energy, SubNetwork.dla_profiling.avg_exe_time, SubNetwork.dla_profiling.avg_power, dla_qps, SubNetwork.dla_profiling.avg_input_time, SubNetwork.dla_profiling.avg_output_time)
            tmpStrAccu += tmpStr

        print("gpu_energy = {} J".format(SubNetwork.gpu_profiling.avg_energy), flush=True)
        print("gpu_time = {} s".format(SubNetwork.gpu_profiling.avg_exe_time), flush=True)
        gpu_qps = 0.0
        if SubNetwork.gpu_profiling.avg_exe_time > 0:
            gpu_qps = 1/SubNetwork.gpu_profiling.avg_exe_time
        print("gpu_qps = {} q/s".format(gpu_qps), flush=True)

        if SubNetwork.gpu_profiling.avg_energy > 0:
            ModelName = os.path.split(trt_gpu_engine_dir)[1]
            ModelName = os.path.splitext(ModelName)[0]
            if not isinstance(SubNetwork, network_type.network_map):
                tmpStr = "sub_model: {}\n".format(ModelName)
            else:
                tmpStr = ModelName + ":\n"
            tmpStrAccu += tmpStr
            tmpStr = "gpu_energy: {:.4e} J; gpu_time: {:.4e} s; gpu_power: {:.4e} W; gpu_qps: {:.2f} q/s; gpu_input_time: {:.4e} s; gpu_output_time: {:.4e} s\n".format(SubNetwork.gpu_profiling.avg_energy, SubNetwork.gpu_profiling.avg_exe_time, SubNetwork.gpu_profiling.avg_power, gpu_qps, SubNetwork.gpu_profiling.avg_input_time, SubNetwork.gpu_profiling.avg_output_time)
            tmpStrAccu += tmpStr

        if isUseDLA == True and isUseGPU == True and \
            SubNetwork.gpu_profiling.avg_energy > SubNetwork.dla_profiling.avg_energy and SubNetwork.gpu_profiling.avg_energy > 0:
        
            tmpStr = "DLA 节能 {:.4%}\n".format((SubNetwork.gpu_profiling.avg_energy - SubNetwork.dla_profiling.avg_energy)/SubNetwork.gpu_profiling.avg_energy)
            print("{}".format(tmpStr), end="", flush=True)
            tmpStrAccu += tmpStr

    else:
        # 测量能耗/性能, 并写入文件
        OrinNXMeasuring = orin_nx_measuring(0.05)

                # 测量子节点在 DLA 上的能耗-性能
        if os.path.exists(trt_dla_engine_dir):
            ModelName = SubNetwork.getName(useless_prefix)
            tmpDir = os.path.split(trt_dla_engine_dir)[0]

            trt_dla_layer_information_file_dir = os.path.join(tmpDir, "{}_layer_information_dla.json".format(ModelName))
            trt_dla_perf_log_dir = os.path.join(tmpDir, "{}_trtexec_perf_dla.log".format(ModelName))

            print("measureTRTEngine: trt_dla_layer_information_file_dir: {}".format(trt_dla_layer_information_file_dir), flush=True)
            print("measureTRTEngine: trt_dla_perf_log_dir: {}".format(trt_dla_perf_log_dir), flush=True)

            if not os.path.exists(trt_dla_layer_information_file_dir) and os.path.exists(trt_dla_engine_dir):
            
                strCMD = "trtexec"
                strCMD += " --loadEngine=\"" + trt_dla_engine_dir + "\""
                strCMD += " --useDLACore=0 --allowGPUFallback"
                # strCMD += " --best" # --threads --useManagedMemory" # --useManagedMemory
                strCMD += " --exportProfile=\"" + trt_dla_layer_information_file_dir + "\""
                strCMD += " 1>/dev/null 2>&1"
                print(strCMD, flush=True)
                # time_begin = time.time()
                os.system(strCMD)
                # time_duration = time.time() - time_begin
                # print("generating {} engine time: {}".format(DEVICE, time_duration), flush=True)

            # if True:
            if not os.path.exists(trt_dla_perf_log_dir) and os.path.exists(trt_dla_engine_dir):
            
                strCMD = "trtexec"
                strCMD += " --loadEngine=\"" + trt_dla_engine_dir + "\""
                strCMD += " --useDLACore=0 --allowGPUFallback"
                # strCMD += " --best" # --threads --useManagedMemory" # --useManagedMemory
                strCMD += " 1>" + trt_dla_perf_log_dir + " 2>&1"
                print(strCMD, flush=True)
                # time_begin = time.time()
                os.system(strCMD)
                # time_duration = time.time() - time_begin
                # print("generating {} engine time: {}".format(DEVICE, time_duration), flush=True)

            avg_input_time, avg_exe_time, avg_output_time = 0, 0, 0
            if isinstance(SubNetwork, network_type.network_map):
                avg_exe_time = getMeanLatency(trt_dla_perf_log_dir)
            else:
                avg_input_time, avg_exe_time, avg_output_time = getAvgTimes(trt_dla_layer_information_file_dir)

            measure_network_energy(OrinNXMeasuring, trt_dla_engine_dir, MeasurementDuration, False, dictInputTensor, dictTensorShape)

            SubNetwork.dla_profiling.avg_input_time = avg_input_time
            SubNetwork.dla_profiling.avg_output_time = avg_output_time
            SubNetwork.dla_profiling.avg_exe_time = avg_exe_time
            SubNetwork.dla_profiling.avg_power = OrinNXMeasuring.power_vdd.value
            SubNetwork.dla_profiling.avg_energy = SubNetwork.dla_profiling.avg_exe_time * SubNetwork.dla_profiling.avg_power

            SubNetwork.dla_profiling.power_dynamic = SubNetwork.dla_profiling.avg_power - idle_power
            SubNetwork.dla_profiling.energy_dynamic = SubNetwork.dla_profiling.power_dynamic * SubNetwork.dla_profiling.avg_exe_time
            SubNetwork.dla_profiling.energy_static = idle_power * SubNetwork.dla_profiling.avg_exe_time

            print("dla_energy = {} J".format(SubNetwork.dla_profiling.avg_energy), flush=True)
            print("dla_time = {} s".format(SubNetwork.dla_profiling.avg_exe_time), flush=True)
            dla_qps = 0.0
            if SubNetwork.dla_profiling.avg_exe_time > 0:
                dla_qps = 1/SubNetwork.dla_profiling.avg_exe_time
            print("dla_qps = {} q/s".format(dla_qps), flush=True)

            ModelName = os.path.split(trt_dla_engine_dir)[1]
            ModelName = os.path.splitext(ModelName)[0]
            if not isinstance(SubNetwork, network_type.network_map):
                tmpStr = "sub_model: {}\n".format(ModelName)
            else:
                tmpStr = ModelName + ":\n"
            tmpStrAccu += tmpStr
            tmpStr = "dla_energy: {:.4e} J; dla_time: {:.4e} s; dla_power: {:.4e} W; dla_qps: {:.2f} q/s; dla_input_time: {:.4e} s; dla_output_time: {:.4e} s\n".format(SubNetwork.dla_profiling.avg_energy, SubNetwork.dla_profiling.avg_exe_time, SubNetwork.dla_profiling.avg_power, dla_qps, SubNetwork.dla_profiling.avg_input_time, SubNetwork.dla_profiling.avg_output_time)
            tmpStrAccu += tmpStr
            
            isUseDLA = True

        # 测量子节点在 GPU 上的能耗-性能
        if os.path.exists(trt_gpu_engine_dir):
            ModelName = SubNetwork.getName(useless_prefix)
            tmpDir = os.path.split(trt_gpu_engine_dir)[0]

            trt_gpu_layer_information_file_dir = os.path.join(tmpDir, "{}_layer_information_gpu.json".format(ModelName))
            trt_gpu_perf_log_dir = os.path.join(tmpDir, "{}_trtexec_perf_gpu.log".format(ModelName))

            print("measureTRTEngine: trt_gpu_layer_information_file_dir: {}".format(trt_gpu_layer_information_file_dir), flush=True)
            print("measureTRTEngine: trt_gpu_perf_log_dir: {}".format(trt_gpu_perf_log_dir), flush=True)

            if not os.path.exists(trt_gpu_layer_information_file_dir) and os.path.exists(trt_gpu_engine_dir):
            
                strCMD = "trtexec"
                strCMD += " --loadEngine=\"" + trt_gpu_engine_dir + "\""

                # if DEVICE == "DLA":
                #     strCMD += " --useDLACore=0 --allowGPUFallback"

                # strCMD += " --best" # --threads --useManagedMemory" # --useManagedMemory
                strCMD += " --exportProfile=\"" + trt_gpu_layer_information_file_dir + "\""
                strCMD += " 1>/dev/null 2>&1"
                print(strCMD, flush=True)
                # time_begin = time.time()
                os.system(strCMD)
                # time_duration = time.time() - time_begin
                # print("generating {} engine time: {}".format(DEVICE, time_duration), flush=True)

            # if True:
            if not os.path.exists(trt_gpu_perf_log_dir) and os.path.exists(trt_gpu_engine_dir):
            
                strCMD = "trtexec"
                strCMD += " --loadEngine=\"" + trt_gpu_engine_dir + "\""

                # if DEVICE == "DLA":
                #     strCMD += " --useDLACore=0 --allowGPUFallback"

                # strCMD += " --best" # --threads --useManagedMemory" # --useManagedMemory
                strCMD += " 1>" + trt_gpu_perf_log_dir + " 2>&1"
                print(strCMD, flush=True)
                # time_begin = time.time()
                os.system(strCMD)
                # time_duration = time.time() - time_begin
                # print("generating {} engine time: {}".format(DEVICE, time_duration), flush=True)

            avg_input_time, avg_exe_time, avg_output_time = 0, 0, 0
            if isinstance(SubNetwork, network_type.network_map):
                avg_exe_time = getMeanLatency(trt_gpu_perf_log_dir)
            else:
                avg_input_time, avg_exe_time, avg_output_time = getAvgTimes(trt_gpu_layer_information_file_dir)

            measure_network_energy(OrinNXMeasuring, trt_gpu_engine_dir, MeasurementDuration, False, dictInputTensor, dictTensorShape)

            SubNetwork.gpu_profiling.avg_input_time = avg_input_time
            SubNetwork.gpu_profiling.avg_output_time = avg_output_time
            SubNetwork.gpu_profiling.avg_exe_time = avg_exe_time
            SubNetwork.gpu_profiling.avg_power = OrinNXMeasuring.power_vdd.value
            SubNetwork.gpu_profiling.avg_energy = SubNetwork.gpu_profiling.avg_exe_time * SubNetwork.gpu_profiling.avg_power

            SubNetwork.gpu_profiling.power_dynamic = SubNetwork.gpu_profiling.avg_power - idle_power
            SubNetwork.gpu_profiling.energy_dynamic = SubNetwork.gpu_profiling.power_dynamic * SubNetwork.gpu_profiling.avg_exe_time
            SubNetwork.gpu_profiling.energy_static = idle_power * SubNetwork.gpu_profiling.avg_exe_time

            print("gpu_energy = {} J".format(SubNetwork.gpu_profiling.avg_energy), flush=True)
            print("gpu_time = {} s".format(SubNetwork.gpu_profiling.avg_exe_time), flush=True)
            gpu_qps = 0.0
            if SubNetwork.gpu_profiling.avg_exe_time > 0:
                gpu_qps = 1/SubNetwork.gpu_profiling.avg_exe_time
            print("gpu_qps = {} q/s".format(gpu_qps), flush=True)

            ModelName = os.path.split(trt_gpu_engine_dir)[1]
            ModelName = os.path.splitext(ModelName)[0]
            if not isinstance(SubNetwork, network_type.network_map):
                tmpStr = "sub_model: {}\n".format(ModelName)
            else:
                tmpStr = ModelName + ":\n"
            tmpStrAccu += tmpStr
            tmpStr = "gpu_energy: {:.4e} J; gpu_time: {:.4e} s; gpu_power: {:.4e} W; gpu_qps: {:.2f} q/s; gpu_input_time: {:.4e} s; gpu_output_time: {:.4e} s\n".format(SubNetwork.gpu_profiling.avg_energy, SubNetwork.gpu_profiling.avg_exe_time, SubNetwork.gpu_profiling.avg_power, gpu_qps, SubNetwork.gpu_profiling.avg_input_time, SubNetwork.gpu_profiling.avg_output_time)
            tmpStrAccu += tmpStr
            
            isUseGPU = True
        
        if isUseDLA == True and isUseGPU == True and \
            SubNetwork.gpu_profiling.avg_energy > SubNetwork.dla_profiling.avg_energy and SubNetwork.gpu_profiling.avg_energy > 0:
        
            tmpStr = "DLA 节能 {:.4%}\n".format((SubNetwork.gpu_profiling.avg_energy - SubNetwork.dla_profiling.avg_energy)/SubNetwork.gpu_profiling.avg_energy)
            print("{}".format(tmpStr), end="", flush=True)
            tmpStrAccu += tmpStr

        print("SubNetwork.dla_profiling.avg_power = {:.4e} W".format(SubNetwork.dla_profiling.avg_power), flush=True)
        print("SubNetwork.gpu_profiling.avg_power = {:.4e} W".format(SubNetwork.gpu_profiling.avg_power), flush=True)
        if SubNetwork.dla_profiling.avg_power > 0 or SubNetwork.gpu_profiling.avg_power > 0:
            print(eng_perf_log_dir, flush=True)
            with open(eng_perf_log_dir, "w") as eng_perf_file:
                eng_perf_file.write(tmpStrAccu)

        OrinNXMeasuring.exit()

    return isUseDLA, isUseGPU, tmpStrAccu


# 生成 trt engine 所需输入: onnx文件路径, 仅GPU的 engine 的路径, 尽量DLA的 engine 的路径
def generateTRTEngine(ONNXFileDir, EngineFolderDir, EngineNamePrefix, listDevice = {}):

    print("generateTRTEngine: {}".format(ONNXFileDir))
    print("generateTRTEngine: {}".format(EngineFolderDir))
    print("generateTRTEngine: {}".format(EngineNamePrefix))

    NetworkMap = network_type.network_map(ONNXFileDir)

    # 使用简单逻辑, 如果存在静态输入数据就使用
    strInput = ""
    for input in NetworkMap.listModelInput:
        # 静态文件名
        data_name = get_legal_name(input.name, NetworkMap.useless_prefix)
        # 静态文件路径
        data_dir = os.path.join(NetworkMap.onnx_folder_dir, data_name+".dat")
        if not os.path.exists(data_dir):
            continue
        strInput += "\"" + input.name + "\":" + data_dir + ","
        # " --loadInputs=\"TFNodes/yolo_evaluation_layer_1/Reshape__10:0\":./input_tensor.dat"
    if len(strInput) > 0:
        strInput = strInput[:-1]
        strInput = " --loadInputs=" + strInput

    if len(listDevice) == 0:
        listDevice = ["gpu", "dla"]
    for device in listDevice:

        trt_engine_dir = os.path.join(EngineFolderDir, "{}_{}.trt".format(EngineNamePrefix, device))
        trt_layer_information_file_dir = os.path.join(EngineFolderDir, "{}_layer_information_{}.json".format(EngineNamePrefix, device))
        trt_log_dir = os.path.join(EngineFolderDir, "{}_trtexec_verbose_{}.log".format(EngineNamePrefix, device))
        trt_perf_log_dir = os.path.join(EngineFolderDir, "{}_trtexec_perf_{}.log".format(EngineNamePrefix, device))

        if not os.path.exists(trt_log_dir) or not os.path.exists(trt_engine_dir): # 这个没有使用
            
            strCMD = "trtexec"
            strCMD += " --onnx=\"" + ONNXFileDir + "\""
            strCMD += " --saveEngine=\"" + trt_engine_dir + "\""
            strCMD += strInput

            if device == "dla":
                strCMD += " --useDLACore=0 --allowGPUFallback"

            strCMD += " --best --buildOnly" # --threads --useManagedMemory
            strCMD += " 1>" + trt_log_dir + " 2>&1"
            print(strCMD, flush=True)
            # time_begin = time.time()
            os.system(strCMD)
            # time_duration = time.time() - time_begin
            # print("generating {} engine time: {}".format(device, time_duration), flush=True)
        # end if
    # end for
    return

def splitLayers(NetworkMap, GPULayerNames, GPULayerTime=None):

    listGPULayerNames = []
    listNodeName = []
    for NodeName in GPULayerNames:
        if not NodeName in NetworkMap.dictNetworkNode.keys():
            continue
        tmpNode = NetworkMap.dictNetworkNode[NodeName]

        # 是分支节点
        if len(tmpNode.list_parent) > 1 or len(tmpNode.list_child) > 1:
            if len(listNodeName) > 0:
                listGPULayerNames.append(listNodeName)
                listNodeName = []
            listGPULayerNames.append([NodeName])
        else:
            listNodeName.append(NodeName)

    if len(listNodeName) > 0:
        listGPULayerNames.append(listNodeName)

    countLayer = len(GPULayerNames)
    listGPULayerTime = []
    if GPULayerTime != None:
        for tmpGPULayerNames in listGPULayerNames:
            listGPULayerTime.append(GPULayerTime*len(tmpGPULayerNames)/countLayer)

    return listGPULayerNames, listGPULayerTime

# 修补未知维度 并 保存输出到文件
def get_all_output(NetworkMap, dictInputTensor = {}, dictTensorShape = {}):
    ort_session = onnxruntime.InferenceSession(NetworkMap.all_output_file_dir, providers=['CPUExecutionProvider'])

    # 初始化输入 tensor
    ort_inputs_info = ort_session.get_inputs()
    ort_inputs = {}
    for i in range(len(ort_inputs_info)):
        tmp_name = ort_inputs_info[i].name
        tmp_elem_type_id = NetworkMap.dictInput[tmp_name].type.tensor_type.elem_type

        # elem_type: 1 --> float32
        if tmp_elem_type_id == 1:
            elem_value_type = np.float32

        # elem_type: 2 --> uint8
        elif tmp_elem_type_id == 2:
            elem_value_type = np.uint8

        # elem_type: 3 --> int8
        elif tmp_elem_type_id == 3:
            elem_value_type = np.int8

        # elem_type: 4 --> uint16
        elif tmp_elem_type_id == 4:
            elem_value_type = np.uint16

        # elem_type: 5 --> int16
        elif tmp_elem_type_id == 5:
            elem_value_type = np.int16

        # elem_type: 6 --> int32
        elif tmp_elem_type_id == 6:
            elem_value_type = np.int32

        # elem_type: 7 --> int64
        elif tmp_elem_type_id == 7:
            elem_value_type = np.int64

        # elem_type: 8 --> string
        # elem_type: 9 --> boolean
        elif tmp_elem_type_id == 9:
            elem_value_type = np.boolean

        # elem_type: 10 --> float16
        elif tmp_elem_type_id == 10:
            elem_value_type = np.float16

        # elem_type: 11 --> float64
        elif tmp_elem_type_id == 11:
            elem_value_type = np.float64

        # elem_type: 12 --> uint32
        elif tmp_elem_type_id == 12:
            elem_value_type = np.uint32

        # elem_type: 14 --> uint64
        elif tmp_elem_type_id == 14:
            elem_value_type = np.uint64

        # elem_type: 15 --> complex128
        elif tmp_elem_type_id == 15:
            elem_value_type = np.complex128

        # elem_type: 16 --> bfloat16
        elif tmp_elem_type_id == 16:
            elem_value_type = np.float16

        list_dim = [tmp_dim.dim_value for tmp_dim in NetworkMap.dictInput[tmp_name].type.tensor_type.shape.dim]
        # dictInput[tmp_name].type.tensor_type.shape.dim[0].dim_value
        # list_dim = dictTensorShape[tmp_name]

        ort_inputs[tmp_name] = dictInputTensor[tmp_name].astype(elem_value_type)

    ort_output = ort_session.run(None, ort_inputs)
    ort_outputs_info = ort_session.get_outputs()

    dictActualOutput = {}
    for i in range(len(ort_output)):
        dictActualOutput[ort_outputs_info[i].name] = ort_output[i]

    # 修补未知维度 并 保存输出到文件
    for tmp_name, tmp_output in dictActualOutput.items():
        if tmp_name in NetworkMap.dictOutput.keys():
            tmp_shape = tmp_output.shape
            OutputTensor =  NetworkMap.dictOutput[tmp_name]
            print("get_all_output: output name = {}".format(tmp_name), flush=True)
            # print("get_all_output: type = {}".format(type(OutputTensor.type.tensor_type.shape.dim)))
            if len(OutputTensor.type.tensor_type.shape.dim) != len(tmp_shape):
                while len(OutputTensor.type.tensor_type.shape.dim) > 0:
                    OutputTensor.type.tensor_type.shape.dim.pop()
                for tmp_dim in tmp_shape:
                    OutputTensor.type.tensor_type.shape.dim.append(onnx.onnx_ml_pb2.TensorShapeProto.Dimension(dim_value = tmp_dim))
            else:
                for i in range(len(tmp_shape)):
                    if OutputTensor.type.tensor_type.shape.dim[i].dim_value < 1:
                        OutputTensor.type.tensor_type.shape.dim[i].dim_value = tmp_shape[i]

        if tmp_name in NetworkMap.dictValueInfo.keys():
            tmp_shape = tmp_output.shape
            OutputTensor =  NetworkMap.dictValueInfo[tmp_name]
            print("get_all_output: value_info name = {}".format(tmp_name), flush=True)
            # print("get_all_output: type = {}".format(type(OutputTensor.type.tensor_type.shape.dim)))
            if len(OutputTensor.type.tensor_type.shape.dim) != len(tmp_shape):
                while len(OutputTensor.type.tensor_type.shape.dim) > 0:
                    OutputTensor.type.tensor_type.shape.dim.pop()
                for tmp_dim in tmp_shape:
                    OutputTensor.type.tensor_type.shape.dim.append(onnx.onnx_ml_pb2.TensorShapeProto.Dimension(dim_value = tmp_dim))
            else:
                for i in range(len(tmp_shape)):
                    if OutputTensor.type.tensor_type.shape.dim[i].dim_value < 1:
                        OutputTensor.type.tensor_type.shape.dim[i].dim_value = tmp_shape[i]

        # 保存输出到文件
        data_name = get_legal_name(tmp_name, NetworkMap.useless_prefix)
        data_dir = os.path.join(NetworkMap.onnx_folder_dir, data_name+".dat")
        if not os.path.exists(data_dir):
            data_file = open(data_dir, "wb")
            pickle.dump(tmp_output, data_file)
            data_file.close()

def measure_engine_pipeline(OrinNXMeasuring, engine_pipeline, MeasurementDuration = 2):
    
    # 先运行 1 s
    # 再测量 MeasurementDuration s

    NumExec = 0

    # 先跑 warm_up, 之后再正式开始测量
    warm_up_time = 1.2
    warm_up_count = 500 # 400
    power_prev = 1e-5
    # 确定 warm_up 所需的执行次数
    for i in range(16):
        OrinNXMeasuring.start()
        engine_pipeline.run(warm_up_count)
        OrinNXMeasuring.stop()
        warm_up_count = min(int(10 * warm_up_count), int(warm_up_time / (OrinNXMeasuring.measurement_duration.value/warm_up_count)))
        print("measurement_duration = {:.4e}".format(OrinNXMeasuring.measurement_duration.value), flush=True)
        print("warm_up_count = {}".format(warm_up_count), flush=True)

        tmp_err = abs((power_prev - OrinNXMeasuring.power_vdd.value) / power_prev)
        power_prev = OrinNXMeasuring.power_vdd.value
        print("power_evariation: {:.4%}".format(tmp_err), flush=True)
        if i > 4 or tmp_err < 0.01:
            break


    NumPerSecond = int(1 / (warm_up_time / warm_up_count))
    NumExec = int(MeasurementDuration / (warm_up_time / warm_up_count))
    print("NumExec = {}".format(NumExec), flush=True)

    time_begin = time.time()
    OrinNXMeasuring.start()
    engine_pipeline.run(NumExec)
    OrinNXMeasuring.stop()
    time_end = time.time()

    time_elapsed = time_end - time_begin
    avgExeTime = time_elapsed / NumExec
    print("total execution time: {} s".format(OrinNXMeasuring.measurement_duration.value), flush=True)
    print("avgExeTime: {} s".format(OrinNXMeasuring.measurement_duration.value/NumExec), flush=True)
    print("total energy: {} J".format(OrinNXMeasuring.energy_vdd.value), flush=True)
    print("avgEnergy: {} J".format(OrinNXMeasuring.energy_vdd.value/NumExec), flush=True)

    return NumExec

def save_eng_perf_log_dir(eng_perf_log_dir, FusedNode, trt_dla_engine_dir, trt_gpu_engine_dir, useless_prefix):

    isUseDLA = FusedNode.isCanUseDLA
    isUseGPU = FusedNode.isCanUseGPU
    tmpStrAccu = ""
    SubModelName = FusedNode.getName(useless_prefix)
    tmpStr = "\nsub_model: {}\n".format(SubModelName)
    print("{}".format(tmpStr), end="", flush=True)
    tmpStrAccu += tmpStr
    if FusedNode.isCanUseDLA == True:
        print("dla_energy = {} J".format(FusedNode.dla_profiling.avg_energy), flush=True)
        print("dla_time = {} s".format(FusedNode.dla_profiling.avg_exe_time), flush=True)
        dla_qps = 0.0
        if FusedNode.dla_profiling.avg_exe_time > 0:
            dla_qps = 1/FusedNode.dla_profiling.avg_exe_time
        print("dla_qps = {} q/s".format(dla_qps), flush=True)

        ModelName = os.path.split(trt_dla_engine_dir)[1]
        ModelName = os.path.splitext(ModelName)[0]
        if isinstance(FusedNode, network_type.fused_node):
            tmpStr = "sub_model: {}\n".format(ModelName)
        else:
            tmpStr = ModelName + ":\n"
        tmpStrAccu += tmpStr
        tmpStr = "dla_energy: {:.4e} J; dla_time: {:.4e} s; dla_power: {:.4e} W; dla_qps: {:.2f} q/s; dla_input_time: {:.4e} s; dla_output_time: {:.4e} s\n".format(FusedNode.dla_profiling.avg_energy, FusedNode.dla_profiling.avg_exe_time, FusedNode.dla_profiling.avg_power, dla_qps, FusedNode.dla_profiling.avg_input_time, FusedNode.dla_profiling.avg_output_time)
        tmpStrAccu += tmpStr

    if FusedNode.isCanUseGPU == True:
        print("gpu_energy = {} J".format(FusedNode.gpu_profiling.avg_energy), flush=True)
        print("gpu_time = {} s".format(FusedNode.gpu_profiling.avg_exe_time), flush=True)
        gpu_qps = 0.0
        if FusedNode.gpu_profiling.avg_exe_time > 0:
            gpu_qps = 1/FusedNode.gpu_profiling.avg_exe_time
        print("gpu_qps = {} q/s".format(gpu_qps), flush=True)

        ModelName = os.path.split(trt_gpu_engine_dir)[1]
        ModelName = os.path.splitext(ModelName)[0]
        if isinstance(FusedNode, network_type.fused_node):
            tmpStr = "sub_model: {}\n".format(ModelName)
        else:
            tmpStr = ModelName + ":\n"
        tmpStrAccu += tmpStr
        tmpStr = "gpu_energy: {:.4e} J; gpu_time: {:.4e} s; gpu_power: {:.4e} W; gpu_qps: {:.2f} q/s; gpu_input_time: {:.4e} s; gpu_output_time: {:.4e} s\n".format(FusedNode.gpu_profiling.avg_energy, FusedNode.gpu_profiling.avg_exe_time, FusedNode.gpu_profiling.avg_power, gpu_qps, FusedNode.gpu_profiling.avg_input_time, FusedNode.gpu_profiling.avg_output_time)
        tmpStrAccu += tmpStr

    if isUseDLA == True and isUseGPU == True and \
        FusedNode.gpu_profiling.avg_energy > FusedNode.dla_profiling.avg_energy and FusedNode.gpu_profiling.avg_energy > 0:
        
        tmpStr = "DLA 节能 {:.4%}\n".format((FusedNode.gpu_profiling.avg_energy - FusedNode.dla_profiling.avg_energy)/FusedNode.gpu_profiling.avg_energy)
        print("{}".format(tmpStr), end="", flush=True)
        tmpStrAccu += tmpStr

    if FusedNode.dla_profiling.avg_power > 0 or FusedNode.gpu_profiling.avg_power > 0:
        if not os.path.exists(eng_perf_log_dir):
            print(eng_perf_log_dir, flush=True)
            with open(eng_perf_log_dir, "w") as eng_perf_file:
                eng_perf_file.write(tmpStrAccu)

    return isUseDLA, isUseGPU, tmpStrAccu

def measure_save_network_eng_perf(eng_perf_log_dir, FusedNode, idle_power, useless_prefix, OrinNXMeasuring, trt_dla_engine_dir, trt_dla_perf_log_dir, trt_gpu_engine_dir, trt_gpu_perf_log_dir, MeasurementDuration = 2, use_managed_memory = False, dictInputTensor = {}, dictTensorShape = {}):

    isUseDLA, isUseGPU = False, False

    tmpStrAccu = ""
    if isinstance(FusedNode, network_type.fused_node):
        SubModelName = FusedNode.getName(useless_prefix)
        tmpStr = "\nsub_model: {}\n".format(SubModelName)
        print("{}".format(tmpStr), end="", flush=True)
        tmpStrAccu += tmpStr
    else:
        tmpStr = FusedNode.onnx_file_name + ":\n"
        print("\n{}".format(tmpStr), end="", flush=True)
        tmpStrAccu += tmpStr

    if os.path.exists(eng_perf_log_dir):
        # 从文件读取 能耗/性能
        isUseDLA, isUseGPU = extractEngPerf(FusedNode, eng_perf_log_dir, idle_power)

        print("dla_energy = {} J".format(FusedNode.dla_profiling.avg_energy), flush=True)
        print("dla_time = {} s".format(FusedNode.dla_profiling.avg_exe_time), flush=True)
        dla_qps = 0.0
        if FusedNode.dla_profiling.avg_exe_time > 0:
            dla_qps = 1/FusedNode.dla_profiling.avg_exe_time
        print("dla_qps = {} q/s".format(dla_qps), flush=True)

        if FusedNode.dla_profiling.avg_energy > 0:
            ModelName = os.path.split(trt_dla_engine_dir)[1]
            ModelName = os.path.splitext(ModelName)[0]
            if isinstance(FusedNode, network_type.fused_node):
                tmpStr = "sub_model: {}\n".format(ModelName)
            else:
                tmpStr = ModelName + ":\n"
            tmpStrAccu += tmpStr
            tmpStr = "dla_energy: {:.4e} J; dla_time: {:.4e} s; dla_power: {:.4e} W; dla_qps: {:.2f} q/s; dla_input_time: {:.4e} s; dla_output_time: {:.4e} s\n".format(FusedNode.dla_profiling.avg_energy, FusedNode.dla_profiling.avg_exe_time, FusedNode.dla_profiling.avg_power, dla_qps, FusedNode.dla_profiling.avg_input_time, FusedNode.dla_profiling.avg_output_time)
            tmpStrAccu += tmpStr

        print("gpu_energy = {} J".format(FusedNode.gpu_profiling.avg_energy), flush=True)
        print("gpu_time = {} s".format(FusedNode.gpu_profiling.avg_exe_time), flush=True)
        gpu_qps = 0.0
        if FusedNode.gpu_profiling.avg_exe_time > 0:
            gpu_qps = 1/FusedNode.gpu_profiling.avg_exe_time
        print("gpu_qps = {} q/s".format(gpu_qps), flush=True)

        if FusedNode.gpu_profiling.avg_energy > 0:
            ModelName = os.path.split(trt_gpu_engine_dir)[1]
            ModelName = os.path.splitext(ModelName)[0]
            if isinstance(FusedNode, network_type.fused_node):
                tmpStr = "sub_model: {}\n".format(ModelName)
            else:
                tmpStr = ModelName + ":\n"
            tmpStrAccu += tmpStr
            tmpStr = "gpu_energy: {:.4e} J; gpu_time: {:.4e} s; gpu_power: {:.4e} W; gpu_qps: {:.2f} q/s; gpu_input_time: {:.4e} s; gpu_output_time: {:.4e} s\n".format(FusedNode.gpu_profiling.avg_energy, FusedNode.gpu_profiling.avg_exe_time, FusedNode.gpu_profiling.avg_power, gpu_qps, FusedNode.gpu_profiling.avg_input_time, FusedNode.gpu_profiling.avg_output_time)
            tmpStrAccu += tmpStr

        if FusedNode.isCanUseDLA == True and FusedNode.isCanUseGPU == True and \
            FusedNode.gpu_profiling.avg_energy > FusedNode.dla_profiling.avg_energy and FusedNode.gpu_profiling.avg_energy > 0:
            
            tmpStr = "DLA 节能 {:.4%}\n".format((FusedNode.gpu_profiling.avg_energy - FusedNode.dla_profiling.avg_energy)/FusedNode.gpu_profiling.avg_energy)
            print("{}".format(tmpStr), end="", flush=True)
            tmpStrAccu += tmpStr

    else:
        # 测量能耗/性能, 并写入文件

        # 测量子节点在 DLA 上的能耗/性能
        if (FusedNode.isCanUseDLA == True and os.path.exists(trt_dla_engine_dir)):
            measure_network_energy(OrinNXMeasuring, trt_dla_engine_dir, MeasurementDuration, use_managed_memory, dictInputTensor, dictTensorShape)

            avg_input_time, avg_exe_time, avg_output_time = 0, 0, 0
            if os.path.splitext(trt_dla_perf_log_dir)[-1] == ".log":
                avg_exe_time = getMeanLatency(trt_dla_perf_log_dir)
            elif os.path.splitext(trt_dla_perf_log_dir)[-1] == ".json":
                avg_input_time, avg_exe_time, avg_output_time = getAvgTimes(trt_dla_perf_log_dir)

            FusedNode.dla_profiling.avg_input_time = avg_input_time
            FusedNode.dla_profiling.avg_output_time = avg_output_time
            FusedNode.dla_profiling.avg_exe_time = avg_exe_time
            FusedNode.dla_profiling.avg_power = OrinNXMeasuring.power_vdd.value
            FusedNode.dla_profiling.avg_energy = FusedNode.dla_profiling.avg_exe_time * FusedNode.dla_profiling.avg_power

            FusedNode.dla_profiling.power_dynamic = FusedNode.dla_profiling.avg_power - idle_power
            FusedNode.dla_profiling.energy_dynamic = FusedNode.dla_profiling.power_dynamic * FusedNode.dla_profiling.avg_exe_time
            FusedNode.dla_profiling.energy_static = idle_power * FusedNode.dla_profiling.avg_exe_time

            print("dla_energy = {} J".format(FusedNode.dla_profiling.avg_energy), flush=True)
            print("dla_time = {} s".format(FusedNode.dla_profiling.avg_exe_time), flush=True)
            dla_qps = 0.0
            if FusedNode.dla_profiling.avg_exe_time > 0:
                dla_qps = 1/FusedNode.dla_profiling.avg_exe_time
            print("dla_qps = {} q/s".format(dla_qps), flush=True)

            ModelName = os.path.split(trt_dla_engine_dir)[1]
            ModelName = os.path.splitext(ModelName)[0]
            if isinstance(FusedNode, network_type.fused_node):
                tmpStr = "sub_model: {}\n".format(ModelName)
            else:
                tmpStr = ModelName + ":\n"
            tmpStrAccu += tmpStr
            tmpStr = "dla_energy: {:.4e} J; dla_time: {:.4e} s; dla_power: {:.4e} W; dla_qps: {:.2f} q/s; dla_input_time: {:.4e} s; dla_output_time: {:.4e} s\n".format(FusedNode.dla_profiling.avg_energy, FusedNode.dla_profiling.avg_exe_time, FusedNode.dla_profiling.avg_power, dla_qps, FusedNode.dla_profiling.avg_input_time, FusedNode.dla_profiling.avg_output_time)
            tmpStrAccu += tmpStr
            
            isUseDLA = True
        
        
        # 测量子节点在 GPU 上的能耗/性能
        if FusedNode.isCanUseGPU == True and os.path.exists(trt_gpu_engine_dir):
            measure_network_energy(OrinNXMeasuring, trt_gpu_engine_dir, MeasurementDuration, use_managed_memory, dictInputTensor, dictTensorShape)

            avg_input_time, avg_exe_time, avg_output_time = 0, 0, 0
            if os.path.splitext(trt_gpu_perf_log_dir)[-1] == ".log":
                avg_exe_time = getMeanLatency(trt_gpu_perf_log_dir)
            elif os.path.splitext(trt_gpu_perf_log_dir)[-1] == ".json":
                avg_input_time, avg_exe_time, avg_output_time = getAvgTimes(trt_gpu_perf_log_dir)

            FusedNode.gpu_profiling.avg_input_time = avg_input_time
            FusedNode.gpu_profiling.avg_output_time = avg_output_time
            FusedNode.gpu_profiling.avg_exe_time = avg_exe_time
            FusedNode.gpu_profiling.avg_power = OrinNXMeasuring.power_vdd.value
            FusedNode.gpu_profiling.avg_energy = FusedNode.gpu_profiling.avg_exe_time * FusedNode.gpu_profiling.avg_power

            FusedNode.gpu_profiling.power_dynamic = FusedNode.gpu_profiling.avg_power - idle_power
            FusedNode.gpu_profiling.energy_dynamic = FusedNode.gpu_profiling.power_dynamic * FusedNode.gpu_profiling.avg_exe_time
            FusedNode.gpu_profiling.energy_static = idle_power * FusedNode.gpu_profiling.avg_exe_time

            print("gpu_energy = {} J".format(FusedNode.gpu_profiling.avg_energy), flush=True)
            print("gpu_time = {} s".format(FusedNode.gpu_profiling.avg_exe_time), flush=True)
            gpu_qps = 0.0
            if FusedNode.gpu_profiling.avg_exe_time > 0:
                gpu_qps = 1/FusedNode.gpu_profiling.avg_exe_time
            print("gpu_qps = {} q/s".format(gpu_qps), flush=True)

            ModelName = os.path.split(trt_gpu_engine_dir)[1]
            ModelName = os.path.splitext(ModelName)[0]
            if isinstance(FusedNode, network_type.fused_node):
                tmpStr = "sub_model: {}\n".format(ModelName)
            else:
                tmpStr = ModelName + ":\n"
            tmpStrAccu += tmpStr
            tmpStr = "gpu_energy: {:.4e} J; gpu_time: {:.4e} s; gpu_power: {:.4e} W; gpu_qps: {:.2f} q/s; gpu_input_time: {:.4e} s; gpu_output_time: {:.4e} s\n".format(FusedNode.gpu_profiling.avg_energy, FusedNode.gpu_profiling.avg_exe_time, FusedNode.gpu_profiling.avg_power, gpu_qps, FusedNode.gpu_profiling.avg_input_time, FusedNode.gpu_profiling.avg_output_time)
            tmpStrAccu += tmpStr

            isUseGPU = True
        

        if isUseDLA == True and isUseGPU == True and \
            FusedNode.gpu_profiling.avg_energy > FusedNode.dla_profiling.avg_energy and FusedNode.gpu_profiling.avg_energy > 0:
            
            tmpStr = "DLA 节能 {:.4%}\n".format((FusedNode.gpu_profiling.avg_energy - FusedNode.dla_profiling.avg_energy)/FusedNode.gpu_profiling.avg_energy)
            print("{}".format(tmpStr), end="", flush=True)
            tmpStrAccu += tmpStr

        print("FusedNode.dla_profiling.avg_power = {:.4e} W".format(FusedNode.dla_profiling.avg_power), flush=True)
        print("FusedNode.gpu_profiling.avg_power = {:.4e} W".format(FusedNode.gpu_profiling.avg_power), flush=True)
        if FusedNode.dla_profiling.avg_power > 0 or FusedNode.gpu_profiling.avg_power > 0:
            print(eng_perf_log_dir, flush=True)
            with open(eng_perf_log_dir, "w") as eng_perf_file:
                eng_perf_file.write(tmpStrAccu)

    return isUseDLA, isUseGPU, tmpStrAccu

def measure_network_energy(OrinNXMeasuring, trt_engine_file_dir, MeasurementDuration = 2, use_managed_memory = False, dictInputTensor = {}, dictTensorShape = {}):
    print("measure_network_energy: in", flush=True)
    # print("measure_network_energy: dictTensorShape = {}".format(dictTensorShape))
    # print("dictInputTensor[\"input_1\"][0][0][0][0] = {}".format(dictInputTensor["input_1"][0][0][0][0]))
    # print("dictInputTensor[\"image_shape\"] = {}".format(dictInputTensor["image_shape"]))
    # 先运行 1 s
    # 再测量 MeasurementDuration s

    OrinNXMeasuring.init_data()
    NumExec = 0

    if not os.path.exists(trt_engine_file_dir):
        return
    
    # If a serialized engine exists, load it.
    print("\nReading tensorrt engine from file: {}".format(trt_engine_file_dir), flush=True)
    
    with open(trt_engine_file_dir, "rb") as f, \
        trt.Runtime(TRT_LOGGER_PROFILING) as runtime:
        runtime.DLA_core = 1
        trt_engine = runtime.deserialize_cuda_engine(f.read())

        print("deserialize_cuda_engine: done", flush=True)

        context = trt_engine.create_execution_context()

        print("create_execution_context: done", flush=True)
        
        # print("allocate buffers")
        inputs, outputs, bindings, stream = allocate_buffers(trt_engine, use_managed_memory, True, dictTensorShape)
        print("allocate_buffers: done", flush=True)

        for i in range(len(inputs)):
            if inputs[i].name in dictInputTensor:
                np.copyto(inputs[i].host, dictInputTensor[inputs[i].name]) # 将 numpy array 拷贝到 cuda 分配的锁页 numpy array / managed_memory
            else:
                # tmpInput = np.ones(inputs[0].host.shape, dtype=inputs[0].host.dtype)
                tmpInput = np.random.random(inputs[i].host.shape).astype(inputs[i].host.dtype) * 254 - 127
                np.copyto(inputs[i].host, tmpInput) # 将 numpy array 拷贝到 cuda 分配的锁页 numpy array / managed_memory
                # inputs[i].host[:] = tmpInput[:]

        # Transfer input data to device
        if use_managed_memory == False:
            for i in range(len(inputs)):
                cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
        stream.synchronize()

        # 先跑 warm_up, 之后再正式开始测量
        warm_up_time = 1.2
        warm_up_count = 1000 # 400
        power_prev = 1
        # 确定 warm_up 所需的执行次数
        for i in range(16):
            OrinNXMeasuring.start()
            for j in range(int(warm_up_count)):
                context.execute_async_v2(bindings, stream.handle, None)
            stream.synchronize()
            OrinNXMeasuring.stop()
            warm_up_count = min(int(10 * warm_up_count), int(warm_up_time / (OrinNXMeasuring.measurement_duration.value/warm_up_count)))
            print("measurement_duration = {:.4e}".format(OrinNXMeasuring.measurement_duration.value), flush=True)
            print("warm_up_count = {}".format(warm_up_count), flush=True)

            tmp_err = abs((power_prev - OrinNXMeasuring.power_vdd.value) / power_prev)
            power_prev = OrinNXMeasuring.power_vdd.value
            print("power_evariation: {:.4%}".format(tmp_err), flush=True)
            if i > 4 or tmp_err < 0.01:
                break
        # 至此 完成 warm_up

        # NumPerSecond = int(1 / (warm_up_time / warm_up_count))
        NumExec = int(MeasurementDuration / (warm_up_time / warm_up_count))
        print("NumExec = {}".format(NumExec), flush=True)

        time_begin = time.time()
        OrinNXMeasuring.start()
        for i in range(int(NumExec)):
            # Execute model
            context.execute_async_v2(bindings, stream.handle, None)
        stream.synchronize()
        OrinNXMeasuring.stop()
        time_end = time.time()

        # Transfer predictions back
        if use_managed_memory == False:
            for i in range(len(outputs)):
                cuda.memcpy_dtoh_async(outputs[i].host, outputs[i].device, stream)
        # Syncronize threads
        stream.synchronize()
        
        # print("one_layer_trt complete inference")

        time_elapsed = time_end - time_begin
        avgExeTime = time_elapsed / NumExec
        print("total execution time: {} s".format(OrinNXMeasuring.measurement_duration.value), flush=True)
        print("avgExeTime: {} s".format(OrinNXMeasuring.measurement_duration.value/NumExec), flush=True)
        print("qps: {} q/s".format(NumExec/OrinNXMeasuring.measurement_duration.value), flush=True)
        print("total energy: {} J".format(OrinNXMeasuring.energy_vdd.value), flush=True)
        print("avgPower: {} W".format(OrinNXMeasuring.power_vdd.value), flush=True)
        print("avgEnergy: {} J/q".format(OrinNXMeasuring.energy_vdd.value/NumExec), flush=True)
        # del bindings, inputs, outputs, stream, context, trt_engine

        # 手动删除变量 为 后续的 mapping 过程 节省内存
        del context, trt_engine, bindings, inputs, outputs, stream


def measure_network_energy_trtexec(OrinNXMeasuring, trt_engine_file_dir, MeasurementDuration = 2, use_managed_memory = False, dictInputTensor = {}, dictTensorShape = {}):

    print("measure_network_energy_trtexec: in", flush=True)
    # print("measure_network_energy: dictTensorShape = {}".format(dictTensorShape))
    # print("dictInputTensor[\"input_1\"][0][0][0][0] = {}".format(dictInputTensor["input_1"][0][0][0][0]))
    # print("dictInputTensor[\"image_shape\"] = {}".format(dictInputTensor["image_shape"]))
    # 先运行 1 s
    # 再测量 MeasurementDuration s

    OrinNXMeasuring.init_data()
    if not os.path.exists(trt_engine_file_dir):
        return

    # 处理给定输入数据
    tmpStr = os.path.split(trt_engine_file_dir)
    folder_dir = tmpStr[0]
    strInput = ""
    if len(dictInputTensor) > 0:
        for data_name, data in dictInputTensor.items():
            data_name = get_legal_name(data_name)
            data_dir = os.path.join(folder_dir, data_name+".pkl")
            if not os.path.exists(data_dir):
                data_file = open(data_dir, "wb")
                pickle.dump(data, data_file)
                data_file.close()
            strInput += "\"" + data_name + "\":" + data_dir + ","
        if len(strInput) > 0:
            strInput = strInput[:-1]
            strInput = " --loadInputs=" + strInput
    
    strCMD = "trtexec"
    strCMD += " --loadEngine=\"" + trt_engine_file_dir + "\""
    strCMD += strInput
    strCMD += " --useDLACore=1" # --warmUp=0" # --useManagedMemory"
    strDrop = " >/dev/null 2>&1"

    # 先跑 warm_up, 之后再正式开始测量
    warm_up_time = 1.2
    strWarmUpCMD = strCMD + " --duration=" + str(warm_up_time) + strDrop
    power_prev = 10000
    print("{}".format(strWarmUpCMD), flush=True)
    # 确定 warm_up 所需的执行次数
    for i in range(16):
        OrinNXMeasuring.start()
        os.system(strWarmUpCMD)
        OrinNXMeasuring.stop()
        tmp_err = abs((power_prev - OrinNXMeasuring.power_vdd.value) / (power_prev + OrinNXMeasuring.power_vdd.value))
        power_prev = OrinNXMeasuring.power_vdd.value
        print("power_evariation: {:.4%}".format(tmp_err), flush=True)
        if i > 6 or tmp_err < 0.01:
            break
    # 至此 完成 warm_up

    # 使用 trtexec 进行测量
    skip_time = 1.0
    strCMD += " --duration=" + str(MeasurementDuration+skip_time) + strDrop #  --iterations=20
    print("{}".format(strCMD), flush=True)
    OrinNXMeasuring.start(skip_time)
    os.system(strCMD)
    OrinNXMeasuring.stop()
    print("trtexec measurement:", flush=True)
    print("total execution time: {} s".format(OrinNXMeasuring.measurement_duration.value), flush=True)
    print("total energy: {} J".format(OrinNXMeasuring.energy_vdd.value), flush=True)
    print("avgPower: {} W".format(OrinNXMeasuring.power_vdd.value), flush=True)

def measure_idle_power():
    idle_power = -1.0
    time.sleep(2)
    with orin_nx_measuring(0.05) as OrinNXMeasuring:
        OrinNXMeasuring.start()
        time.sleep(6)
        OrinNXMeasuring.stop()
        idle_power = OrinNXMeasuring.power_vdd.value
    return idle_power

def getMeanLatency(trt_log_file_dir):
    mean_latency = -1e6
    with open(trt_log_file_dir, "r") as log:
        tmp_flag = "Throughput:"
        for line in log.readlines():
            if tmp_flag == "Throughput:":
                pos = line.find(tmp_flag)
                if pos < 0:
                    continue
                else:
                    tmp_flag = "GPU Compute Time:"

            elif tmp_flag == "GPU Compute Time:":
                pos = line.find(tmp_flag)
                if pos < 0:
                    continue
                else:
                    tmp_flag = "mean = "
                    pos0 = line.find(tmp_flag)
                    if pos0 < 0:
                        break
                    else:
                        pos1 = line.find(" ms", pos0)
                        sub_str = line[pos0:pos1]
                        sub_str = sub_str.strip("mean = ")
                        mean_latency = float(sub_str) / 1000

    return mean_latency

def getAvgTimes(trt_json_file_dir):
    avg_input_time, avg_exe_time, avg_output_time = 0, 0, 0

    with open(trt_json_file_dir, "r") as file:
        listJson = json.load(file)
        print("Average running time of fused NN layers on GPU: ", flush=True)

        count = listJson[0]["count"]
        for i in range(1, len(listJson), 1):
            dictTmp = listJson[i]
            names = dictTmp["name"]

            if names.find("Reformatting CopyNode for Input Tensor") >= 0:
                avg_input_time += dictTmp["averageMs"] / 1000
            elif names.find("Reformatting CopyNode for Output Tensor") >= 0:
                avg_output_time += dictTmp["averageMs"] / 1000
            else:
                avg_exe_time += dictTmp["averageMs"] / 1000

        # print("")
    # 至此, 获得了只使用 GPU 时, 节点融合情况

    return avg_input_time, avg_exe_time, avg_output_time

def extractEngPerf(FusedNode, sub_eng_perf_log_dir, idle_power):
    isUseDLA, isUseGPU = False, False

    with open(sub_eng_perf_log_dir, "r") as log:
        for line in log.readlines():
            
            str_pattern = "dla_energy: "
            pos0 = line.find(str_pattern)
            if pos0 >= 0:
                isUseDLA = True

                pos1 = line.find(" J", pos0)
                sub_str = line[pos0:pos1]
                sub_str = sub_str.strip(str_pattern)
                FusedNode.dla_profiling.avg_energy = float(sub_str)

                str_pattern = "dla_time: "
                pos0 = line.find(str_pattern, pos1)
                if pos0 > 0:
                    pos1 = line.find(" s", pos0)
                    sub_str = line[pos0:pos1]
                    sub_str = sub_str.strip(str_pattern)
                    FusedNode.dla_profiling.avg_exe_time = float(sub_str)

                str_pattern = "dla_power: "
                pos0 = line.find(str_pattern, pos1)
                if pos0 > 0:
                    pos1 = line.find(" W", pos0)
                    sub_str = line[pos0:pos1]
                    sub_str = sub_str.strip(str_pattern)
                    FusedNode.dla_profiling.avg_power = float(sub_str)

                str_pattern = "dla_input_time: "
                pos0 = line.find(str_pattern, pos1)
                if pos0 > 0:
                    pos1 = line.find(" s", pos0)
                    sub_str = line[pos0:pos1]
                    sub_str = sub_str.strip(str_pattern)
                    FusedNode.dla_profiling.avg_input_time = float(sub_str)

                str_pattern = "dla_output_time: "
                pos0 = line.find(str_pattern, pos1)
                if pos0 > 0:
                    pos1 = line.find(" s", pos0)
                    sub_str = line[pos0:pos1]
                    sub_str = sub_str.strip(str_pattern)
                    FusedNode.dla_profiling.avg_output_time = float(sub_str)

                continue

            str_pattern = "gpu_energy: "
            pos0 = line.find(str_pattern)
            if pos0 >= 0:
                isUseGPU = True

                pos1 = line.find(" J", pos0)
                sub_str = line[pos0:pos1]
                sub_str = sub_str.strip(str_pattern)
                FusedNode.gpu_profiling.avg_energy = float(sub_str)

                str_pattern = "gpu_time: "
                pos0 = line.find(str_pattern, pos1)
                if pos0 > 0:
                    pos1 = line.find(" s", pos0)
                    sub_str = line[pos0:pos1]
                    sub_str = sub_str.strip(str_pattern)
                    FusedNode.gpu_profiling.avg_exe_time = float(sub_str)

                str_pattern = "gpu_power: "
                pos0 = line.find(str_pattern, pos1)
                if pos0 > 0:
                    pos1 = line.find(" W", pos0)
                    sub_str = line[pos0:pos1]
                    sub_str = sub_str.strip(str_pattern)
                    FusedNode.gpu_profiling.avg_power = float(sub_str)

                str_pattern = "gpu_input_time: "
                pos0 = line.find(str_pattern, pos1)
                if pos0 > 0:
                    pos1 = line.find(" s", pos0)
                    sub_str = line[pos0:pos1]
                    sub_str = sub_str.strip(str_pattern)
                    FusedNode.gpu_profiling.avg_input_time = float(sub_str)

                str_pattern = "gpu_output_time: "
                pos0 = line.find(str_pattern, pos1)
                if pos0 > 0:
                    pos1 = line.find(" s", pos0)
                    sub_str = line[pos0:pos1]
                    sub_str = sub_str.strip(str_pattern)
                    FusedNode.gpu_profiling.avg_output_time = float(sub_str)

        FusedNode.dla_profiling.power_dynamic = FusedNode.dla_profiling.avg_power - idle_power
        FusedNode.dla_profiling.energy_dynamic = FusedNode.dla_profiling.power_dynamic * FusedNode.dla_profiling.avg_exe_time
        FusedNode.dla_profiling.energy_static = idle_power * FusedNode.dla_profiling.avg_exe_time

        FusedNode.gpu_profiling.power_dynamic = FusedNode.gpu_profiling.avg_power - idle_power
        FusedNode.gpu_profiling.energy_dynamic = FusedNode.gpu_profiling.power_dynamic * FusedNode.gpu_profiling.avg_exe_time
        FusedNode.gpu_profiling.energy_static = idle_power * FusedNode.gpu_profiling.avg_exe_time
        
    return isUseDLA, isUseGPU

def get_legal_name(name, useless_prefix=""):
    if len(useless_prefix) > 0:
        name = re.sub(useless_prefix, "", name)
    name = re.sub("/", "-", name)
    name = re.sub("\:", "--", name)

    return name