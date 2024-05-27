# coding=utf-8
##############################################################################################
# 输入测量得到的融合节点在 GPU/DLA 上的 性能/能耗
# 设计融合节点到 DLA/GPU 的映射算法
##############################################################################################
import os, json, functools, re, copy
import numpy as np
import onnx
from network_type import network_map, SUB_GRAPH, createMergedSubGraph, canUseDLA_GPU
import tensorrt as trt
from profiling import get_legal_name
from trt_engine_memory import get_engine, allocate_buffers, allocate_input_buffers, allocate_output_buffers
from runtime import ENGINE_STREAM, ENGINE_PIPELINE
TRT_LOGGER = trt.Logger()

def map_network_manual1(NetworkMap, my_calibrator):

    print("map_network_manual1: in", flush=True)

    mapping_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_mapping_manual.json")

    listEngineConfig = loadMappingConfig(mapping_file_dir)

    generateMultiStageEngines(NetworkMap, listEngineConfig, my_calibrator)

    return

def map_network_manual(NetworkMap, my_calibrator):

    print("map_network_manual: in", flush=True)

    dictNodeName_Device = {}
    dictNodeName_DeviceName = {}

    mapping_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_mapping_manual.json")
    # mapping_file_dir = "/home/wfr/work/DLA/onnx_model_zoo/yolov4_1/yolov4_mapping_manual.json"
    if os.path.exists(mapping_file_dir):
        with open(mapping_file_dir, "r") as file:
            mapping_config = json.load(file)

            for key, value in mapping_config.items():
                if value == "GPU":
                    continue
                dictNodeName_Device[key] = trt.DeviceType.DLA
                dictNodeName_DeviceName[key] = "DLA"

    # dictNodeName_Device["vgg0_conv0_fwd"] = trt.DeviceType.DLA
    # dictNodeName_DeviceName["vgg0_conv0_fwd"] = "DLA"

    
    # 2. 根据融合节点到 DLA/GPU 的映射方案
    onnx_file_dir = NetworkMap.infered_file_dir
    trt_engine_gpu_dla_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_gpu_dla_manual.trt")
    BatchSize = 1
    default_device_type = trt.DeviceType.GPU

    trt_engine_gpu_dla0 = get_engine(onnx_file_dir, trt_engine_gpu_dla_file_dir, BatchSize, default_device_type, dictNodeName_Device, 0, my_calibrator)
    print("get_engine DLA0 done", flush=True)

    trt_engine_gpu_dla1 = get_engine(onnx_file_dir, trt_engine_gpu_dla_file_dir, BatchSize, default_device_type, dictNodeName_Device, 1, my_calibrator)
    print("get_engine DLA1 done", flush=True)

    # 2(GPU-DLA)
    list_pipeline_stage_engine = \
    [[ENGINE_STREAM(trt_engine_gpu_dla0),ENGINE_STREAM(trt_engine_gpu_dla1)]]
    # list_pipeline_stage_engine = [[ENGINE_STREAM(trt_engine_gpu_dla0)]]
    print("map_network_manual: ENGINE_STREAM complete", flush=True)

    RingLen = 2
    engine_pipeline = ENGINE_PIPELINE(list_pipeline_stage_engine, RingLen)
    print("map_network_manual: ENGINE_PIPELINE complete", flush=True)

    return engine_pipeline


class ENGINE_CONFIG:
    def __init__(self, device, listLayerID, NumStreams = 0) -> None:
        self.device = device # "DLA" / "GPU"
        self.listLayerID = sorted(listLayerID)
        if NumStreams == 0:
            if self.device == "DLA":
                self.NumStreams = 2
            else:
                self.NumStreams = 1
        else:
            self.NumStreams = NumStreams

def loadMappingConfig(MappingConfigDir):
    listEngineConfig = []

    if os.path.exists(MappingConfigDir):
        with open(MappingConfigDir, "r") as file:
            listDump = json.load(file)

    for Dump in listDump:
        tmpEngineConfig = ENGINE_CONFIG(Dump[0], Dump[1], Dump[2])
        listEngineConfig.append(tmpEngineConfig)
        print("{}: {}, {}\n".format(Dump[0], Dump[1], Dump[2]))

    # exit(0)
    return listEngineConfig

def dumpMappingConfig(MappingConfigDir, listEngineConfig):
    listDump = []
    for EngineConfig in listEngineConfig:
        Dump = [] # ["DLA", [0,30], 4]
        Dump.append(EngineConfig.device)
        Dump.append(EngineConfig.listLayerID)
        Dump.append(EngineConfig.NumStreams)
        listDump.append(Dump)

    mapping_scheme = json.dumps(listDump, indent=4)
    with open(MappingConfigDir, "w") as file:
        file.write(mapping_scheme)

def map_network_subgraph(NetworkMap, my_calibrator):
    io_energy_discount = 1.0
    ratio_time_dla = 1.0
    ratio_bubble_time = 0.5
    sumTimeGPU = NetworkMap.sumTimeGPU # 融合节点都在 GPU 上运行的总时间
    sumEnergyGPU = NetworkMap.sumEnergyGPU
    sumTimeOnlyGPU = 0.0
    countOnlyGPU = 0
    sumTimeDLA = 0.0
    tmpSumTimeGPU = 0.0 # 支持 GPU 和 DLA 的节点 在 GPU 上的 运行时间
    print("\nmap_network_subgraph: in")
    print("io_energy_discount = {}".format(io_energy_discount))
    print("ratio_time_dla = {}".format(ratio_time_dla))
    print("ratio_bubble_time = {}".format(ratio_bubble_time))

    dictNodeName_Device = {}
    dictNodeName_DeviceName = {}
    dictID_FusedNode = {}
    listFusedNode = NetworkMap.getListFusedNode()
    for FusedNode in listFusedNode:
        if FusedNode.isCanUseDLA == True and FusedNode.isCanUseGPU == True \
            and FusedNode.gpu_profiling.avg_exe_time > 0.0 and FusedNode.dla_profiling.avg_exe_time > 0.0:
            sumTimeDLA += FusedNode.dla_profiling.avg_exe_time
            # tmpSumTimeGPU += FusedNode.gpu_profiling.avg_exe_time
        else:
            sumTimeOnlyGPU += FusedNode.gpu_profiling.avg_exe_time
            countOnlyGPU += 1

        dictID_FusedNode[FusedNode.id] = FusedNode

    sumTimeOnlyGPU *= NetworkMap.NumDLA
    sumTimeDLA *= NetworkMap.NumGPU
    sumTimeGPU *= NetworkMap.NumDLA
    sumEnergyGPU *= NetworkMap.NumDLA
    print("sumTimeDLA = {:.4e}".format(sumTimeDLA), flush=True)
    print("sumTimeGPU = {:.4e}".format(sumTimeGPU), flush=True)
    print("sumTimeOnlyGPU = {:.4e}".format(sumTimeOnlyGPU), flush=True)
    print("sumEnergyGPU = {:.4e}".format(sumEnergyGPU), flush=True)

    listSubGraph = getSubGraph(NetworkMap, dictID_FusedNode, sumTimeGPU, sumEnergyGPU, io_energy_discount, ratio_time_dla, 1.0)
    # listSubGraph = getSubGraph(NetworkMap, dictID_FusedNode, sumTimeGPU, sumEnergyGPU, io_energy_discount, ratio_time_dla, ratio_bubble_time)

    # 至此已经提取出 有潜力的子图: SingleIOMap / Sequence 两种类型
    # 接下来就要从中优选节能潜力更大子图的映射到 DLA
    # 对于 Sequence 类型, 还需要滑动窗口遍历

    print("")
    for SubGraph in listSubGraph:
        print("SubGraph: [", end="", flush=True)
        for FusedNode in SubGraph.list_fused_node:
            print("{}, ".format(FusedNode.getName(NetworkMap.useless_prefix)), end="", flush=True)
        
        print("]", flush=True)
    print("")

    # exit(0)

    sumFlexibleTime = sumTimeGPU
    sumBubbleTime = 0.0
    avgBubbleTime = 0.0
    countBubble = 0
    sumGPUTime = sumTimeOnlyGPU
    avgGPUTime = 0.0
    countGPUTime = countOnlyGPU
    maxBubbleTime = 0.0
    maxGPUTime = 0.0
    for i in range(len(listSubGraph)):
        tmpSubGraph = listSubGraph[i]
        tmpEnergySaving, tmpMinTime, remainingFlexibleTime, tmpMinEnergy = tmpSubGraph.getEnergyTimeWithDLA(1.0, sumTimeGPU, sumTimeGPU, sumEnergyGPU)
        # tmpEnergySaving, tmpMinTime, remainingFlexibleTime = tmpSubGraph.getBestEngPerfAllSubSeq(ratio_bubble_time, sumTimeGPU, sumTimeGPU)

        if tmpEnergySaving > 0.0 and tmpMinTime < sumTimeGPU:
            sumFlexibleTime -= tmpSubGraph.gpu_time
            sumBubbleTime += tmpSubGraph.bubble_time
            maxBubbleTime = max(maxBubbleTime, tmpSubGraph.bubble_time)
            countBubble += 1
        else:
            sumGPUTime += tmpSubGraph.gpu_time
            maxGPUTime = max(maxGPUTime, tmpSubGraph.gpu_time)
            countGPUTime += 1

    print("\n", flush=True)
    print("Init sumFlexibleTime = {:.4e}".format(sumFlexibleTime), flush=True)
    print("Init sumBubbleTime = {:.4e}".format(sumBubbleTime), flush=True)
    if countBubble >= 1:
        avgBubbleTime = sumBubbleTime/countBubble
    print("Init avgBubbleTime = {:.4e}".format(avgBubbleTime), flush=True)
    print("Init maxBubbleTime = {:.4e}".format(maxBubbleTime), flush=True)
    

    print("Init sumGPUTime = {:.4e}".format(sumGPUTime), flush=True)
    if countGPUTime >= 1:
        avgGPUTime = sumGPUTime/countGPUTime
    print("Init avgGPUTime = {:.4e}".format(avgGPUTime), flush=True)
    print("Init maxGPUTime = {:.4e}".format(maxGPUTime), flush=True)
    
    print("\n", flush=True)
    # 平均 gpu_time 和 bubble_time
    x = avgGPUTime / avgBubbleTime
    # x = sumGPUTime / sumBubbleTime
    # k = 0.5888441978311874
    # b = -0.1109856949399129
    # k = 0.60
    # b = -0.11
    # k = 1.3206777053411285
    # b = -0.4345452112873241
    # k = 1.4545700611572643
    # b = -0.4845336513298597
    k = 1.289777741647531
    b = -0.42203842968984234
    ratio_bubble_time = k*(x**(1/3)) + b
    ratio_bubble_time = max(ratio_bubble_time, 0.1)
    ratio_bubble_time = min(ratio_bubble_time, 0.9)
    
    # k = 0.6655772332194388
    # b = -0.20823923343339684
    # ratio_bubble_time = k*(x**(1/4)) + b
    ratio_bubble_time = 1.0
    print("ratio_bubble_time = {}".format(ratio_bubble_time), flush=True)

    listCandidateSubGraph = []
    sumFlexibleTime = sumTimeGPU
    # 各个子图中最佳窗口, SingleIOMap整体计算, Sequence找最佳窗口
    for i in range(len(listSubGraph)):
        SubGraph = listSubGraph[i]

        if SubGraph.str_type == "SingleIOMap":
            print("SubGraph:")
            print("first FusedNode: {}".format(SubGraph.list_fused_node[0].getName(NetworkMap.useless_prefix)))
            print("last FusedNode: {}".format(SubGraph.list_fused_node[-1].getName(NetworkMap.useless_prefix)))
            tmpEnergySaving, tmpMinTime, remainingFlexibleTime, tmpMinEnergy = SubGraph.getEnergyTimeWithDLA(ratio_bubble_time, sumTimeGPU, sumTimeGPU, sumEnergyGPU)
            listCandidateSubGraph.append(SubGraph)
            print("tmpEnergySaving = {:.4e} J".format(tmpEnergySaving))
            print("tmpMinTime = {:.4e} s\n".format(tmpMinTime), flush=True)
            continue

        minTime = sumTimeGPU
        tmpMinTime = sumTimeGPU
        minEnergy = sumEnergyGPU
        tmpMinEnergy = sumEnergyGPU
        minEDP = minEnergy * minTime
        maxEnergySavingInSubGraph = -1e9
        minTimeInSubGraph = sumTimeGPU
        BestWindowInSubGraph = None
        for j in range(len(SubGraph.list_fused_node)): # 从 j 位置起始的 各个窗口
            listWindow = []
            for k in range(len(SubGraph.list_fused_node)-j): # k+1 长度的窗口
                tmpSubGraph = SUB_GRAPH(SubGraph.str_type, SubGraph.list_fused_node[j:j+k+1], NetworkMap, io_energy_discount, ratio_time_dla, False)
                listWindow.append(tmpSubGraph)

                tmpEnergySaving, tmpMinTime, remainingFlexibleTime, tmpMinEnergy = tmpSubGraph.getEnergyTimeWithDLA(ratio_bubble_time, sumTimeGPU, sumTimeGPU, sumEnergyGPU)
                tmpMinEDP = tmpMinEnergy * tmpMinTime
                
                if tmpMinEDP < minEDP and tmpMinTime < sumTimeGPU:
                    minEDP = tmpMinEDP
                    minEnergy = tmpMinEnergy
                    minTime = tmpMinTime
                    maxEnergySavingInSubGraph = tmpEnergySaving
                    minTimeInSubGraph = tmpMinTime
                    BestWindowInSubGraph = tmpSubGraph

                # if tmpEnergySaving > maxEnergySavingInSubGraph \
                #     and tmpMinTime <= minTimeInSubGraph:
                #     maxEnergySavingInSubGraph = tmpEnergySaving
                #     minTimeInSubGraph = tmpMinTime
                #     BestWindowInSubGraph = tmpSubGraph
            # end for
        # end for
        if BestWindowInSubGraph != None:
            listCandidateSubGraph.append(BestWindowInSubGraph)
            print("\nBestWindowInSubGraph:")
            print("first FusedNode: {}".format(BestWindowInSubGraph.list_fused_node[0].getName(NetworkMap.useless_prefix)))
            print("last FusedNode: {}".format(BestWindowInSubGraph.list_fused_node[-1].getName(NetworkMap.useless_prefix)))
            print("minEnergy = {:.4e} J".format(minEnergy))
            print("minTime = {:.4e} s\n".format(minTime))
            print("minEDP = {:.4e} s\n".format(minEDP))
            print("{}: {}".format("list_network_node_id", BestWindowInSubGraph.list_network_node_id))
        else:
            print("\nBestWindowInSubGraph: None\n")
    # end for
    print("build windows done\n", flush=True)

    # listCandidateSubGraph.sort(key=lambda x:x.energy_saving_with_dla, reverse=True)

    # print("listCandidateSubGraph:")
    # for tmpSubGraph in listCandidateSubGraph:
    #     print("{}: {}".format("list_network_node_id", tmpSubGraph.list_network_node_id))
    # print("")

    # # 这里尝试进行子图融合, 考虑能效: 如果有EDP节省且性能不弱于GPU则进行融合
    # listCandidateSubGraph = mergeSubGraph(listCandidateSubGraph, NetworkMap, io_energy_discount, ratio_time_dla, True, ratio_bubble_time, sumTimeGPU, sumTimeGPU, sumTimeGPU, sumEnergyGPU)

    # print("listCandidateSubGraph:")
    # for tmpSubGraph in listCandidateSubGraph:
    #     print("{}: {}".format("list_network_node_id", tmpSubGraph.list_network_node_id))
    # print("")

    # # 在 Orin NX 上 如果两个 DLA 同时使用, 且每个 DLA 上 SubGraph 数量相同
    # # 那么一个 DLA 上 MaxSubGraphsPerDLA 就是 10
    # # 对 listCandidateSubGraph 排序, 按找节能多少降序排列
    # listCandidateSubGraph.sort(key=lambda x:x.energy_saving_with_dla, reverse=True)

    print("listCandidateSubGraph:")
    for tmpSubGraph in listCandidateSubGraph:
        print("{}: {}".format("list_network_node_id", tmpSubGraph.list_network_node_id))
    print("")


    remainingFlexibleTime = sumTimeGPU
    sumFlexibleTime = sumTimeGPU
    sumBubbleTime = 0.0
    minTime = sumTimeGPU
    tmpMinTime = sumTimeGPU
    minEnergy = sumEnergyGPU
    tmpMinEnergy = sumEnergyGPU
    minEDP = minEnergy * minTime
    # k = 0.22
    # k = 0.7
    # b = -0.14478
    # k = 1.4545700611572643
    k = 0.9
    b = -0.4845336513298597
    # k = 1.0
    # b = 0.0
    listEngineConfig = []
    listSelectedSubGraph = []
    listResult = []
    # 使用变长滑动窗口找最优配置
    for i in range(len(listCandidateSubGraph)): # 从 i 位置起始的 各个窗口
        for j in range(i+1, len(listCandidateSubGraph)+1): # j+1 长度的窗口
            print("sliding window: listCandidateSubGraph[{}:{}]".format(i,j))
            subListCandidateSubGraph = listCandidateSubGraph[i:j]
            tmp_ratio_bubble_time = 1.0

            tmplistSubGraph = mergeSubGraph(subListCandidateSubGraph, NetworkMap, io_energy_discount, ratio_time_dla, False, ratio_bubble_time, sumTimeGPU, sumTimeGPU, sumTimeGPU, sumEnergyGPU)
            # 子图数量不能超过上限
            print("sliding window: number of subgraphs = {}".format(len(tmplistSubGraph)))
            if len(tmplistSubGraph) > NetworkMap.MaxSubGraphsPerDLA:
                break
            
            tmpSubGraph = createMergedSubGraph(subListCandidateSubGraph)

            tmpEnergySaving, tmpMinTime, remainingFlexibleTime, tmpMinEnergy = tmpSubGraph.getEnergyTimeWithDLA(tmp_ratio_bubble_time, sumTimeGPU, sumTimeGPU, sumEnergyGPU)

            tmpMinEDP = tmpMinEnergy * tmpMinTime

            print("tmpMinEnergy = {:.4e}".format(tmpMinEnergy))
            print("tmpMinTime = {:.4e}".format(tmpMinTime))
            print("tmpMinEDP = {:.4e}".format(tmpMinEDP))

            print("minEnergy = {:.4e}".format(minEnergy))
            print("minTime = {:.4e}".format(minTime))
            print("minEDP = {:.4e}".format(minEDP))
            print("sumTimeGPU = {:.4e}".format(sumTimeGPU), flush=True)

            if (tmpMinEDP < minEDP or tmpMinEnergy < minEnergy) and tmpMinTime < sumTimeGPU:
            # if (tmpMinEDP < minEDP or tmpMinTime < minTime) and tmpMinTime < sumTimeGPU:
            # if tmpMinEDP < minEDP and tmpMinTime < sumTimeGPU:
                listSelectedSubGraph = copy.copy(subListCandidateSubGraph)
                minEDP = tmpMinEDP
                minEnergy = tmpMinEnergy
                minTime = tmpMinTime
            # end if
        # end for
    # end for

    # listSelectedSubGraph = listResult
    print("listSelectedSubGraph:")
    for tmpSubGraph in listSelectedSubGraph:
        print("{}: {}".format("list_network_node_id", tmpSubGraph.list_network_node_id))
    print("")

    # 这里尝试进行子图融合, 不考虑能效
    listSelectedSubGraph = mergeSubGraph(listSelectedSubGraph, NetworkMap, io_energy_discount, ratio_time_dla, False, ratio_bubble_time, sumTimeGPU, sumTimeGPU, sumTimeGPU, sumEnergyGPU)

    print("listSelectedSubGraph:")
    for tmpSubGraph in listSelectedSubGraph:
        print("{}: {}".format("list_network_node_id", tmpSubGraph.list_network_node_id))
    print("")

    # 舍弃太小的 DLA子图
    i = 0
    while i < len(listSelectedSubGraph):
        if len(listSelectedSubGraph[i].list_network_node_id) <= 3:
            listSelectedSubGraph.pop(i)
        else:
            i += 1
    # end while

    print("listSelectedSubGraph:")
    for tmpSubGraph in listSelectedSubGraph:
        print("{}: {}".format("list_network_node_id", tmpSubGraph.list_network_node_id))
    print("")

    for SelectedSubGraph in listSelectedSubGraph:
        listLayerID = []
        for FusedNode in SelectedSubGraph.list_fused_node:
            for onnx_node in FusedNode.list_onnx_node:
                dictNodeName_Device[onnx_node.name] = trt.DeviceType.DLA
                dictNodeName_DeviceName[onnx_node.name] = "DLA"
                listLayerID.append(NetworkMap.dictNetworkNode[onnx_node.name].id)
        listLayerID.sort()
        EngineConfig = ENGINE_CONFIG("DLA", listLayerID)
        print("listLayerID: {}\n".format(listLayerID))
        listEngineConfig.append(EngineConfig)
    # 按照 层ID 升序排列
    listEngineConfig.sort(key=lambda x:x.listLayerID[0])

    # 补全配置: 之前的配置只包含映射到 DLA 的层, 这里补上映射到 GPU 的层
    idxPrev = None
    idxPost = 0
    listLayerIDDLA = []
    for EngineConfig in listEngineConfig:
        listLayerIDDLA += EngineConfig.listLayerID

    tmpSetLayerID = set(range(len(NetworkMap.dictNetworkNode)))
    listLayerIDGPU = list(tmpSetLayerID - set(listLayerIDDLA))
    listLayerIDGPU.sort()
    
    tmpListLayerID = []
    i = 0
    if listLayerIDGPU[0] == 0:
        for ID in listLayerIDGPU:
            if ID < listEngineConfig[0].listLayerID[0]:
                tmpListLayerID.append(ID)
                i += 1
            else:
                break
        EngineConfig = ENGINE_CONFIG("GPU", tmpListLayerID)
        listEngineConfig.insert(0, EngineConfig)
        tmpListLayerID = []
        idxPrev = 1
        idxPost = 2
    else:
        idxPrev = 0
        idxPost = 1
    while i < len(listLayerIDGPU):
        ID = listLayerIDGPU[i]
        if idxPost < len(listEngineConfig):
            if listEngineConfig[idxPrev].listLayerID[0] < ID and ID < listEngineConfig[idxPost].listLayerID[0]:
                tmpListLayerID.append(ID)
                i += 1
            else:
                if len(tmpListLayerID) > 0:
                    EngineConfig = ENGINE_CONFIG("GPU", tmpListLayerID)
                    listEngineConfig.insert(idxPost, EngineConfig)
                    idxPost += 2
                else:
                    idxPost += 1
                idxPrev = idxPost - 1
                tmpListLayerID = []
        else:
            tmpListLayerID.append(ID)
            if i == len(listLayerIDGPU) - 1:
                EngineConfig = ENGINE_CONFIG("GPU", tmpListLayerID)
                listEngineConfig.insert(len(listEngineConfig), EngineConfig)
            i += 1

    print("listEngineConfig:")
    for EngineConfig in listEngineConfig:
        print("{}: {}".format(EngineConfig.device, EngineConfig.listLayerID))
    print("")

    # 融合太小的 GPU子图
    i = 0
    while i < len(listEngineConfig):
        EngineConfig = listEngineConfig[i]
        if EngineConfig.device == "DLA" or len(EngineConfig.listLayerID) > 2:
            i += 1
            continue

        # 如果有的层不能运行在 DLA 上, 则不融合 GPU子图
        isMerge = True
        for ID in EngineConfig.listLayerID:
            tmpName = NetworkMap.onnx_model.graph.node[ID].name
            if not tmpName in NetworkMap.dictNetworkNode.keys():
                print("cannot find {} in dictNetworkNode".format(tmpName))
                isMerge = False
                break
            if NetworkMap.dictNetworkNode[tmpName].isCanUseDLA == False:
                print("{}: isCanUseDLA = {}".format(tmpName, False))
                isMerge = False
                break
            else:
                continue
        if isMerge == False:
            i += 1
            continue

        if i == 0 and listEngineConfig[1].device == "DLA":
            listEngineConfig[1].listLayerID = listEngineConfig[0].listLayerID + listEngineConfig[1].listLayerID
            listEngineConfig[1].listLayerID.sort()
            listEngineConfig.pop(0)
            i = 1
        elif i == len(listEngineConfig) - 1 and listEngineConfig[-2].device == "DLA":
            listEngineConfig[-2].listLayerID = listEngineConfig[-2].listLayerID + listEngineConfig[-1].listLayerID
            listEngineConfig[-2].listLayerID.sort()
            listEngineConfig.pop(-1)
            break
        elif listEngineConfig[i-1].device == "DLA" and listEngineConfig[i+1].device == "DLA":
            listEngineConfig[i-1].listLayerID = listEngineConfig[i-1].listLayerID + listEngineConfig[i].listLayerID + listEngineConfig[i+1].listLayerID
            listEngineConfig[i-1].listLayerID.sort()
            listEngineConfig.pop(i)
            listEngineConfig.pop(i)
        else:
            i += 1
    # end while

    print("listEngineConfig:")
    for EngineConfig in listEngineConfig:
        print("{}: {}".format(EngineConfig.device, EngineConfig.listLayerID))
    print("")

    # 将 融合节点到 DLA/GPU 的映射方案 保存成 json
    engines_config_json_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_engines_config.json")
    listDictEngineConfig = []
    for EngineConfig in listEngineConfig:
        dictEngineConfig = {}
        dictID_NodeName = {}
        for id in EngineConfig.listLayerID:
            dictID_NodeName[id] = NetworkMap.onnx_model.graph.node[id].name
        dictEngineConfig[EngineConfig.device] = dictID_NodeName
        listDictEngineConfig.append((dictEngineConfig))
    engines_config_scheme = json.dumps(listDictEngineConfig, indent=4)
    with open(engines_config_json_dir, "w") as file:
        file.write(engines_config_scheme)


    # 将 融合节点到 DLA/GPU 的映射方案 保存成 json
    mapping_scheme_json_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_mapping_subgraph.json")
    dumpMappingConfig(mapping_scheme_json_dir, listEngineConfig)

    # exit(0)

    # TODO: 这里只生成 trt 文件, 后边设计逻辑, 找合适的 stream 数量

    # 2. 根据融合节点到 DLA/GPU 的映射方案
    generateMultiStageEngines(NetworkMap, listEngineConfig, my_calibrator)

    # exit(0)

    suffix = "subgraph"
    engine_pipeline = loadHybridEnginePipeline(NetworkMap, suffix, dictNodeName_Device, my_calibrator)

    return engine_pipeline

def mergeSubGraph(listCandidateSubGraph, NetworkMap, io_energy_discount, ratio_time_dla, isEnergyEfficiency, ratio_bubble_time, sumTimeGPU, currExeTime, remainingFlexibleTime, currEnergy):

    listCandidateSubGraph = copy.copy(listCandidateSubGraph)
    print("\nmergeSubGraph: in", flush=True)
    if len(listCandidateSubGraph) <= 1:
        print("mergeSubGraph: len(listCandidateSubGraph) = {}".format(len(listCandidateSubGraph)), flush=True)
        return listCandidateSubGraph
    # 这里尝试进行子图融合, 如果有EDP节省且性能不弱于GPU则进行融合
    # 有相同节点
    # 前后相继
    # 按照 network_node_id 升序排列
    listCandidateSubGraph.sort(key=lambda x:x.list_network_node_id[0])

    i, j = 0, 1
    while i < len(listCandidateSubGraph) \
        and j < len(listCandidateSubGraph):
        # 判断 listCandidateSubGraph[i] 和 listCandidateSubGraph[j] 是否能融合
        # 有相同节点 / 前后相继
        SubGraph_i = listCandidateSubGraph[i]
        SubGraph_j = listCandidateSubGraph[j]

        # 完善逻辑: 防止出现执行顺序违反拓扑排序的情况
        # j子图的父节点 全在 i子图中
        flag1, flag2 = False, False
        if len(SubGraph_j.set_parent_fused_node_id) > 0:
            flag1 = (len(SubGraph_j.set_parent_fused_node_id) == len(SubGraph_j.set_parent_fused_node_id & SubGraph_i.set_fused_node_id))
        # i子图的父节点 全在 j子图中
        if len(SubGraph_i.set_parent_fused_node_id) > 0:
            flag2 = (len(SubGraph_i.set_parent_fused_node_id) == len(SubGraph_i.set_parent_fused_node_id & SubGraph_j.set_fused_node_id))

        # 从模型的网络拓扑角度看可以融合
        # if len(set_fused_node_id) > 0:
        if flag1 == True or flag2 == True:
            print("\nmergeSubGraph: 子图拓扑可融合", flush=True)
            tmpSubGraph = createMergedSubGraph([SubGraph_i, SubGraph_j])

            # 如果考虑能效: 即有能效提升才进行融合
            if isEnergyEfficiency == True:

                EnergySaving2, MinTime2, remainingFlexibleTime2, MinEnergy2 = tmpSubGraph.getEnergyTimeWithDLA(ratio_bubble_time, currExeTime, remainingFlexibleTime, currEnergy)
                EDP2 = MinEnergy2 * MinTime2

                EnergySaving0, MinTime0, remainingFlexibleTime0, MinEnergy0 = SubGraph_i.getEnergyTimeWithDLA(ratio_bubble_time, currExeTime, remainingFlexibleTime, currEnergy)
                EDP0 = MinEnergy0 * MinTime0

                EnergySaving1, MinTime1, remainingFlexibleTime1, MinEnergy1 = SubGraph_j.getEnergyTimeWithDLA(ratio_bubble_time, currExeTime, remainingFlexibleTime, currEnergy)
                EDP1 = MinEnergy1 * MinTime1

                print("EDP0 = {:.4e}; MinEnergy0 = {:.4e}; MinTime0 = {:.4e}".format(EDP0, MinEnergy0, MinTime0), flush=True)
                print("EDP1 = {:.4e}; MinEnergy1 = {:.4e}; MinTime1 = {:.4e}".format(EDP1, MinEnergy1, MinTime1), flush=True)
                print("EDP2 = {:.4e}; MinEnergy2 = {:.4e}; MinTime2 = {:.4e}".format(EDP2, MinEnergy2, MinTime2), flush=True)

                print("sumTimeGPU = {:.4e}".format(sumTimeGPU), flush=True)

                # 有能效提升 要进行融合
                if EDP2 < EDP0 and EDP2 < EDP1 and MinTime2 < sumTimeGPU:
                # if EDP2 < EDP0 and EDP2 < EDP1 and MinTime2 < sumTimeGPU \
                #     and MinEnergy2 < MinEnergy0 and MinEnergy2 < MinEnergy1:
                    print("mergeSubGraph: 有能效提升, 进行融合\n", flush=True)
                    listCandidateSubGraph[i] = tmpSubGraph
                    listCandidateSubGraph.pop(j)
                else:
                    print("mergeSubGraph: 无能效提升, 不进行融合\n", flush=True)
                    j += 1

            else:
                print("mergeSubGraph: 不考虑能效, 进行融合\n", flush=True)
                listCandidateSubGraph[i] = tmpSubGraph
                listCandidateSubGraph.pop(j)

        else:
            print("{}: {}".format("SubGraph_i", SubGraph_i.list_network_node_id))
            print("{}: {}".format("SubGraph_j", SubGraph_j.list_network_node_id))
            print("mergeSubGraph: 子图拓扑不可融合\n", flush=True)
            j += 1

        if j >= len(listCandidateSubGraph):
            i += 1
            j = i + 1

    return listCandidateSubGraph

# 找有唯一入口出口的子图
# 外层先广搜索, 找更大的子图
# 不算整个模型的入口
# 唯一入口要有多个子节点
# 先尝试找唯一入口出口子图, 找不到就拆成串行分支(一个分支中没有分叉, 首尾可以是分支节点)
def getSubGraph(NetworkMap, dictID_FusedNode, sumTimeGPU, sumEnergyGPU, io_energy_discount, ratio_time_dla, ratio_bubble_time):

    listSubGraph = []

    listBFSNode = []
    print("\ndictFusedEntryNode:")
    for tmpNode in NetworkMap.dictFusedEntryNode.values():
        listBFSNode.append(tmpNode)
        print("{}, ".format(tmpNode.getName()))
    
    setProcessedNodeID = set()
    setMIMONodeID = set() # 多入多出节点 ID
    setSIMONodeID = set() # 单入多出节点 ID
    while len(listBFSNode) > 0:
        listBFSNodeNew = []
        for ParentNode in listBFSNode:
            # tmpListFusedNode = [NetworkMap.dictFusedNode[child_name] for child_name in ParentNode.list_child_name]
            # listBFSNodeNew.extend(tmpListFusedNode)

            # 先判断当前节点是否已经处理过了
            if ParentNode.id in setProcessedNodeID:    
                continue
            # 更新 setProcessedNodeID
            setProcessedNodeID.add(ParentNode.id)
            
            # 先检查是否能提取到 唯一入口出口子图
            if ParentNode.id in NetworkMap.dictSingleIOMap:
                tmpSubGraph = NetworkMap.dictSingleIOMap[ParentNode.id]
                tmpEnergySaving, tmpMinTime, remainingFlexibleTime, tmpMinEnergy = tmpSubGraph.getEnergyTimeWithDLA(ratio_bubble_time, sumTimeGPU, sumTimeGPU, NetworkMap.sumEnergyGPU)

                # 先检查下 这个子图整体有没有 能效优化空间
                # 如果有就打包成子图处理
                # 如果没有的话就仍然拆成顺序分支处理
                # if tmpEnergySaving > 0.0 and tmpMinTime < sumTimeGPU:
                if tmpEnergySaving > 0.0 or tmpMinTime < sumTimeGPU:
                    listSubGraph.append(tmpSubGraph)

                    # 将子图出口节点加入 setMIMONodeID
                    # 以处理 两个 唯一入口出口子图 前后相继 的情况
                    listSubGraphNodeID = tmpSubGraph.list_fused_node_id 
                    setMIMONodeID.add(listSubGraphNodeID[-1])
                    # 将子图节点加入 setProcessedNode
                    setProcessedNodeID.update(listSubGraphNodeID[:-1])
                    
                    # 更新下次先广搜索buf
                    OutNode = tmpSubGraph.list_fused_node[-1]
                    listBFSNodeNew.append(OutNode)
                    continue

            else:
                # 对于前驱 唯一入口出口子图 的出口节点
                # 没有找到 唯一入口出口子图
                if ParentNode.id in setMIMONodeID:
                    # 更新下次先广搜索buf
                    tmpListFusedNode = [NetworkMap.dictFusedNode[child_name] for child_name in ParentNode.list_child_name]
                    listBFSNodeNew.extend(tmpListFusedNode)
                    # 更新 setProcessedNodeID
                    setProcessedNodeID.add(ParentNode.id)
                    continue
                # end if
            # end if

            # 如果不行就 将所在分支作为子图打包, 多入多出节点 不融合 形成单独一个子图
            listSubGraphNodeID = getSubGraph_Sequence(NetworkMap, dictID_FusedNode, ParentNode.id)
            if listSubGraphNodeID != None:
                
                # 如果只有一个节点, 且在 setSIMONodeID 中, 说明该节点之前已经被处理过一次, 本次尝试找 SingleIOMap 也没有找到
                if len(listSubGraphNodeID) == 1 and listSubGraphNodeID[0] in setSIMONodeID:
                    tmpNode = dictID_FusedNode[listSubGraphNodeID[0]]
                    # 更新下次先广搜索buf
                    tmpListFusedNode = [NetworkMap.dictFusedNode[child_name] for child_name in tmpNode.list_child_name]
                    listBFSNodeNew.extend(tmpListFusedNode)
                    # 更新 setProcessedNodeID
                    setProcessedNodeID.add(tmpNode.id)
                    continue

                tmpListFusedNode = [dictID_FusedNode[i] for i in listSubGraphNodeID]
                tmpSubGraph = SUB_GRAPH("Sequence", tmpListFusedNode, NetworkMap, io_energy_discount, ratio_time_dla, False)
                listSubGraph.append(tmpSubGraph)

                tmpOutNode = tmpListFusedNode[-1]
                if len(tmpOutNode.list_child_name) == 1:
                    # 更新下次先广搜索buf
                    tmpListFusedNode = NetworkMap.dictFusedNode[tmpOutNode.list_child_name[0]]
                    listBFSNodeNew.append(tmpListFusedNode)
                    # 将子图节点加入 setProcessedNodeID
                    setProcessedNodeID.update(listSubGraphNodeID)
                
                elif len(tmpOutNode.list_child_name) > 1:
                    # 这里有问题, 单入多出节点 怎么处理, 要避免错过 单入/单出 子图
                    # 更新下次先广搜索buf
                    listBFSNodeNew.append(tmpOutNode)
                    # 将子图节点加入 setProcessedNodeID
                    setProcessedNodeID.update(listSubGraphNodeID[:-1])
                    setSIMONodeID.add(listSubGraphNodeID[-1])
            else:
                # 更新下次先广搜索buf
                tmpListFusedNode = [NetworkMap.dictFusedNode[child_name] for child_name in ParentNode.list_child_name]
                listBFSNodeNew.extend(tmpListFusedNode)
                # 更新 setProcessedNodeID
                setProcessedNodeID.add(ParentNode.id)

        listBFSNode = list(set(listBFSNodeNew))

    return listSubGraph

# 尝试获得 唯一输入输出子图
# 输入是可能的唯一入口节点, 即只需向子节点方向搜索
def getSubGraph_SingleIOMap(NetworkMap, dictID_FusedNode, InNodeID):

    # print("getSubGraph_SingleIOMap: InNode: {}".format(dictID_FusedNode[InNodeID].getName(NetworkMap.useless_prefix)))
    
    # 如果子节点数量 <= 1 就直接返回
    # 检查出口节点是否在 GPU/DLA 上都能运行
    if not InNodeID in dictID_FusedNode.keys():
        return None # 这里怎么返回值 ???

    # print("getSubGraph_SingleIOMap: in: InNode: {}".format(dictID_FusedNode[InNodeID].getName()))

    if len(dictID_FusedNode[InNodeID].list_child_name) <= 1:
        return None # 这里怎么返回值 ???
    
    InNode = dictID_FusedNode[InNodeID]
    # if False == canUseDLA_GPU(InNode):
    if False == InNode.isCanUseDLA:
        return None # 检查入口节点是否在 GPU/DLA 上都能运行

    # 先广搜索 记录 trace
    listBFSTraceID = [] # 先广搜索的各个 trace, 记录几点的 id
    listBFSTraceIDNew = []
    listOutNodeID = []

    # 初始化 listBFSTraceID
    for child_name in InNode.list_child_name:
        ChildNode = NetworkMap.dictFusedNode[child_name]
        listBFSTraceID.append([ChildNode.id])
    
    countNewChild = len(InNode.list_child_name)

    # 先广搜索 找 SingleIOMap
    while countNewChild > 0:
        countNewChild = 0
        listBFSTraceIDNew = []
        for BFSTraceID in listBFSTraceID:
            if len(BFSTraceID) >= 10:
                listBFSTraceIDNew.append(BFSTraceID)
                continue
            TailNode = dictID_FusedNode[BFSTraceID[-1]]

            if len(TailNode.list_child_name) == 0:
                listBFSTraceIDNew.append(BFSTraceID)

            else:
                countNewChild += len(TailNode.list_child_name)
                for child_name in TailNode.list_child_name:
                    ChildNode = NetworkMap.dictFusedNode[child_name]
                    tmpBFSTraceID = BFSTraceID.copy()
                    tmpBFSTraceID.append(ChildNode.id)
                    listBFSTraceIDNew.append(tmpBFSTraceID)
        # end for

        listBFSTraceID = listBFSTraceIDNew
        # print("listBFSTraceID: ", flush=True)
            # for BFSTraceID in listBFSTraceID:
            #     print("BFSTraceID: ", end="", flush=True)
            #     for NodeID in BFSTraceID:
            #         print("{}, ".format(dictID_FusedNode[NodeID].getName(self.useless_prefix)), end="", flush=True)
            #     print("\n", flush=True)

        if len(listBFSTraceID) == 0:
            continue
        intersection = set(listBFSTraceID[-1])
        for i in range(len(listBFSTraceID)-1):
            intersection = intersection & set(listBFSTraceID[i])

        # 交集不为空 则 说明找到了公共的出口节点
        if len(intersection) >= 1:
            listOutNodeID = list(intersection)
            break
    # end while

    # 允许有一个路径没有交集
    if len(listOutNodeID) == 0 and len(listBFSTraceID) >= 3:
        # print("getOneSingleIOMap: 尝试舍弃一个根路径")
        setFirstNodeID = set([BFSTraceID[0] for BFSTraceID in listBFSTraceID])

        for FirstNodeID in list(setFirstNodeID):
            tmpListBFSTraceID = []
            for BFSTraceID in listBFSTraceID:
                if BFSTraceID[0] != FirstNodeID:
                    tmpListBFSTraceID.append(BFSTraceID)

            intersection = set(tmpListBFSTraceID[-1])
            for i in range(len(tmpListBFSTraceID)-1):
                intersection = intersection & set(tmpListBFSTraceID[i])

            # 交集不为空 则 说明找到了公共的出口节点
            if len(intersection) >= 1:
                listOutNodeID = list(intersection)
                listBFSTraceID = tmpListBFSTraceID
                # print("getOneSingleIOMap: 舍弃一个根路径")
                break
        # end for
        if len(intersection) >= 1:
            listOutNodeID = list(intersection)
        # print("getOneSingleIOMap: listOutNodeID = {}".format(listOutNodeID))

    # if len(listOutNodeID) != 1:
    #     return None
    if len(listOutNodeID) == 0:
        return None
    
    # 确定出口节点
    listOutNodeID.sort()
    OutNodeID = listOutNodeID[0]
    if not OutNodeID in dictID_FusedNode.keys():
        return None # 检查出口节点是否在 GPU/DLA 上都能运行
    OutNode = dictID_FusedNode[OutNodeID]
    # if False == canUseDLA_GPU(InNode):
    if False == OutNode.isCanUseDLA:
        return None # 检查出口节点是否在 GPU/DLA 上都能运行

    # 整理 SubGraph 节点, 输入节点在最前, 输出节点在最后
    listSubGraphNodeID = []
    listSubGraphNodeID.append(InNodeID)

    setInternalNodeID = set()
    for BFSTraceID in listBFSTraceID:
        idx = BFSTraceID.index(OutNodeID)
        for NodeID in BFSTraceID[:idx]:
            if not NodeID in dictID_FusedNode.keys():
                return None
            tmpNode = dictID_FusedNode[NodeID]
            # if False == canUseDLA_GPU(tmpNode):
            if False == tmpNode.isCanUseDLA:
                return None # 检查内部节点是否在 GPU/DLA 上都能运行
        setInternalNodeID.update(BFSTraceID[:idx])
    listSubGraphNodeID.extend(list(setInternalNodeID))
    listSubGraphNodeID.append(OutNodeID)
    if len(listSubGraphNodeID) < 4:
            return None

    # 检查内部节点的 所有父/子节点是否都在子图内
    for NodeID in setInternalNodeID:
        FusedNode = dictID_FusedNode[NodeID]
        for parent_name in FusedNode.list_parent_name:
            ParentID = NetworkMap.dictFusedNode[parent_name].id
            if not ParentID in listSubGraphNodeID:
                return None

        for child_name in FusedNode.list_child_name:
            ChildID = NetworkMap.dictFusedNode[child_name].id
            if not ChildID in listSubGraphNodeID:
                return None

    # 检查出口节点的 所有父节点是否都在子图内
    for parent_name in OutNode.list_parent_name:
        ParentID = NetworkMap.dictFusedNode[parent_name].id
        if not ParentID in listSubGraphNodeID:
            return None
    
    print("getSubGraph_SingleIOMap: InNode: {}".format(dictID_FusedNode[InNodeID].getName()))
    print("getSubGraph_SingleIOMap: OutNode: {}".format(dictID_FusedNode[OutNodeID].getName()))
    print("")
    print("getSubGraph_SingleIOMap: listSubGraphNodeID: {}".format(listSubGraphNodeID))
    print("")

    # 经过检查, 至此确定子图只有唯一的入口和出口节点
    return listSubGraphNodeID

# 尝试获得 无分支的 顺序的 融合节点序列
# 不聚合多入多出节点
# 融合 单入多出 / 多入单出 节点
# 输入是可能的唯一入口节点, 即只需向子节点方向搜索
def getSubGraph_Sequence(NetworkMap, dictID_FusedNode, InNodeID):

    if not InNodeID in dictID_FusedNode.keys():
        return None
    InNode = dictID_FusedNode[InNodeID]
    # if False == canUseDLA_GPU(InNode):
    if False == InNode.isCanUseDLA:
        return None # 如果不是 既能运行在DLA 又能运行在GPU 则 不考虑该节点
    # 不聚合多入多出节点
    # 聚合 单入多出 / 多入单出 节点
    if len(InNode.list_parent_name) > 1 and len(InNode.list_child_name) > 1:
        return [InNodeID]
    
    listSubGraphNodeID = [InNodeID]
    ParentNode = dictID_FusedNode[listSubGraphNodeID[-1]]
    # 尝试聚合后继节点
    while len(ParentNode.list_child_name) == 1:
        ChildNode = NetworkMap.dictFusedNode[ParentNode.list_child_name[0]]
        ParentNode = ChildNode

        # if False == canUseDLA_GPU(ChildNode):
        if False == ChildNode.isCanUseDLA:
            break # 如果不是 既能运行在DLA 又能运行在GPU 则 不考虑该节点

        if ChildNode.id in listSubGraphNodeID:
            break # 防止出现环

        # 1 个父节点 且 1 个子结点, append(ChildNode) 并 continue
        if len(ChildNode.list_parent_name) == 1 and len(ChildNode.list_child_name) <= 1:
            listSubGraphNodeID.append(ChildNode.id)

        # 1 个父节点 且 多个子结点, append(ChildNode) 并 break
        elif len(ChildNode.list_parent_name) == 1 and len(ChildNode.list_child_name) > 1:
            listSubGraphNodeID.append(ChildNode.id)
            break

        # 多个父节点, break
        else:
            break

    return listSubGraphNodeID

def loadHybridEnginePipeline(NetworkMap, suffix, dictNodeName_Device, my_calibrator):
    onnx_file_dir = NetworkMap.infered_file_dir
    trt_engine_gpu_dla_file_dir = os.path.join(NetworkMap.onnx_folder_dir, NetworkMap.onnx_file_name + "_gpu_dla_" + suffix + ".trt")
    BatchSize = 1
    default_device_type = trt.DeviceType.GPU

    print("\nstart generating trt engine ...", flush=True)
    trt_engine_gpu_dla0 = get_engine(onnx_file_dir, trt_engine_gpu_dla_file_dir, BatchSize, default_device_type, dictNodeName_Device, 0, my_calibrator)
    print("get_engine DLA0 done", flush=True)
    trt_engine_gpu_dla1 = get_engine(onnx_file_dir, trt_engine_gpu_dla_file_dir, BatchSize, default_device_type, dictNodeName_Device, 1, my_calibrator)
    print("get_engine DLA1 done", flush=True)

    # 2(GPU-DLA)
    list_pipeline_stage_engine = \
    [[ENGINE_STREAM(trt_engine_gpu_dla0),ENGINE_STREAM(trt_engine_gpu_dla1)]]
    print("loadHybridEnginePipeline: ENGINE_STREAM complete", flush=True)

    RingLen = 4
    engine_pipeline = ENGINE_PIPELINE(list_pipeline_stage_engine, RingLen)
    print("loadHybridEnginePipeline: ENGINE_PIPELINE complete", flush=True)

    return engine_pipeline

def loadMultiStagePipeline(NetworkMap, listEngineConfig):
    print("loadMultiStagePipeline: in")
    
    # 下边生成多阶段的多个 engine
    # onnx / engine 命名示例:
    # vgg16_stage0_dla.onnx
    # vgg16_stage0_dla.trt
    # vgg16_stage1_gpu.onnx
    # vgg16_stage1_gpu.trt
    for i in range(len(listEngineConfig)):
        EngineConfig = listEngineConfig[i]
        # 生成子 onnx 模型文件
        SubModelName = NetworkMap.onnx_file_name + "_stage" + str(i)
        if EngineConfig.device == "DLA":
            SubModelName += "_dla"
        else:
            SubModelName += "_gpu"

        sub_onnx_model_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + ".onnx")
        sub_trt_engine_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + ".trt")
        sub_trt_layer_information_file_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_layer_information.json")
        sub_trt_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_verbose.log")
        sub_trt_perf_log_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + "_trtexec_perf.log")

        SubModel, isValidModel = NetworkMap.createSubOnnxModel(EngineConfig.listLayerID, SubModelName)

        if isValidModel == True:
            onnx.save_model(SubModel, sub_onnx_model_dir)
        else:
            tmp_dir = re.sub("\.onnx", "_error.onnx", sub_onnx_model_dir)
            onnx.save_model(SubModel, tmp_dir)
            print("子模型生成错误: {}".format(SubModelName), flush=True)
            exit(0)

        SubNetworkMap = network_map(sub_onnx_model_dir)
        # 使用简单逻辑, 如果存在静态输入数据就使用
        strInput = ""
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

        if not os.path.exists(sub_trt_engine_dir):
            strCMD = "trtexec"
            strCMD += " --onnx=\"" + sub_onnx_model_dir + "\""
            strCMD += " --saveEngine=\"" + sub_trt_engine_dir + "\""
            strCMD += strInput
            strCMD += " --best --buildOnly" # --useManagedMemory --threads" # --profilingVerbosity=detailed"
            
            if EngineConfig.device == "DLA":
                strCMD += " --useDLACore=1 --allowGPUFallback"

            # strCMD += " --exportProfile=\"" + sub_trt_layer_information_file_dir + "\""
            strCMD += " 1>\"" + sub_trt_log_dir + "\"" + " 2>&1"
            print(strCMD, flush=True)
            os.system(strCMD)
        if not os.path.exists(sub_trt_perf_log_dir) and os.path.exists(sub_trt_engine_dir):
            strCMD = "trtexec"
            strCMD += " --loadEngine=\"" + sub_trt_engine_dir + "\""
            strCMD += strInput
            strCMD += " --iterations=20 --warmUp=1000" # --useManagedMemory"

            if EngineConfig.device == "DLA":
                strCMD += " --useDLACore=1"

            strCMD += " 1>\"" + sub_trt_perf_log_dir + "\"" + " 2>&1"
            print(strCMD, flush=True)
            os.system(strCMD)
    
    list_pipeline_stage_engine = []
    for i in range(len(listEngineConfig)):
        pipeline_stage_engine = []
        EngineConfig = listEngineConfig[i]
        # 生成子 onnx 模型文件
        SubModelName = NetworkMap.onnx_file_name + "_stage" + str(i)
        if EngineConfig.device == "DLA":
            SubModelName += "_dla"
        else:
            SubModelName += "_gpu"

        sub_onnx_model_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + ".onnx")
        sub_trt_engine_dir = os.path.join(NetworkMap.onnx_folder_dir, SubModelName + ".trt")

        if EngineConfig.device == "DLA":
            with open(sub_trt_engine_dir, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                runtime.max_threads = 8
                runtime.DLA_core = 0
                sub_trt_engine_dla0 = runtime.deserialize_cuda_engine(f.read())
                print("loadMultiStagePipeline: load dla0 engine done", flush=True)

            with open(sub_trt_engine_dir, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                runtime.max_threads = 8
                runtime.DLA_core = 1
                sub_trt_engine_dla1 = runtime.deserialize_cuda_engine(f.read())
                print("loadMultiStagePipeline: load dla1 engine done", flush=True)
            pipeline_stage_engine = [ENGINE_STREAM(sub_trt_engine_dla0), ENGINE_STREAM(sub_trt_engine_dla1)]
        
        else:
            with open(sub_trt_engine_dir, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                runtime.max_threads = 8
                sub_trt_engine_gpu = runtime.deserialize_cuda_engine(f.read())
                print("loadMultiStagePipeline: load gpu engine done", flush=True)
            pipeline_stage_engine = [ENGINE_STREAM(sub_trt_engine_gpu)]
            
            list_pipeline_stage_engine.append(pipeline_stage_engine)
        print("loadMultiStagePipeline: load {}.trt done\n".format(SubModelName), flush=True)
    
    RingLen = 4
    engine_pipeline = ENGINE_PIPELINE(list_pipeline_stage_engine, RingLen)
    print("loadMultiStagePipeline: ENGINE_PIPELINE done", flush=True)

    return engine_pipeline

def generateMultiStageEngines(NetworkMap, listEngineConfig, my_calibrator):

    print("generateMultiStageEngines: in", flush=True)

    for EngineConfig in listEngineConfig:

        device_name = EngineConfig.device
        listLayerID = EngineConfig.listLayerID

        isConsecutive = False
        if len(listLayerID) - 1 == listLayerID[-1] - listLayerID[0]:
            isConsecutive = True

        # 对于连续的映射 sub_model_[0-10].trt
        # 对于不连续的映射 sub_model_[0...10].trt

        onnx_file_dir = NetworkMap.infered_file_dir
        
        if isConsecutive == True:
            sub_onnx_name = NetworkMap.onnx_file_name + "_[{}-{}]".format(listLayerID[0], listLayerID[-1])
        else:
            sub_onnx_name = NetworkMap.onnx_file_name + "_[{}...{}]".format(listLayerID[0], listLayerID[-1])

        sub_onnx_file_dir = os.path.join(NetworkMap.onnx_folder_dir, sub_onnx_name+".onnx")

        if device_name == "DLA":
            sub_trt_name = sub_onnx_name + "_dla"
        else:
            sub_trt_name = sub_onnx_name + "_gpu"
            
        sub_trt_engine_file_dir = os.path.join(NetworkMap.onnx_folder_dir, sub_trt_name+".trt")
        sub_log_dir = os.path.join(NetworkMap.onnx_folder_dir, sub_trt_name+".log")
        print("generateMultiStageEngines: sub_trt_engine_file_dir: {}".format(sub_trt_engine_file_dir))
        print("generateMultiStageEngines: sub_log_dir: {}".format(sub_log_dir))


        SubOnnxModel, isValidModel = NetworkMap.createSubOnnxModel(listLayerID, sub_onnx_name)
        onnx.save_model(SubOnnxModel, sub_onnx_file_dir)
        
        if not os.path.exists(sub_trt_engine_file_dir) or not os.path.exists(sub_log_dir):
            strCMD = "trtexec"
            strCMD += " --onnx=\"" + sub_onnx_file_dir + "\""
            strCMD += " --saveEngine=\"" + sub_trt_engine_file_dir + "\""

            if device_name == "DLA":
                strCMD += " --useDLACore=0 --allowGPUFallback"

            strCMD += " --best --buildOnly" # --threads --useManagedMemory" # --useManagedMemory
            strCMD += " 1>" + sub_log_dir + " 2>&1"
            print(strCMD, flush=True)
            # time_begin = time.time()
            os.system(strCMD)
            # time_duration = time.time() - time_begin
            # print("generating GPU-only engine time: {}".format(time_duration), flush=True)

    return

