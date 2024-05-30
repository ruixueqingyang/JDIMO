# coding=utf-8
import os, json, re
from time import time
import copy
import onnx
import onnx.version_converter
import onnxruntime
# import sys, getopt
import numpy as np
# import torch
from trt_engine_memory import getDLALayerNames

isGPU = "GPU"
isDLA = "DLA"

# 定义 profiling 数据结构
class profiling_data:
    def __init__(self) -> None:
        # self.num_executions = 0.0 # 测量时执行次数
        # self.total_exe_time = 0.0 # 总执行时间 (s)
        # self.total_energy = 0.0 # 总能耗 (J)
        # self.total_computation = 0.0 # 总计算量
        # self.total_mem_access = 0.0 # 总内存访问量
        self.avg_exe_time = 0.0 # 平均执行时间 (s)
        self.avg_energy = 0.0 # 平均能耗 (J)
        self.avg_power = 0.0 # 平均功率 (W)
        self.avg_computation = 0.0 # 平均计算量
        self.avg_mem_access = 0.0 # 平均内存访问量
        self.energy_efficiency_index = -1 # 能效指数 / 优化目标函数

        self.avg_input_time = 0.0 # 输入开销时间
        self.avg_output_time = 0.0 # 输出开销时间

        self.power_dynamic = 0.0 # 动态功率 (W)
        self.energy_dynamic = 0.0 # 动态能耗 (J)
        self.energy_static = 0.0 # 静态能耗 (J)

        # # 对 DLA 上的测量数据进行补偿, 减去一半的静态功耗, 执行时间减半
        # self.avg_exe_time_compensation = 0.0 # 平均执行时间 (s)
        # self.avg_energy_compensation = 0.0 # 平均能耗 (J)
        # self.avg_power_compensation = 0.0 # 平均功率 (W)

    def set(self, inData):
        self.avg_exe_time = inData.avg_exe_time
        self.avg_energy = inData.avg_energy
        self.avg_power = inData.avg_power
        self.avg_computation = inData.avg_computation
        self.avg_mem_access = inData.avg_mem_access
        self.energy_efficiency_index = inData.energy_efficiency_index
        self.avg_input_time = inData.avg_input_time
        self.avg_output_time = inData.avg_output_time
        self.power_dynamic = inData.power_dynamic
        self.energy_dynamic = inData.energy_dynamic
        self.energy_static = inData.energy_static

class fused_node:
    def __init__(self, id, list_onnx_node = [], useless_prefix="") -> None:
        self.id = id
        self.str_ops = ""
        # self.list_onnx_node = copy.copy(list_onnx_node)
        self.list_onnx_node = list_onnx_node
        # self.list_onnx_node_name = [] # 融合节点包含的 onnx 节点名
        self.list_parent_name = [] # network_node 类型 父节点列表
        self.list_child_name = [] # network_node 类型 子节点列表

        self.isCanUseDLA = False # 能够在 DLA 上运行
        self.isCanUseGPU = True # 能够在 GPU 上运行
        self.DeviceType = isGPU # 选择 DLA/GPU 来运行
        
        # 节点是否能单独运行, 对于有未知维度的节点, 认为不能单独运行
        # 此类节点不进行测量, 且默认在GPU上运行
        self.hasUnknownDim = False # 是否有未知维度

        self.gpu_profiling = profiling_data() # 当前网络节点运行在 GPU 上测得的数据
        self.dla_profiling = profiling_data() # 当前网络节点运行在 DLA 上测得的数据

        self.resolveOps()

    def canUseDLA(self, setDLALayerName):
        self.isCanUseDLA = True
        for onnx_node in self.list_onnx_node:
            if not onnx_node.name in setDLALayerName:
                print("canUseDLA: LayerName: {}; isCanUseDLA: {}".format(onnx_node.name, False))
                self.isCanUseDLA = False
                break
            else:
                print("canUseDLA: LayerName: {}; isCanUseDLA: {}".format(onnx_node.name, True))
        return self.isCanUseDLA

    def copyProfilingData(self, inFusedNode):
        self.isCanUseDLA = inFusedNode.isCanUseDLA
        self.isCanUseGPU = inFusedNode.isCanUseGPU
        self.gpu_profiling.set(inFusedNode.gpu_profiling)
        self.dla_profiling.set(inFusedNode.dla_profiling)

    def resolveOps(self):
        for i in range(len(self.list_onnx_node)):
            onnx_node = self.list_onnx_node[i]
            self.str_ops += onnx_node.op_type
            if i != len(self.list_onnx_node)-1:
                self.str_ops += "+"

# name: "resnetv22_stage1_activation1"
# type {
#   tensor_type {
#     elem_type: 1
#     shape {
#       dim {
#         dim_param: "N"
#       }

    # 获得文件名前缀, Linux 文件名长度上限 255
    def getName(self, useless_prefix=""):
        listSubModelName = [onnx_node.name + "+" for onnx_node in self.list_onnx_node]
        SubModelName = "".join(listSubModelName)
        SubModelName = SubModelName[:-1]

        # useless_prefix 节点名中重复的/无用的前缀字符串, 在生成测量文件名时, 可以移除这些前缀以缩短文件名
        if len(useless_prefix) > 0:
            SubModelName = re.sub(useless_prefix, "", SubModelName)
        SubModelName = re.sub("/", "-", SubModelName)
        SubModelName = re.sub("\:", "--", SubModelName)

        # 文件名前缀不能过长, 还需要给后缀留空间
        if len(SubModelName) > 200:
            SubModelName = SubModelName[:200]
        
        return SubModelName

# 定义NN网络节点
class network_node:
    def __init__(self, id, onnx_node) -> None:
        print("network_node: onnx_node.name = \"{}\"".format(onnx_node.name))
        self.id = id
        self.str_type = "" # 用来描述融合节点的类型: layer_name:1x2x3x4+layer_name:1x2x3x4......
        self.onnx_node = onnx_node # onnx 模型格式的节点
        self.list_parent = [] # network_node 类型 父节点列表
        self.list_child = [] # network_node 类型 子节点列表

        self.isCanUseDLA = False # 能够在 DLA 上运行
        self.isCanUseGPU = True # 能够在 GPU 上运行

        self.gpu_profiling = profiling_data() # 当前网络节点运行在 GPU 上测得的数据
        self.dla_profiling = profiling_data() # 当前网络节点运行在 DLA 上测得的数据

    # 获得文件名前缀, Linux 文件名长度上限 255
    def getName(self, useless_prefix=""):
        SubModelName = self.onnx_node.name

        # useless_prefix 节点名中重复的/无用的前缀字符串, 在生成测量文件名时, 可以移除这些前缀以缩短文件名
        if len(useless_prefix) > 0:
            SubModelName = re.sub(useless_prefix, "", SubModelName)
        SubModelName = re.sub("/", "-", SubModelName)
        SubModelName = re.sub("\:", "--", SubModelName)

        # 文件名前缀不能过长, 还需要给后缀留空间
        if len(SubModelName) > 200:
            SubModelName = SubModelName[:200]
        
        return SubModelName


class network_map:

# name: "resnetv22_stage1_activation1"
# type {
#   tensor_type {
#     elem_type: 1
#     shape {
#       dim {
#         dim_param: "N"
#       }

    # list_input_dim: 用来设置 模型输入 tensor 中未知维度
    def __init__(self, onnx_file_dir, dictUnknownShape={}, useless_prefix=""):
        # self.list_input_dim = list_input_dim
        self.dictUnknownShape = dictUnknownShape
        self.onnx_file_dir = onnx_file_dir # 源 onnx 模型文件路径
        tmpStr = os.path.split(onnx_file_dir)
        self.onnx_folder_dir = tmpStr[0] # onnx 文件所在文件夹
        self.onnx_file_name = os.path.splitext(tmpStr[1])[0] # onnx 文件名 (不包括扩展名)
        self.infered_file_dir = os.path.join(self.onnx_folder_dir, self.onnx_file_name+"_infered.onnx")
        self.all_output_file_dir = os.path.join(self.onnx_folder_dir, self.onnx_file_name+"_all_output.onnx")
        # 节点名中重复的/无用的前缀字符串, 在生成测量文件名时, 可以移除这些前缀以缩短文件名
        self.useless_prefix = useless_prefix

        print("onnx_file_dir = {}".format(self.onnx_file_dir))
        print("network_map: infered_file_dir = {}".format(self.infered_file_dir))

        # 获得 输入/输出 (未知输入维度) 信息
        provider = "CPUExecutionProvider"
        # InferenceSession 这个开销不太大, 可以使用
        onnxrt_session = onnxruntime.InferenceSession(self.onnx_file_dir, providers=[provider])

        self.input_tensors = onnxrt_session.get_inputs() # 获得 onnx模型 的输入 该 API 会返回列表
        self.output_tensors = onnxrt_session.get_outputs() # 获得 onnx模型 的输出 该 API 会返回列表
        del onnxrt_session
        self.unknown_dim_count = 0
        # self.listModelInputName = [] # 所有 输入 tensor 名称
        self.listUnknownDimInputName = [] # 包含未知维度的 输入 tensor 名称
        for input_tensor in self.input_tensors:
            # self.listModelInputName.append(input_tensor.name)
            tmp_flag = False
            for dim in input_tensor.shape:
                # 如果维度不是整数, 或者 < 1, 则认为该维度是未知维度
                if isinstance(dim, int) == False or dim < 1:
                    self.unknown_dim_count += 1
                    tmp_flag = True
            if tmp_flag == True: 
                self.listUnknownDimInputName.append(input_tensor.name)

        # 检查 含有未知维度输入变量的数量 是否等于 dictUnknownShape 给出的shape数量
        if len(self.listUnknownDimInputName) != len(self.dictUnknownShape):
            print("ERROR: The model contains {} unknown input shapes. But {} shapes are given.".format(len(self.listUnknownDimInputName), len(self.dictUnknownShape)))
            exit(1)

        # # 检查 未知维度数量 是否等于 输入维度数量
        # if self.unknown_dim_count != len(list_input_dim):
        #     print("ERROR: The model contains {} unknown dimensions. But {} dimensions are given.".format(self.unknown_dim_count, len(list_input_dim)))
        #     exit(1)

        self.onnx_model = onnx.load(onnx_file_dir) # 加载 onnx 模型
        onnx.checker.check_model(self.onnx_model) # 检查 onnx 模型
        # print("network_map: load onnx model done")

        self.dla_profiling = profiling_data() # 网络整体尽量运行在 DLA 上测得的数据
        self.gpu_profiling = profiling_data() # 网络整体运行在 GPU 上测得的数据
        self.dla_sum_exe_time, self.dla_sum_energy = 0.0, 0.0
        self.gpu_sum_exe_time, self.gpu_sum_energy = 0.0, 0.0
        self.array_dla_gpu_count = None # 两组测量(尽量使用DLA/GPU)中 使用 DLA/GPU 的数量
        self.array_delta_time = None # 两组测量(尽量使用DLA/GPU)中 各层加和时间 减去 整体测量时间

        self.dictInput = {} # onnx input 转换成字典形式保存
        self.dictOutput = {} # onnx output 转换成字典形式保存
        self.dictValueInfo = {} # onnx value_info 转换成字典形式保存
        self.dictInitializer = {} # onnx initializer 转换成字典形式保存

        self.listModelInput = [] # 实际输入 即 不在 initializer 中 数值未知的输入
        self.listModelOutput = [] # 实际输出 即 不在 initializer 中 数值未知的输出
        self.dictNetworkNode = {} # 网络中每个节点的字典
        self.unnamed_node_id = 0

        # 假设所有节点中都不存在同名的 output
        self.dictOutputNameNode = {} # 字典: key=输出名: value=该输出所在节点名
        # 这样可以通过 output name 找到 所在的节点

        self.listEntryNode = [] # 模型入口节点 (可能是常量入口)
        self.listExitNode = [] # 模型出口节点
        
        self.dictFusedEntryNode = {}
        self.dictFusedExitNode = {}
        # self.listFusedNode = [] # 融合节点, 读取 trtexec 输出信息, 获得融合节点信息
        self.fused_node_id = 0
        self.dictFusedNode = {} # 融合节点, 读取 trtexec 输出信息, 获得融合节点信息
        self.dictSingleIOMap = {} # 唯一入口出口子图, 用来进行额外的整体测量, key 是 onnx id 拼接在一起的字符串: "[10,11,12]"

        self.dictOps_listFused = {} # key: 多个层的op_type字符串连接, value: FusedNode的list; 用来避免重复测量: 进行测量前先进行匹配, 如有所有op完全相同的FusedNode, 则读取测量结果, 否则进行测量并存入

        self.isCanUseDLA = True # 能够在 DLA 上运行
        self.isCanUseGPU = True # 能够在 GPU 上运行
        self.setDLALayerName = set()
        self.idle_power = 0.0 # 系统空载功率 W
        self.NumGPU = 1
        self.NumDLA = 2
        self.MaxSubGraphsPerDLA = 10 # 10 # 一个 DLA 支持的子图数量上限
        self.sumTimeGPU = 0.0
        self.sumEnergyGPU = 0.0

        self.buildDictionary() # 将 onnx 格式模型 中 信息 转换为 字典结构 并保存
        self.findParentChild() # 找到每个节点的父节点和子节点

        # 至此原始 onnx 网络解析完成, 推导出了各个中间 tensor 的维度, 将信息存储字典
        # 并 生成图结构, 节点数据类型是 network_node, 图入口 listEntryNode / 图出口 listExitNode

        # self.onnx_folder_dir = tmpStr[0]  # onnx 文件所在文件夹
        # self.onnx_file_name = os.path.splitext(tmpStr[1])[0]  # onnx 文件名 (不包括扩展名)
        # infer_shapes
        if not os.path.exists(self.infered_file_dir):
            onnx.save_model(self.onnx_model, self.infered_file_dir)

        self.setDLALayerName = getDLALayerNames(self.infered_file_dir)
        self.setCanUseDLA()

    # 将 onnx 格式模型 中 信息 转换为 字典结构 并保存
    def buildDictionary(self):
        for input in self.onnx_model.graph.input:
            self.dictInput[input.name] = input
        # 使用 给定的维度信息 设置 输入 tensor 中的未知维度
        # dim_count = 0
        # for input_name in self.listUnknownDimInputName:
        #     tmp_input = self.dictInput[input_name]
        #     for dim in tmp_input.type.tensor_type.shape.dim:
        #         if dim.dim_value < 1:
        #             # print("list_input_dim[{}] = {}".format(dim_count, self.list_input_dim[dim_count]))
        #             # print("dim.dim_value = {}, dim.dim_param = {}".format(dim.dim_value, dim.dim_param))
        #             # dim.dim_param = "" # 只对 dim_param / dim_value 其中一个赋值, 这个结构 似乎类似 C语言的 共用体(union), 多种类型共用同一块内存空间
        #             dim.dim_value = self.list_input_dim[dim_count]
        #             dim_count += 1

        for input in self.onnx_model.graph.input:
            InputShape = []
            for j in range(len(input.type.tensor_type.shape.dim)):
                dim = input.type.tensor_type.shape.dim[j]
                if dim.dim_value < 1:
                    dim.dim_value = self.dictUnknownShape[input.name][j]
                InputShape.append(dim.dim_value)

        # 推理中间变量 shape 存到 value_info
        self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
        
        for input in self.onnx_model.graph.input:
            self.dictInput[input.name] = input
        for output in self.onnx_model.graph.output:
            self.dictOutput[output.name] = output
        for value_info in self.onnx_model.graph.value_info:
            self.dictValueInfo[value_info.name] = value_info
        for initializer in self.onnx_model.graph.initializer:
            self.dictInitializer[initializer.name] = initializer

        # 模型 输入/输出
        for input_tensor in self.input_tensors:
            input_name = input_tensor.name
            self.listModelInput.append(self.dictInput[input_name])
        for output_tensor in self.output_tensors:
            output_name = output_tensor.name
            self.listModelOutput.append(self.dictOutput[output_name])
        
        node_id = 0
        for onnx_node in self.onnx_model.graph.node:
            if onnx_node.name == "":
                onnx_node.name = "unnamed_" + str(self.unnamed_node_id)
                self.unnamed_node_id += 1

            self.dictNetworkNode[onnx_node.name] = network_node(node_id, onnx_node)
            self.dictFusedNode[onnx_node.name] = fused_node(self.fused_node_id, [onnx_node])
            self.fused_node_id += 1

            for output_name in onnx_node.output:
                self.dictOutputNameNode[output_name] = self.dictNetworkNode[onnx_node.name]

            node_id += 1

    # 找到每个节点的父节点和子节点
    def findParentChild(self):
        # print("findParentChild in:")
        for NodeName, NetworkNode in self.dictNetworkNode.items():

            # print("NodeName: {}".format(NodeName))

            for input_name in NetworkNode.onnx_node.input:
                # 如果 input_name 是某个节点的输出
                if input_name in self.dictOutputNameNode.keys():
                    ParentName = self.dictOutputNameNode[input_name].onnx_node.name
                    # print("ParentName: {}".format(ParentName))
                    NetworkNode.list_parent.append(self.dictOutputNameNode[input_name])
                    self.dictFusedNode[NodeName].list_parent_name.append(ParentName)

            # 将 当前节点 存入 父结点的 子节点列表
            for parentNetworkNode in NetworkNode.list_parent:
                parent_name = parentNetworkNode.onnx_node.name
                parentNetworkNode.list_child.append(NetworkNode)
                self.dictFusedNode[parent_name].list_child_name.append(NodeName)
                # print("parent_name: {}".format(parent_name))
                # print("len(list_child) = {}".format(len(parentNetworkNode.list_child)))
        # end for

        for NodeName, NetworkNode in self.dictNetworkNode.items():

            # 如果当前节点没有子节点, 则 当前节点 为 模型出口节点
            if len(NetworkNode.list_child) == 0:
                self.listExitNode.append(NetworkNode)
                self.dictFusedExitNode[NodeName] = self.dictFusedNode[NodeName]

            # 如果当前节点没有父节点, 则 当前节点 为 模型入口节点 (可能是常量入口)
            if len(NetworkNode.list_parent) == 0:
                self.listEntryNode.append(NetworkNode)
                self.dictFusedEntryNode[NodeName] = self.dictFusedNode[NodeName]
        # end for
        # print("len(self.listEntryNode) = {}".format(len(self.listEntryNode)))
        # print("len(self.listExitNode) = {}".format(len(self.listExitNode)))

    # 将必要的节点输出添加为模型输出, 记录输出值, 为后续处理未知维度做准备
    def createAllOutputModel(self):

        # 将必要的节点输出添加为模型输出, 记录输出值, 为后续处理未知维度做准备
        model_all_output = copy.deepcopy(self.onnx_model)

        dictOutputLocal = {}
        for tmp_output in model_all_output.graph.output:
            dictOutputLocal[tmp_output.name] = tmp_output
        dictValueInfoLocal = {}
        for tmp_value_info in model_all_output.graph.value_info:
            dictValueInfoLocal[tmp_value_info.name] = tmp_value_info

        # 遍历 node 找到有未知维度的 输出 tensor 放入 output
        for i in range(len(model_all_output.graph.node)):
            isSaveInput = False
            tmp_node = model_all_output.graph.node[i]
            for output_name in tmp_node.output:
                # print("createAllOutputModel: output_name = {}: ".format(output_name), end="", flush=True)
                # 已经在 output 中
                if output_name in dictOutputLocal.keys():
                    # print("already in dictOutputLocal", flush=True)
                    continue
                # 在 value_info 中
                if output_name in dictValueInfoLocal.keys():
                    tmp_value_info = dictValueInfoLocal[output_name]

                    isUnknownDim = False
                    if tmp_value_info.type.tensor_type.HasField("shape") == False:
                        isUnknownDim = True
                    elif len(tmp_value_info.type.tensor_type.shape.dim) == 0:
                        isUnknownDim = True
                    else:
                        for tmp_dim in tmp_value_info.type.tensor_type.shape.dim:
                            if tmp_dim.dim_value < 1:
                                isUnknownDim = True
                                break
                    if isUnknownDim == True:
                        # print("isUnknownDim = True", flush=True)
                        isSaveInput = True
                        # 在 value_info 中找到对应项目的 idx 并 pop 掉
                        idx = -1
                        for i in range(len(model_all_output.graph.value_info)):
                            if output_name == model_all_output.graph.value_info[i].name:
                                idx = i
                                break
                        if idx >= 0:
                            tmp_output = model_all_output.graph.value_info.pop(idx) # 这里是否要 pop ?
                            model_all_output.graph.output.append(tmp_output)
                            dictValueInfoLocal.pop(output_name)
                            dictOutputLocal[output_name] = tmp_output
                    # else:
                    #     print("isUnknownDim = False", flush=True)
                else:
                    isSaveInput = True
                    # print("construct ValueInfoProto", flush=True)
                    tmp_output = onnx.ValueInfoProto(name=output_name)
                    model_all_output.graph.output.append(tmp_output)
                    dictOutputLocal[output_name] = tmp_output

            if isSaveInput == False:
                continue

            # 处理该节点的输入, 也将输入放入 output
            for input_name in tmp_node.input:
                # 如果已经在 output 中 就 跳过
                if input_name in dictOutputLocal.keys():
                    continue
                # 如果在 value_info 中
                if input_name in dictValueInfoLocal.keys():
                    # 移动到 output
                    idx = -1
                    for i in range(len(model_all_output.graph.value_info)):
                        if input_name == model_all_output.graph.value_info[i].name:
                            idx = i
                            break
                    if idx >= 0:
                        tmp_output = model_all_output.graph.value_info.pop(idx) # 这里是否要 pop ?
                        model_all_output.graph.output.append(tmp_output)
                        dictValueInfoLocal.pop(input_name)
                        dictOutputLocal[input_name] = tmp_output
                else:
                    tmp_output = onnx.ValueInfoProto(name=input_name)
                    model_all_output.graph.output.append(tmp_output)
                    dictOutputLocal[output_name] = tmp_output

        model_all_output = onnx.shape_inference.infer_shapes(model_all_output)
        # onnx.checker.check_model(model_all_output) # 检查 onnx 模型
        if not os.path.exists(self.all_output_file_dir):
            onnx.save_model(model_all_output, self.all_output_file_dir)

        del model_all_output

    def setCanUseDLA(self):
        for DLALayerName in self.setDLALayerName:
            if DLALayerName in self.dictNetworkNode.keys():
                self.dictNetworkNode[DLALayerName].isCanUseDLA = True

        for tmpFusedNode in self.dictFusedNode.values():
            tmpFusedNode.canUseDLA(self.setDLALayerName)

# 判断层相同不是那么简单, 需要判断
# op_type
# 多个 attribute: 
# attribute {
#   name: "dilations"
#   ints: 1
#   ints: 1
#   type: INTS
# }
# 多个 input: 维度相同
# 多个 output: 维度相同
# 需要写个函数来判断两个节点以及融合节点进行的运算是否相同, 而不是自定义类型字符串
    
    def isSameShapeInput(self, input0, input1):
        if input0.type == input1.type:
            return True
        else:
            return False
        
    def isSameShapeOutput(self, Output0, Output1):
        if Output0.type == Output1.type:
            return True
        else:
            return False
        
    def isSameShapeValueInfo(self, value_info0, value_info1):
        if value_info0.type == value_info1.type:
            return True
        else:
            return False
        
    def isSameShapeInitializer(self, initializer0, initializer1):
        if initializer0.dims == initializer1.dims and initializer0.data_type == initializer1.data_type:
            return True
        else:
            return False

    # 判断两个 onnx_node 节点/层 是否有相同的 运算操作, 这样就不用重复测量了
    def isSameOpOnnxNode(self, onnx_node0, onnx_node1):
        # flag = False
        # if onnx_node0.name == "StatefulPartitionedCall/model/conv2d_2/Conv2D" or onnx_node1.name == "StatefulPartitionedCall/model/conv2d_2/Conv2D":
        #     flag = True
        #     print("onnx_node0.name: {}; onnx_node1.name {}".format(onnx_node0.name, onnx_node1.name))

        if onnx_node0.op_type != onnx_node1.op_type:
            return False
        # if flag == True:
        #     print("op_type 相同")
        
        # 判断 attribute 是否相同
        if onnx_node0.attribute != onnx_node1.attribute:
            return False
        # if flag == True:
        #     print("attribute 相同")
        
        if len(onnx_node0.input) != len(onnx_node1.input):
            return False
        
        if len(onnx_node0.output) != len(onnx_node1.output):
            return False
        # if flag == True:
        #     print("input/output 长度相同")
        

        # 判断 input 是否相同
        for i in range(len(onnx_node0.input)):
            input_name0 = onnx_node0.input[i]
            input_name1 = onnx_node1.input[i]
            
            if input_name0 in self.dictInput and input_name1 in self.dictInput and self.isSameShapeInput(self.dictInput[input_name0], self.dictInput[input_name1]):
                continue
            elif input_name0 in self.dictValueInfo and input_name1 in self.dictValueInfo and self.isSameShapeValueInfo(self.dictValueInfo[input_name0], self.dictValueInfo[input_name1]):
                continue
            elif input_name0 in self.dictInitializer and input_name1 in self.dictInitializer and self.isSameShapeInitializer(self.dictInitializer[input_name0], self.dictInitializer[input_name1]):
                continue
            else:
                return False
        # if flag == True:
        #     print("input 相同")

        # 判断 output 是否相同
        for i in range(len(onnx_node0.output)):
            output_name0 = onnx_node0.output[i]
            output_name1 = onnx_node1.output[i]
            if output_name0 in self.dictOutput and output_name1 in self.dictOutput and self.isSameShapeOutput(self.dictOutput[output_name0], self.dictOutput[output_name1]):
                continue
            elif output_name0 in self.dictValueInfo and output_name1 in self.dictValueInfo and self.isSameShapeValueInfo(self.dictValueInfo[output_name0], self.dictValueInfo[output_name1]):
                continue
            else:
                return False
        # if flag == True:
        #     print("output 相同")

        return True

    # 判断两个 fused_node 节点/层 是否有相同的 运算操作, 这样就不用重复测量了
    def isSameOpFusedNode(self, fused_node0, fused_node1):

        # 不匹配自己
        if fused_node0.id == fused_node1.id:
            return False

        if len(fused_node0.list_onnx_node) != len(fused_node1.list_onnx_node):
            return False
        
        for i in range(len(fused_node0.list_onnx_node)):
            if False == self.isSameOpOnnxNode(fused_node0.list_onnx_node[i], fused_node1.list_onnx_node[i]):
                return False

        return True

    def saveFusedNode_SameOps(self, inFusedNode):
        if not inFusedNode.str_ops in self.dictOps_listFused:
            self.dictOps_listFused[inFusedNode.str_ops] = [inFusedNode]
        else:
            self.dictOps_listFused[inFusedNode.str_ops].append(inFusedNode)

    # 先在 dictOps_listFused 中搜索是否有所有op完全相同的 FusedNode
    # 如有则返回, 否则返回 None
    def findFusedNode_SameOps(self, inFusedNode):

        if not inFusedNode.str_ops in self.dictOps_listFused:
            return None
        
        # print("str_ops {} 在 dictOps_listFused 中".format(inFusedNode.str_ops))

        listFusedNode = self.dictOps_listFused[inFusedNode.str_ops]
        for i in range(len(listFusedNode)):
            if inFusedNode.id == listFusedNode[i].id:
                return None # 不匹配自己
            elif True == self.isSameOpFusedNode(inFusedNode, listFusedNode[i]):
                # print("")
                print("\n找到所有 Op 都相同的 FusedNode: {}".format(inFusedNode.getName(self.useless_prefix)))
                return listFusedNode[i]

        return None

    # 获得 FusedNode 的无重复项 list, 且按照 id 升序排列
    def getListFusedNode(self):
        listFusedNode = []
        setFusedNodeID = set()
    
        # 首先取出所有节点, 形成一个 list (会排除重复的融合节点)
        for onnx_node in self.onnx_model.graph.node:
            if onnx_node.name in self.dictFusedNode.keys():
                tmpFusedNode = self.dictFusedNode[onnx_node.name]
                if not tmpFusedNode.id in setFusedNodeID:
                    setFusedNodeID.add(tmpFusedNode.id)
                    listFusedNode.append(tmpFusedNode)

        return listFusedNode

    # 检查 node名 是否合法, 去除 非法 node名, 只输出 合法 node名
    def checkNodeName(self, in_list_node_name):
        list_node_name = []
        for node_name in in_list_node_name:
            if node_name in self.dictNetworkNode.keys():
                list_node_name.append(node_name)

        return list_node_name
    
    # 对 两级 list 中 合法 layer名 进行计数
    def countListLayers(self, list_layer_names):
        NumLayers = 0
        for layer_names in list_layer_names:
            tmp_layer_names = self.checkNodeName(layer_names)
            NumLayers += len(tmp_layer_names)

        return NumLayers

    # 进行节点融合
    def fuseNode(self, in_list_node_name):

        print("\nfuseNode: list_node_name = {}".format(in_list_node_name))

        # 从 list_node_name 中 排除非法层名, (可能是层输入/输出变量名)
        list_node_name = self.checkNodeName(in_list_node_name)

        print("fuseNode: checkNodeName: {}".format(list_node_name))
        if len(list_node_name) == 0:
            return False
        
        set_node_name = set(list_node_name)

        # 对 list_node_name 进行修补, 保证 所有节点 对应的 所有 key 都在 list_node_name 中
        # dictFusedNode 中 多个 key(即被融合的单个层的名称) 都对应同一个融合节点
        for node_name in list_node_name:
            tmp_fused_node = self.dictFusedNode[node_name]
            for onnx_node in tmp_fused_node.list_onnx_node:
                set_node_name.add(onnx_node.name)
        
        list_node_name = [*set_node_name]

        # 从 self.dictFusedNode 中取出相关 node 且 需要去重
        set_fused_node_id = set()
        list_fused_node = []
        for node_name in list_node_name:
            tmp_fused_node = self.dictFusedNode.pop(node_name)
            if not tmp_fused_node.id in set_fused_node_id:
                set_fused_node_id.add(tmp_fused_node.id)
                list_fused_node.append(tmp_fused_node)

        # 将这些节点进行融合
        # list_onnx_node = [self.dictNetworkNode[node_name].onnx_node for node_name in list_node_name]
        listLocalNetworkNode = [self.dictNetworkNode[node_name] for node_name in list_node_name]
        # 按照 id 升序排列, 保证子节点中 onnx_node 相对顺序 与 原模型中 一样
        listLocalNetworkNode.sort(key=lambda x: x.id)
        list_onnx_node = [NetworkNode.onnx_node for NetworkNode in listLocalNetworkNode]
        list_parent_name = []
        list_child_name = []
        # set_parent_name = set()
        # set_child_name = set()
        for tmp_fused_node in list_fused_node:
                    
            for parent_name in tmp_fused_node.list_parent_name:
                # 要判断被融合节点之间是否存在父子关系, 要忽略这些父子关系
                if not parent_name in set_node_name:
                    # set_parent_name.add(parent_name)
                    list_parent_name.append(parent_name)
                    
                # parent_fused_node = self.dictFusedNode[parent_name]
                # # 要判断被融合节点之间是否存在父子关系, 要忽略这些父子关系
                # if not parent_fused_node.id in set_fused_node_id:
                #     set_parent_name.add(parent_name)

            for child_name in tmp_fused_node.list_child_name:
                # 要判断被融合节点之间是否存在父子关系, 要忽略这些父子关系
                if not child_name in set_node_name:
                    # set_child_name.add(child_name)
                    list_child_name.append(child_name)

                # child_fused_node = self.dictFusedNode[child_name]
                # if not child_fused_node.id in set_fused_node_id:
                #     set_child_name.add(child_name)
            
        # list_parent_name = [*set_parent_name]
        # list_child_name = [*set_child_name]

        tmp_fused_node = fused_node(self.fused_node_id, list_onnx_node)
        self.fused_node_id += 1
        tmp_fused_node.canUseDLA(self.setDLALayerName)
        tmp_fused_node.list_parent_name = list_parent_name
        tmp_fused_node.list_child_name = list_child_name

        # print("fuseNode: tmp_fused_node.getName() = {}".format(tmp_fused_node.getName(self.useless_prefix)))
        # if tmp_fused_node.getName(self.useless_prefix) == "tf_op_layer_LeakyRelu_19-LeakyRelu_19":
        #     print("list_parent_name: {}".format(tmp_fused_node.list_parent_name))
        #     print("list_child_name: {}".format(tmp_fused_node.list_child_name))

        # 将融合后的节点 以 多key对一副本 的方式存到 self.dictFusedNode, 其中的每层 name 作为 key 都存入同一副本
        for node_name in list_node_name:
            self.dictFusedNode[node_name] = tmp_fused_node

        # 从 self.dictFusedEntryNode 中取出相关 node, 并存入融合后的节点, 要避免重复
        flag_replace = False
        tmp_name = ""
        for node_name in list_node_name:
            if node_name in self.dictFusedEntryNode.keys():
                self.dictFusedEntryNode.pop(node_name)
                if flag_replace == False:
                    tmp_name = node_name
                flag_replace = True
        if flag_replace == True:
            self.dictFusedEntryNode[tmp_name] = tmp_fused_node

        # 从 self.dictFusedExitNode 中取出相关 node, 并存入融合后的节点, 要避免重复
        flag_replace = False
        tmp_name = ""
        for node_name in list_node_name:
            if node_name in self.dictFusedExitNode.keys():
                self.dictFusedExitNode.pop(node_name)
                if flag_replace == False:
                    tmp_name = node_name
                flag_replace = True
        if flag_replace == True:
            self.dictFusedExitNode[tmp_name] = tmp_fused_node

        return True
    
    # 找到 SingleIOMap
    def getAllSingleIOMap(self):
        
        print("getAllSingleIOMap: in")
        listBFSNode = []
        print("\ndictFusedEntryNode:")
        for tmpNode in self.dictFusedEntryNode.values():
            listBFSNode.append(tmpNode)
            print("{}, ".format(tmpNode.getName()))

        dictID_FusedNode = {}
        listFusedNode = self.getListFusedNode()
        for FusedNode in listFusedNode:
            dictID_FusedNode[FusedNode.id] = FusedNode

        setProcessedNodeID = set()
        setMIMONodeID = set() # 多入多出节点 ID
        while len(listBFSNode) > 0:
            listBFSNodeNew = []
            for ParentNode in listBFSNode:
                # listBFSNodeNewID = [Node.id for Node in listBFSNodeNew]
                # print("getAllSingleIOMap: listBFSNodeNewID = {}".format(listBFSNodeNewID))
                # tmpListFusedNode = [self.dictFusedNode[child_name] for child_name in ParentNode.list_child_name]
                # listBFSNodeNew.extend(tmpListFusedNode)

                # 先判断当前节点是否已经处理过了
                if ParentNode.id in setProcessedNodeID:
                    continue
                # 更新 setProcessedNodeID
                setProcessedNodeID.add(ParentNode.id)
                
                # 先检查是否能提取到 唯一入口出口子图
                listSubGraphNodeID = self.getOneSingleIOMap(dictID_FusedNode, ParentNode.id)
                if listSubGraphNodeID != None:
                    tmpListFusedNode = [dictID_FusedNode[i] for i in listSubGraphNodeID]
                    tmpSubGraph = SUB_GRAPH("SingleIOMap", tmpListFusedNode, self, 1.0, 1.0, False)

                    # 保存子图
                    self.dictSingleIOMap[tmpSubGraph.name] = tmpSubGraph
                    self.dictSingleIOMap[listSubGraphNodeID[0]] = tmpSubGraph
                    
                    # 将子图出口节点加入 setMIMONodeID
                    # 以处理 两个 唯一入口出口子图 前后相继 的情况
                    setMIMONodeID.add(listSubGraphNodeID[-1])
                    # 将子图节点加入 setProcessedNode
                    setProcessedNodeID.update(listSubGraphNodeID[:-1])

                    OutNode = dictID_FusedNode[listSubGraphNodeID[-1]]
                    listBFSNodeNew.append(OutNode)
                    continue

                else:
                    # 更新下次先广搜索buf
                    tmpListFusedNode = [self.dictFusedNode[child_name] for child_name in ParentNode.list_child_name]
                    listBFSNodeNew.extend(tmpListFusedNode)
                    # 更新 setProcessedNodeID
                    setProcessedNodeID.add(ParentNode.id)

                # end if
            # end for

            listBFSNode = list(set(listBFSNodeNew))

        # end while

    # 获得一个 SingleIOMap 中的 FusedNode 的 ID 列表
    def getOneSingleIOMap(self, dictID_FusedNode, InNodeID):
        
        # 如果子节点数量 <= 1 就直接返回
        # 检查出口节点是否在 GPU/DLA 上都能运行
        if not InNodeID in dictID_FusedNode.keys():
            return None # 这里怎么返回值 ???

        # print("getOneSingleIOMap: in: InNode: {}".format(dictID_FusedNode[InNodeID].getName()))

        if len(dictID_FusedNode[InNodeID].list_child_name) <= 1:
            return None # 这里怎么返回值 ???
        
        InNode = dictID_FusedNode[InNodeID]
        if False == InNode.isCanUseDLA:
            return None # 检查入口节点是否在 GPU/DLA 上都能运行

        # 先广搜索 记录 trace
        listBFSTraceID = [] # 先广搜索的各个 trace, 记录几点的 id
        listBFSTraceIDNew = []
        listOutNodeID = []

        # 初始化 listBFSTraceID
        for child_name in InNode.list_child_name:
            ChildNode = self.dictFusedNode[child_name]
            listBFSTraceID.append([ChildNode.id])
        
        countNewChild = len(InNode.list_child_name)

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
                        ChildNode = self.dictFusedNode[child_name]
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
                ParentID = self.dictFusedNode[parent_name].id
                if not ParentID in listSubGraphNodeID:
                    return None

            for child_name in FusedNode.list_child_name:
                ChildID = self.dictFusedNode[child_name].id
                if not ChildID in listSubGraphNodeID:
                    return None

        # 检查出口节点的 所有父节点是否都在子图内
        for parent_name in OutNode.list_parent_name:
            ParentID = self.dictFusedNode[parent_name].id
            if not ParentID in listSubGraphNodeID:
                return None
        
        print("getOneSingleIOMap: InNode: {}".format(dictID_FusedNode[InNodeID].getName()))
        print("getOneSingleIOMap: OutNode: {}".format(dictID_FusedNode[OutNodeID].getName()))
        print("getOneSingleIOMap: listSubGraphNodeID: {}".format(listSubGraphNodeID))
        print("")

        # 经过检查, 至此确定子图只有唯一的入口和出口节点
        return listSubGraphNodeID

    # 拆分构建子模型, 返回 onnx 模型
    # 可选输入: 输入/输出 tensor 维度 的字典
    def createSubOnnxModel(self, list_node_info, model_name="sub_model", dictTensorShape = {}):

        if len(list_node_info) == 0:
            print("createSubOnnxModel: len(list_node_info) = {}".format(len(list_node_info)))
            return None, False

        listSubModelNode = []
        if isinstance(list_node_info[0], int): # 如果输入是 节点 id
            listSubModelNode = [self.onnx_model.graph.node[id] for id in list_node_info]
        elif isinstance(list_node_info[0], str): # 如果输入是 节点名
            listSubModelNode = [self.dictNetworkNode[name].onnx_node for name in list_node_info]
        elif isinstance(list_node_info[0], onnx.onnx_ml_pb2.NodeProto): # 如果输入是 onnx 网络 节点
            listSubModelNode = list_node_info
        else:
            print("ERROR: createSubOnnxModel: input \"list_node_info\" illegal!")

        # listNode = []
        listLocalNetworkNode = []
        listInput = []
        listOutput = []
        listInitializer = []
        listValueInfo = []

        # 需要先找到 输入节点 和 输出节点, 可能有 多个 输入/输出 节点
        # 构建一些局部数据结构
        dictLocalInputNameNode = {}
        dictLocalOutputNameNode = {} # 构建字典, key=输入/输出名, value=network_node
        dictLocalNetworkNode = {} # 构建字典, key=onnx_node名, value=network_node
        for onnx_node in listSubModelNode:
            if not onnx_node.name in self.dictNetworkNode.keys():
                continue
            # if onnx_node.name == "":
            #     onnx_node.name = "unnamed_" + str(self.unnamed_node_id)
            #     self.unnamed_node_id += 1
            dictLocalNetworkNode[onnx_node.name] = network_node(self.dictNetworkNode[onnx_node.name].id, onnx_node)
            for input_name in onnx_node.input:
                dictLocalInputNameNode[input_name] = dictLocalNetworkNode[onnx_node.name]
            for output_name in onnx_node.output:
                dictLocalOutputNameNode[output_name] = dictLocalNetworkNode[onnx_node.name]

        # 将节点组成 list 并排序
        # 按照 id 升序排列, 保证子节点中 onnx_node 相对顺序 与 原模型中 一样
        listLocalNetworkNode = [tmp_node for tmp_node in dictLocalNetworkNode.values()]
        listLocalNetworkNode.sort(key=lambda x: x.id)
        listNode = [NetworkNode.onnx_node for NetworkNode in listLocalNetworkNode]

        # print("dictLocalNetworkNode.keys(): {}".format(dictLocalNetworkNode.keys()))
        # print("\ndictLocalInputNameNode:")
        # print(dictLocalInputNameNode)
        # print("\ndictLocalOutputNameNode:")
        # print(dictLocalOutputNameNode)

        # 为 listSubModelNode/dictLocalNetworkNode 中的 node 找 父/子节点
        # 同时即可 找到 输入节点 和 输出节点, 可能有 多个 输入/输出 节点
        listLocalEntryNode = [] # 入口/出口 节点
        listLocalExitNode = []
        list_local_patch_input = [] # 需要修补的 输入/输出
        list_local_patch_output = []
        for NetworkNode in dictLocalNetworkNode.values():
            tmp_unknown_input_count = 0
            for input_name in NetworkNode.onnx_node.input:
                # 如果 input_name 是子图某个节点的输出
                if input_name in dictLocalOutputNameNode:
                    NetworkNode.list_parent.append(dictLocalOutputNameNode[input_name])
                # 如果 input_name 在 initializer 中, 则 跳过
                elif input_name in self.dictInitializer:
                    continue
                # 如果 input_name 在 value_info 中, 则 存入 list_local_patch_input
                elif input_name in self.dictValueInfo:
                    # 未知输入 +1
                    tmp_unknown_input_count += 1
                    # 到此找到一个需要修补的 输入, 存入 list_local_patch_input
                    # if not input_name in self.dictInput:
                    list_local_patch_input.append(self.dictValueInfo[input_name])
                # 如果 input_name 在 output 中, 则 存入 list_local_patch_input
                elif input_name in self.dictOutput:
                    tmp_unknown_input_count += 1
                    list_local_patch_input.append(self.dictOutput[input_name])

            unknown_output_count = 0
            for output_name in NetworkNode.onnx_node.output:
                # 如果 output_name 是子图某个节点的输入
                if output_name in dictLocalInputNameNode:
                    NetworkNode.list_child.append(dictLocalInputNameNode[output_name])
                elif output_name in self.dictValueInfo:
                    # 若输出不是任何节点的输入, 则未知输出 +1
                    unknown_output_count += 1
                    # 到此找到一个需要修补的 输出, 存入 list_local_patch_output
                    # if not output_name in self.dictOutput:
                    list_local_patch_output.append(self.dictValueInfo[output_name])

            # print("\nNetworkNode.list_parent:")
            # print(NetworkNode.list_parent)
            # print("\nNetworkNode.list_child:")
            # print(NetworkNode.list_child)

            # 判断当前节点是否为 入口/出口 节点, 并保存
            if tmp_unknown_input_count > 0:
                listLocalEntryNode.append(NetworkNode)
            if unknown_output_count > 0:
                listLocalExitNode.append(NetworkNode)

        # print("\nlistLocalEntryNode:")
        # print(listLocalEntryNode)
        # print("\nlistLocalExitNode:")
        # print(listLocalExitNode)
        #
        # print("\nlist_local_patch_input:")
        # print(list_local_patch_input)

        # 修补 子模型的入口 input 参数
        for patch_input in list_local_patch_input:
            if not patch_input in listInput:
                listInput.append(patch_input)

        for onnx_node in listSubModelNode:

            input_name = onnx_node.input
            output_name = onnx_node.output

            tmpListInput = self.getInput(input_name)
            tmpListOutput = self.getOutput(output_name)
            tmpListInitializer = self.getInitializer(input_name)
            tmpArray = np.array([*input_name] + [*output_name])
            value_info_name = np.unique(tmpArray).tolist()
            tmpListValueInfo = self.getValueInfo(value_info_name)

            for tmp_input in tmpListInput:
                if not tmp_input in listInput:
                    listInput.append(tmp_input)
            for tmp_output in tmpListOutput:
                if not tmp_output in listOutput:
                    listOutput.append(tmp_output)
            for tmp_initializer in tmpListInitializer:
                if not tmp_initializer in listInitializer:
                    listInitializer.append(tmp_initializer)
            for tmp_valueInfo in tmpListValueInfo:
                if (not tmp_valueInfo in listValueInfo) \
                    and (not tmp_valueInfo in list_local_patch_input) \
                    and (not tmp_valueInfo in list_local_patch_output):
                    listValueInfo.append(tmp_valueInfo)


        # 修补 子模型的出口 output 参数
        for patch_output in list_local_patch_output:
            if not patch_output in listOutput:
                listOutput.append(patch_output)

        # print("\nlistOutput:")
        # print(listOutput)

        # print("\nlistValueInfo:")
        # print(listValueInfo)
        
        isNeedPatch = False
        # 修补 有 未知维度的 输入
        for i in range(len(listInput)):
            tmp_tensor = listInput[i]
            isUnknownDim = False
            if tmp_tensor.type.tensor_type.HasField("shape") == False:
                isUnknownDim = True
            elif len(tmp_tensor.type.tensor_type.shape.dim) == 0:
                isUnknownDim = True
            else:
                for tmp_dim in tmp_tensor.type.tensor_type.shape.dim:
                    if tmp_dim.dim_value < 1:
                        isUnknownDim = True
                        break
            # dictValueInfo["TFNodes/yolo_evaluation_layer_1/Reshape:0"].type.tensor_type.HasField("shape")
            
            if isUnknownDim == True:
                isNeedPatch = True
                # 修补 有 未知维度的 输入
                if tmp_tensor.name in dictTensorShape.keys():
                    # 新建 value_info
                    # if len(dictTensorShape[tmp_tensor.name]) == 0:
                    #     tmpShape = [1]
                    # else:
                    tmpShape = dictTensorShape[tmp_tensor.name]
                    listInput[i] = onnx.helper.make_tensor_value_info(tmp_tensor.name, tmp_tensor.type.tensor_type.elem_type, tmpShape)

        # 修补 有 未知维度的 输出
        for i in range(len(listOutput)):
            tmp_tensor = listOutput[i]
            isUnknownDim = False
            if tmp_tensor.type.tensor_type.HasField("shape") == False:
                isUnknownDim = True
            elif len(tmp_tensor.type.tensor_type.shape.dim) == 0:
                isUnknownDim = True
            else:
                for tmp_dim in tmp_tensor.type.tensor_type.shape.dim:
                    if tmp_dim.dim_value < 1:
                        isUnknownDim = True
                        break
            # dictValueInfo["TFNodes/yolo_evaluation_layer_1/Reshape:0"].type.tensor_type.HasField("shape")
            
            if isUnknownDim == True:
                isNeedPatch = True
                # 修补 有 未知维度的 输出
                if tmp_tensor.name in dictTensorShape.keys():
                    # 新建 value_info
                    # if len(dictTensorShape[tmp_tensor.name]) == 0:
                    #     tmpShape = [1]
                    # else:
                    tmpShape = dictTensorShape[tmp_tensor.name]
                    listOutput[i] = onnx.helper.make_tensor_value_info(tmp_tensor.name, tmp_tensor.type.tensor_type.elem_type, tmpShape)
                

        sub_graph = onnx.helper.make_graph(
            listNode,
            model_name,
            inputs=listInput, # 输入
            outputs=listOutput, # 输出
            initializer=listInitializer, # initalizer
            value_info=listValueInfo,
        )
        sub_model = onnx.helper.make_model(sub_graph, producer_name='user')
        sub_model.ir_version = self.onnx_model.ir_version
        sub_model.opset_import[0].version = self.onnx_model.opset_import[0].version

        # onnx.checker.check_model(sub_model) # 检查 onnx 模型
        # print(model_name + " 子onnx模型生成完成!")
        
        # tmp_flag = False
        # for onnx_node in listSubModelNode:
        #     pos = onnx_node.name.find("conv2d_1")
        #     if pos >= 0:
        #         tmp_flag = True
        #         break
        # if tmp_flag == True:
        #     onnx.save_model(sub_model, "/home/wfr/work/DLA/ETOpt/tmp.onnx")

        #     print("\nlistNode:")
        #     for onnx_node in listNode:
        #         print("{}".format(onnx_node))

        #     print("\nlistInput:")
        #     for tmp in listInput:
        #         print("{}".format(tmp))

        #     print("\nlistOutput:")
        #     for tmp in listOutput:
        #         print("{}".format(tmp))

        #     print("\nlistValueInfo:")
        #     for tmp in listValueInfo:
        #         print("{}".format(tmp))

        #     print("\nlistInitializer:")
        #     for tmp in listInitializer:
        #         print("{}".format(tmp.name))

        isValidModel = False
        # onnx.checker.check_model(sub_model) # 检查 onnx 模型
        try:
            onnx.checker.check_model(sub_model) # 检查 onnx 模型
            print("{}.onnx generated !".format(model_name), flush=True)
            if isNeedPatch == True:
                isValidModel = False
            else:
                isValidModel = True
        except:
            print("{} check failure !".format(model_name), flush=True)
            isValidModel = False
            # onnx.checker.check_model(sub_model) # 检查 onnx 模型
            # TODO: 这里输出下 输入/输出/value_info 等, 看看是否缺
            # Field 'shape' of 'type' is required but missing.

            # tmp_flag = False
            # for onnx_node in listSubModelNode:
            #     pos = onnx_node.name.find("conv2d_1")
            #     if pos >= 0:
            #         tmp_flag = True
            #         break

            # if tmp_flag == True:
            # #     print("\nlistNode:")
            #     for onnx_node in listSubModelNode:
            #         print("{}".format(onnx_node))

            #     print("\nlistInput:")
            #     for tmp in listInput:
            #         print("{}".format(tmp))

            #     print("\nlistOutput:")
            #     for tmp in listOutput:
            #         print("{}".format(tmp))

            #     print("\nlistValueInfo:")
            #     for tmp in listValueInfo:
            #         print("{}".format(tmp))

            #     print("\nlistInitializer:")
            #     for tmp in listInitializer:
            #         print("{}".format(tmp.name))

            #     onnx.checker.check_model(sub_model) # 检查 onnx 模型
            #     exit(0)
        # print(model_name + " 子onnx模型生成完成!")
                
        # if tmp_flag == True:
        #     exit(0)

        return sub_model, isValidModel

    def generateSubOnnxFile(self, list_node_info, model_name="sub_model", folder_dir = ".", dictTensorShape = {}):
        sub_onnx_model_dir = os.path.join(folder_dir, model_name+".onnx")
        if not os.path.exists(sub_onnx_model_dir):
            SubModel, isValidModel = self.createSubOnnxModel(list_node_info, model_name, dictTensorShape)
            if isValidModel == True:
                onnx.save_model(SubModel, sub_onnx_model_dir)
                print("sub-onnx generation completed: {}".format(sub_onnx_model_dir), flush=True)
                return True
            else:
                print("sub-onnx generation failed: {}".format(sub_onnx_model_dir), flush=True)
                return False
        else:
            print("sub-onnx exists: {}".format(sub_onnx_model_dir), flush=True)
            return True

    # 获取对应输入信息
    def getInput(self, input_name):
        listInput = []
        for name in input_name:
            if name in self.dictInput.keys():
                listInput.append(self.dictInput[name])

        return listInput
    
    # 获取对应输出信息
    def getOutput(self, output_name):
        listOutput = []
        for name in output_name:
            if name in self.dictOutput.keys():
                listOutput.append(self.dictOutput[name])

        return listOutput
    
    # 获取对应超参数值
    def getInitializer(self, input_name):
        listInitializer = []
        for name in input_name:
            if name in self.dictInitializer.keys():
                listInitializer.append(self.dictInitializer[name])

        return listInitializer
    
    # 获取 value_info
    def getValueInfo(self, value_info_name):
        listValueInfo = []
        for name in value_info_name:
            if name in self.dictValueInfo.keys():
                listValueInfo.append(self.dictValueInfo[name])

        return listValueInfo
    
    def getName(self, useless_prefix=""):
        return self.onnx_file_name

class SUB_GRAPH:

    # 新初始化函数能够处理多个输入/输出且融合子节点没有排序好的情况
    # 即只需 融合子节点列表 即可, 无须特定顺序
    # 初始化函数中会自动找 输入/输出, 并进行排序
    # 故而 初始化函数可以用来融合 多个子图
    def __init__(self, str_type, list_fused_node, NetworkMap, io_energy_discount, ratio_time_dla, is_merge = True):
        if len(list_fused_node) <= 0:
            return

        print("\nSUB_GRAPH: is_merge = {}".format(is_merge))
        print("len(list_fused_node) = {}".format(len(list_fused_node)))
        self.NetworkMap = NetworkMap # 整个网络的数据结构
        self.str_type = str_type

        self.list_fused_node = []
        self.list_fused_node_id = [] # 当前子图中的融合节点 id
        self.set_fused_node_id = set()
        self.list_parent_fused_node_id = [] # 当前子图的父节点 id
        self.set_parent_fused_node_id = set()
        self.list_child_fused_node_id = [] # 当前子图的子节点 id
        self.set_child_fused_node_id = set()

        self.list_entry_fused_node = [] # 当前子图的入口节点
        self.list_exit_fused_node = [] # 当前子图的出口节点
        self.list_entry_network_node = [] # 当前子图的入口节点
        self.list_exit_network_node = [] # 当前子图的出口节点

        self.isCanUseDLA = True
        self.isCanUseGPU = True

        self.list_network_node = []
        # 先将 onnx_node 取出 升序排列
        for fused_node in list_fused_node:
            for onnx_node in fused_node.list_onnx_node:
                self.list_network_node.append(self.NetworkMap.dictNetworkNode[onnx_node.name])
        self.list_network_node.sort(key=lambda x: x.id)
        self.list_network_node_id = [network_node.id for network_node in self.list_network_node]

        self.name = self.list_network_node[0].getName(self.NetworkMap.useless_prefix)
        self.name += "..."
        self.name += self.list_network_node[-1].getName(self.NetworkMap.useless_prefix)

        if is_merge == True:
            # 再按 onnx_node 升序排列
            setFusedNodeID = set()
            for NetworkNode in self.list_network_node:
                NodeName = NetworkNode.onnx_node.name
                # 通过节点名 来找 融合节点
                tmpFusedNode = self.NetworkMap.dictFusedNode[NodeName]
                tmpFusedNodeID = tmpFusedNode.id
                # 通过 id 来判断 融合节点 是否重复
                if tmpFusedNodeID in setFusedNodeID:
                    continue # 跳过处理过的 融合节点
                setFusedNodeID.add(tmpFusedNodeID)
                self.list_fused_node.append(tmpFusedNode)
            # 完成对 self.list_fused_node 的重排序

            # SubModelName = self.list_fused_node[0].getName(self.NetworkMap.useless_prefix) + "..." + self.list_fused_node[-1].getName(self.NetworkMap.useless_prefix)
            # sub_onnx_file_dir = os.path.join(self.NetworkMap.onnx_folder_dir, SubModelName + ".onnx")

            # SubOnnxModel, isValidModel = self.NetworkMap.createSubOnnxModel(list_onnx_node, SubModelName)
            # onnx.save_model(SubOnnxModel, sub_onnx_file_dir)
            # self.SubNetworkMap = network_type.network_map(sub_onnx_file_dir, [], self.NetworkMap.useless_prefix)

            # 当前子图中的融合节点 id
            for tmpFusedNode in self.list_fused_node:
                self.set_fused_node_id.add(tmpFusedNode.id)
                self.list_fused_node_id.append(tmpFusedNode.id)
            # 当前子图的父节点 id

            # 遍历所有节点的父/子节点
            # 父/子节点如果不在 self.list_fused_node, 则找到 子图的 入口/出口节点
            # 保存 子图的 入口/出口节点 及 子图的 父/子节点

            for tmpFusedNode in self.list_fused_node:
                isEntry = False
                for ParentName in tmpFusedNode.list_parent_name:
                    ParentFusedNode = self.NetworkMap.dictFusedNode[ParentName]
                    ParentID = ParentFusedNode.id
                    # 如果 父节点 不在 子图中
                    if not ParentID in self.list_fused_node_id:
                        isEntry =  True # 判断是 子图 入口节点
                        self.set_parent_fused_node_id.add(ParentID)
                if isEntry == True or len(tmpFusedNode.list_parent_name) == 0:
                    self.list_entry_fused_node.append(tmpFusedNode)

                isExit = False
                for ChildName in tmpFusedNode.list_child_name:
                    ChildFusedNode = self.NetworkMap.dictFusedNode[ChildName]
                    ChildID = ChildFusedNode.id
                    # 如果 子节点 不在 子图中
                    if not ChildID in self.list_fused_node_id:
                        isExit = True # 判断是 子图 出口节点
                        self.set_child_fused_node_id.add(ChildID)
                if isExit == True or len(tmpFusedNode.list_child_name) == 0:
                    self.list_exit_fused_node.append(tmpFusedNode)

            self.list_parent_fused_node_id = list(self.set_parent_fused_node_id)
            self.list_parent_fused_node_id.sort()
            self.list_child_fused_node_id = list(self.set_child_fused_node_id)
            self.list_child_fused_node_id.sort()

        else:
            self.list_fused_node = list_fused_node
            self.list_fused_node_id = [tmpFusedNode.id for tmpFusedNode in self.list_fused_node]
            self.set_fused_node_id = set(self.list_fused_node_id)
            self.list_entry_fused_node = [self.list_fused_node[0]]
            self.list_exit_fused_node = [self.list_fused_node[-1]]

            for NodeName in self.list_fused_node[0].list_parent_name:
                tmpFusedNode = self.NetworkMap.dictFusedNode[NodeName]
                self.set_parent_fused_node_id.add(tmpFusedNode.id)
            self.list_parent_fused_node_id = list(self.set_parent_fused_node_id)
            self.list_parent_fused_node_id.sort()

            for NodeName in self.list_fused_node[-1].list_child_name:
                tmpFusedNode = self.NetworkMap.dictFusedNode[NodeName]
                self.set_child_fused_node_id.add(tmpFusedNode.id)
            self.list_child_fused_node_id = list(self.set_child_fused_node_id)
            self.list_child_fused_node_id.sort()
        
        print("len(self.list_fused_node) = {}".format(len(self.list_fused_node)))
        print("SUB_GRAPH: first FusedNode: {}".format(self.list_fused_node[0].getName(self.NetworkMap.useless_prefix)))
        print("SUB_GRAPH: last FusedNode: {}".format(self.list_fused_node[-1].getName(self.NetworkMap.useless_prefix)))

        # print("SUB_GRAPH: list_parent_fused_node_id: {}".format(self.list_parent_fused_node_id))
        # print("SUB_GRAPH: list_fused_node_id: {}".format(self.list_fused_node_id))
        # print("SUB_GRAPH: list_child_fused_node_id: {}".format(self.list_child_fused_node_id))

        for network_node in self.list_network_node:
            isEntry = False
            for ParentNetworkNode in network_node.list_parent:
                ParentID = ParentNetworkNode.id
                # 如果 父节点 不在 子图中
                if not ParentID in self.list_network_node_id:
                    isEntry =  True # 判断是 子图 入口节点
            if isEntry == True or len(network_node.list_parent) == 0:
                self.list_entry_network_node.append(network_node)

            isExit = False
            for ChildNetworkNode in network_node.list_child:
                ChildID = ChildNetworkNode.id
                # 如果 子节点 不在 子图中
                if not ChildID in self.list_network_node_id:
                    isExit = True # 判断是 子图 出口节点
            if isExit == True or len(network_node.list_child) == 0:
                self.list_exit_network_node.append(network_node)

        print("SUB_GRAPH: list_network_node_id = {}".format(self.list_network_node_id), flush=True)

        # # 这里先广搜索确定包含多少个独立子图
        # tmpListNetworkNodeID = copy.copy(self.list_network_node_id)
        # dictID_NetworkNode = {}
        # for tmpNetworkNode in self.list_network_node:
        #     dictID_NetworkNode[tmpNetworkNode.id] = tmpNetworkNode

        # self.list_independent_subgraph_onnx_ids = []
        # i = 0
        # while len(tmpListNetworkNodeID) > 0:
        #     tmpListONNXIDs = []
        #     tmpID = tmpListNetworkNodeID.pop(0)
        #     tmpListONNXIDs.append(tmpID)

        #     listBFSID = [tmpID]
        #     while len(listBFSID) > 0:
                
        #         listBFSIDNew = []
        #         for tmpID in listBFSID:
        #             tmpNetworkNode = dictID_NetworkNode[tmpID]
        #             for parentNetworkNode in tmpNetworkNode.list_parent:
        #                 parentID = parentNetworkNode.id
        #                 if not parentID in tmpListNetworkNodeID:
        #                     continue
        #                 tmpListONNXIDs.append(parentID)
        #                 listBFSIDNew.append(parentID)
        #                 tmpListNetworkNodeID.remove(parentID)
        #             for childNetworkNode in tmpNetworkNode.list_child:
        #                 childID = childNetworkNode.id
        #                 if not childID in tmpListNetworkNodeID:
        #                     continue
        #                 tmpListONNXIDs.append(childID)
        #                 listBFSIDNew.append(childID)
        #                 tmpListNetworkNodeID.remove(childID)

        #             # end for 
        #         # end for
        #         listBFSID = listBFSIDNew
        #     # end while
        #     self.list_independent_subgraph_onnx_ids.append(tmpListONNXIDs)
        # # end while
        # print("SUB_GRAPH: number of independent subgraphs: {}".format(len(self.list_independent_subgraph_onnx_ids)), flush=True)

        self.gpu_profiling = profiling_data() # 当前网络节点运行在 GPU 上测得的数据
        self.dla_profiling = profiling_data() # 当前网络节点运行在 DLA 上测得的数据

        self.io_energy_discount = io_energy_discount
        self.ratio_time_dla = ratio_time_dla
        self.NumDLA = NetworkMap.NumDLA
        self.NumGPU = NetworkMap.NumGPU
        self.useless_prefix = NetworkMap.useless_prefix

        self.idle_power = 0.0 # 静态功率
        self.gpu_time = 0.0 # 使用 GPU 时 运行时间
        self.gpu_energy = 0.0 # 使用 GPU 时 总能耗

        self.dla_time = 0.0 # 使用 DLA 时 运行时间
        self.dla_energy = 0.0 # 使用 DLA 时 总能耗
        self.bubble_time = 0.0 # 灵活 GPU 时间使用量
        self.in4dla_time = 0.0 # 一个 GPU stream 中为 DLA 准备 输入/输出 时间加和
        self.out4dla_time = 0.0 # 一个 GPU stream 中为 DLA 准备 输入/输出 时间加和
        self.io4dla_time = 0.0 # 一个 GPU stream 中为 DLA 准备 输入/输出 时间加和
        self.time_dla_node = 0.0 # 使用 DLA 时 一次推理 DLA 所有 node 运行时间加和
        self.energy_saving = -1e9

        self.energy_saving_with_dla = 0.0
        self.energy_with_dla = 0.0
        self.exe_time_with_dla = 0.0
        self.bubble_time_used = 0.0

        # 估计 / 推导 使用 DLA / GPU 时的 时间 和 能耗
        # 多个 GPU/DLA核 并行运行, 有 DLA node 运行完成时, 优先处理输出
        # 不支持抢占
        # 假设有足够多的缓冲进行上述操作, 最坏情况需要 NumDLA * NumGPU 个输入和输出缓冲单元
        self.num_inferences = self.NumDLA * self.NumGPU

        # 对 gpu 也考虑 input/output
        tmpFusedNode = self.list_fused_node[0]
        self.idle_power = tmpFusedNode.gpu_profiling.avg_power - tmpFusedNode.gpu_profiling.power_dynamic

        self.accumulateFusedNodeData(False)
        self.calculateBubbleTime()

        return
    
    def getName(self, useless_prefix=""):
        return self.name

    def accumulateFusedNodeData(self, useIntegratedMeasurementData = False):

        self.gpu_time = 0.0 # 使用 GPU 时 运行时间
        self.gpu_energy = 0.0 # 使用 GPU 时 总能耗

        self.dla_time = 0.0 # 使用 DLA 时 运行时间
        self.dla_energy = 0.0 # 使用 DLA 时 总能耗
        self.bubble_time = 0.0 # 灵活 GPU 时间使用量
        self.in4dla_time = 0.0 # 一个 GPU stream 中为 DLA 准备 输入/输出 时间加和
        self.out4dla_time = 0.0 # 一个 GPU stream 中为 DLA 准备 输入/输出 时间加和
        self.io4dla_time = 0.0 # 一个 GPU stream 中为 DLA 准备 输入/输出 时间加和
        self.time_dla_node = 0.0 # 使用 DLA 时 一次推理 DLA 所有 node 运行时间加和
        self.energy_saving = -1e9

        self.energy_saving_with_dla = 0.0
        self.energy_with_dla = 0.0
        self.exe_time_with_dla = 0.0
        self.bubble_time_used = 0.0

        print("accumulateFusedNodeData: useIntegratedMeasurementData = {}".format(useIntegratedMeasurementData))
        if useIntegratedMeasurementData == False:
            
            print("self.list_entry_fused_node:")
            for tmpFusedNode in self.list_entry_fused_node:
                print("{}".format(tmpFusedNode.getName(self.NetworkMap.useless_prefix)))
            print("self.list_exit_fused_node:")
            for tmpFusedNode in self.list_exit_fused_node:
                print("{}".format(tmpFusedNode.getName(self.NetworkMap.useless_prefix)))
            print("")

            # 这里循环处理输入节点
            for tmpFusedNode in self.list_entry_fused_node:
                # 加 gpu 输入 时间
                self.gpu_time += self.NumDLA * tmpFusedNode.gpu_profiling.avg_input_time
                # 加 gpu 输入 能耗
                self.gpu_energy += (self.NumGPU * tmpFusedNode.gpu_profiling.power_dynamic * self.io_energy_discount + self.idle_power) * self.NumDLA * tmpFusedNode.gpu_profiling.avg_input_time
                # 加 dla 输入 动态能耗
                self.dla_energy += self.num_inferences * tmpFusedNode.dla_profiling.avg_input_time * tmpFusedNode.dla_profiling.power_dynamic * self.io_energy_discount
                # 加 为 DLA 准备 输入/输出 的时间
                self.in4dla_time += tmpFusedNode.dla_profiling.avg_input_time
                self.io4dla_time += tmpFusedNode.dla_profiling.avg_input_time
            # end for

            # 这里循环处理输出节点
            for tmpFusedNode in self.list_exit_fused_node:
                # 加 gpu 输出 时间
                self.gpu_time += self.NumDLA * tmpFusedNode.gpu_profiling.avg_output_time
                # 加 gpu 输出 能耗
                self.gpu_energy += (self.NumGPU * tmpFusedNode.gpu_profiling.power_dynamic * self.io_energy_discount + self.idle_power) * self.NumDLA * tmpFusedNode.gpu_profiling.avg_output_time
                # 加 dla 输出 动态能耗
                self.dla_energy += self.num_inferences * tmpFusedNode.dla_profiling.avg_output_time * tmpFusedNode.dla_profiling.power_dynamic * self.io_energy_discount
                # 加 为 DLA 准备 输入/输出 的时间
                self.out4dla_time += tmpFusedNode.dla_profiling.avg_output_time
                self.io4dla_time += tmpFusedNode.dla_profiling.avg_output_time
            # end for

            for tmpFusedNode in self.list_fused_node:
                # 加 各个 节点 gpu 执行时间
                self.gpu_time += self.NumDLA * tmpFusedNode.gpu_profiling.avg_exe_time
                # 加 各个 节点 gpu 能耗
                self.gpu_energy += self.num_inferences * tmpFusedNode.gpu_profiling.energy_dynamic + self.NumDLA * tmpFusedNode.gpu_profiling.energy_static

                # 加 使用 DLA 时 DLA node 运行时间
                self.time_dla_node += tmpFusedNode.dla_profiling.avg_exe_time
                # 加 各个 节点 dla 动态能耗
                self.dla_energy += self.num_inferences * tmpFusedNode.dla_profiling.energy_dynamic
            # end for

        else:

            print("accumulateFusedNodeData: idle_power = {:.4e}".format(self.idle_power))
            
            print("accumulateFusedNodeData: gpu_profiling.avg_input_time = {:.4e}".format(self.gpu_profiling.avg_input_time))
            print("accumulateFusedNodeData: gpu_profiling.avg_exe_time = {:.4e}".format(self.gpu_profiling.avg_exe_time))
            print("accumulateFusedNodeData: gpu_profiling.avg_output_time = {:.4e}".format(self.gpu_profiling.avg_output_time))

            print("accumulateFusedNodeData: gpu_profiling.avg_energy = {:.4e}".format(self.gpu_profiling.avg_energy))
            print("accumulateFusedNodeData: gpu_profiling.energy_dynamic = {:.4e}".format(self.gpu_profiling.energy_dynamic))
            print("accumulateFusedNodeData: gpu_profiling.energy_static = {:.4e}".format(self.gpu_profiling.energy_static))
            print("accumulateFusedNodeData: gpu_profiling.power_dynamic = {:.4e}".format(self.gpu_profiling.power_dynamic))
            

            print("accumulateFusedNodeData: dla_profiling.avg_input_time = {:.4e}".format(self.dla_profiling.avg_input_time))
            print("accumulateFusedNodeData: dla_profiling.avg_exe_time = {:.4e}".format(self.dla_profiling.avg_exe_time))
            print("accumulateFusedNodeData: dla_profiling.avg_output_time = {:.4e}".format(self.dla_profiling.avg_output_time))

            print("accumulateFusedNodeData: dla_profiling.avg_energy = {:.4e}".format(self.dla_profiling.avg_energy))
            print("accumulateFusedNodeData: dla_profiling.energy_dynamic = {:.4e}".format(self.dla_profiling.energy_dynamic))
            print("accumulateFusedNodeData: dla_profiling.energy_static = {:.4e}".format(self.dla_profiling.energy_static))
            print("accumulateFusedNodeData: dla_profiling.power_dynamic = {:.4e}".format(self.dla_profiling.power_dynamic))

            # 加 gpu 时间
            self.gpu_time += self.NumDLA * (self.gpu_profiling.avg_input_time + self.gpu_profiling.avg_exe_time + self.gpu_profiling.avg_output_time)
            # 加 gpu 能耗
            self.gpu_energy += (self.NumGPU * self.gpu_profiling.power_dynamic * self.io_energy_discount + self.idle_power) * self.NumDLA * (self.gpu_profiling.avg_input_time + self.gpu_profiling.avg_output_time) + self.num_inferences * self.gpu_profiling.energy_dynamic + self.NumDLA * self.gpu_profiling.energy_static

            # 加 dla 动态能耗
            self.dla_energy += self.num_inferences * self.io_energy_discount * self.dla_profiling.power_dynamic * (self.dla_profiling.avg_input_time + self.dla_profiling.avg_output_time) + self.num_inferences * self.dla_profiling.energy_dynamic
            # 加 使用 DLA 时 DLA node 运行时间
            self.time_dla_node += self.dla_profiling.avg_exe_time

            # 加 为 DLA 准备 输入/输出 的时间
            self.in4dla_time += self.dla_profiling.avg_input_time
            self.io4dla_time += self.dla_profiling.avg_input_time
            self.out4dla_time += self.dla_profiling.avg_output_time
            self.io4dla_time += self.dla_profiling.avg_output_time
        # end if

    def calculateBubbleTime(self):
        class TMP_TASK:
            def __init__(self, exe_time, id_task) -> None:
                self.exe_time = exe_time
                self.id_task = id_task
                self.id_stream = -1
                self.time_start = -1.0
                self.time_end = -1.0

        print("calculateBubbleTime: in4dla_time = {}".format(self.in4dla_time))
        print("calculateBubbleTime: time_dla_node = {}".format(self.time_dla_node))
        print("calculateBubbleTime: out4dla_time = {}".format(self.out4dla_time))

        listInputTask = []
        listDLATask = []
        listOutputTask = []
        for i in range(self.num_inferences):

            listInputTask.append(TMP_TASK(self.in4dla_time, i))

            listDLATask.append(TMP_TASK(self.time_dla_node, i))

            listOutputTask.append(TMP_TASK(self.out4dla_time, i))

        dla_stream_time = np.zeros(self.NumDLA) # dla stream idle 时刻
        gpu_stream_time = np.zeros(self.NumGPU) # gpu stream idle 时刻

        done_count = 0
        while done_count < self.num_inferences:

            # 检查 gpu stream 是否可以执行 output
            for i in range(self.num_inferences):
                if listOutputTask[i].time_end >= 0:
                    continue # 当前 output 已经处理完
                id_task = listOutputTask[i].id_task
                if listDLATask[id_task].time_end < 0:
                    continue # 依赖的前驱 dla 任务还没完成
                # 找最早的 gpu stream
                idx_gpu_early = gpu_stream_time.argmin()
                time_gpu_early = gpu_stream_time[idx_gpu_early]
                # 推导 output 开始/完成 时间
                tmp_time_start = max(time_gpu_early, listDLATask[id_task].time_end)
                tmp_time_end = tmp_time_start + listOutputTask[i].exe_time
                # 写入当前 output 任务信息
                listOutputTask[i].id_stream = idx_gpu_early
                listOutputTask[i].time_start = tmp_time_start
                listOutputTask[i].time_end = tmp_time_end
                gpu_stream_time[idx_gpu_early] = tmp_time_end
                # 任务完成计数
                done_count += 1
                if done_count == self.num_inferences:
                    break
            # end for
            if done_count == self.num_inferences:
                break
            # 检查 dla stream 是否可以执行 dla node
            for i in range(self.num_inferences):
                if listDLATask[i].time_end >= 0:
                    continue # 当前 dla node 已经处理完
                id_task = listDLATask[i].id_task
                if listInputTask[id_task].time_end < 0:
                    continue # 依赖的前驱 input 任务还没完成
                # 找最早的 dla stream
                idx_dla_early = dla_stream_time.argmin()
                time_dla_early = dla_stream_time[idx_dla_early]
                # 推导 dla node 开始/完成 时间
                tmp_time_start = max(time_dla_early, listInputTask[id_task].time_end)
                tmp_time_end = tmp_time_start + listDLATask[i].exe_time
                # 写入当前 dla node 任务信息
                listDLATask[i].id_stream = idx_dla_early
                listDLATask[i].time_start = tmp_time_start
                listDLATask[i].time_end = tmp_time_end
                dla_stream_time[idx_dla_early] = tmp_time_end
            # end for
            # 检查 gpu stream 是否可以执行 input
            for i in range(self.num_inferences):
                if listInputTask[i].time_end >= 0:
                    continue # 当前 input 已经处理完
                id_task = listInputTask[i].id_task
                # 找最早的 gpu stream
                idx_gpu_early = gpu_stream_time.argmin()
                time_gpu_early = gpu_stream_time[idx_gpu_early]
                # 推导 input 开始/完成 时间
                tmp_time_start = time_gpu_early
                tmp_time_end = tmp_time_start + listInputTask[i].exe_time
                # 写入当前 input 任务信息
                listInputTask[i].id_stream = idx_gpu_early
                listInputTask[i].time_start = tmp_time_start
                listInputTask[i].time_end = tmp_time_end
                gpu_stream_time[idx_gpu_early] = tmp_time_end
            # end for
        # end while

        # 得到使用 DLA 后的执行时间
        self.dla_time = gpu_stream_time.max()
        self.bubble_time = gpu_stream_time.mean() - (self.NumDLA * self.io4dla_time)
        self.dla_energy += self.dla_time * self.idle_power

        self.energy_saving = self.gpu_energy - self.dla_energy

        print("SUB_GRAPH: str_type = {}".format(self.str_type))
        print("SUB_GRAPH: gpu_time = {:.4e} s".format(self.gpu_time))
        print("SUB_GRAPH: gpu_energy = {:.4e} J".format(self.gpu_energy))
        print("SUB_GRAPH: dla_time = {:.4e} s".format(self.dla_time))
        print("SUB_GRAPH: dla_energy = {:.4e} J".format(self.dla_energy))
        print("SUB_GRAPH: bubble_time = {:.4e} s".format(self.bubble_time))
        print("SUB_GRAPH: energy_saving = {:.4e} J".format(self.energy_saving))

    # 使用 bubble time 时 也节省了相应的静态功耗, 因此需要根据实际使用的 bubble time 的数量 计算 能耗 和 时间
    # 输入: bubble time 利用率, 整体执行时间, 剩余灵活可并行时间
    # 返回: 能耗节省, 当前任务集使用DLA后整体执行时间, 使用部分 bubble time 后 更新的剩余灵活可并行时间
    def getEnergyTimeWithDLA(self, ratio_bubble_time, exe_time, remainingFlexibleTime, currEnergy):

        print("getEnergyTimeWithDLA: exe_time = {:.4e} s".format(exe_time))
        print("getEnergyTimeWithDLA: remainingFlexibleTime = {:.4e} s".format(remainingFlexibleTime))
        print("getEnergyTimeWithDLA: currEnergy = {:.4e} s".format(currEnergy))
        
        self.energy_saving_with_dla = 0.0
        self.energy_with_dla = currEnergy
        self.exe_time_with_dla = exe_time
        self.bubble_time_used = 0.0

        # 若 剩余的灵活可并行时间 小于 当前节点的 gpu_time
        # 则说明 bubble 时间 不够用了, 需要将之前使用的 bubble 时间 进行回退
        if remainingFlexibleTime < self.gpu_time:

            bubble_time_roll_back = self.gpu_time - remainingFlexibleTime
            print("getEnergyTimeWithDLA: bubble_time_roll_back = {:.4e} s".format(bubble_time_roll_back))
            remainingFlexibleTime = 0.0
            self.bubble_time_used = 0.0

            self.energy_saving_with_dla = self.energy_saving - self.idle_power * bubble_time_roll_back

            self.exe_time_with_dla = exe_time - self.gpu_time + self.dla_time
            self.energy_with_dla -= self.energy_saving_with_dla
        
        else:
            remainingFlexibleTime -= self.gpu_time
            # 实际使用的 bubble 时间, 是二者中最小值: 可用灵活时间, 可用 bubble 时间
            self.bubble_time_used = min(remainingFlexibleTime, ratio_bubble_time * self.bubble_time)

            self.energy_saving_with_dla = self.energy_saving + self.idle_power * self.bubble_time_used
            self.exe_time_with_dla = exe_time - self.gpu_time + self.dla_time - self.bubble_time_used
            remainingFlexibleTime -= self.bubble_time_used
            self.energy_with_dla -= self.energy_saving_with_dla

        print("getEnergyTimeWithDLA: bubble_time_used = {:.4e} s".format(self.bubble_time_used))
        print("getEnergyTimeWithDLA: energy_saving_with_dla = {:.4e} J".format(self.energy_saving_with_dla))
        print("getEnergyTimeWithDLA: exe_time_with_dla = {:.4e} s".format(self.exe_time_with_dla))
        print("getEnergyTimeWithDLA: remainingFlexibleTime = {:.4e} s".format(remainingFlexibleTime))
        return self.energy_saving_with_dla, self.exe_time_with_dla, remainingFlexibleTime, self.energy_with_dla

    def getBestEnergyEfficienySubGraph(self, ratio_bubble_time, exe_time, remainingFlexibleTime, currEnergy):

        maxEnergySavingInSubGraph = -1e9
        minTimeInSubGraph = exe_time
        bestRemainingFlexibleTime = remainingFlexibleTime
        # BestWindowInSubGraph = None
        for i in range(len(self.list_fused_node)):
            for j in range(len(self.list_fused_node)-i):
                tmpSubGraph = SUB_GRAPH(self.str_type, self.list_fused_node[i:i+j+1], self.NetworkMap, self.io_energy_discount, ratio_bubble_time, False)

                tmpEnergySaving, tmpMinTime, remainingFlexibleTime, tmpMinEnergy = tmpSubGraph.getEnergyTimeWithDLA(ratio_bubble_time, exe_time, remainingFlexibleTime, currEnergy)

                if tmpEnergySaving > maxEnergySavingInSubGraph \
                    and tmpMinTime <= minTimeInSubGraph:
                    maxEnergySavingInSubGraph = tmpEnergySaving
                    minTimeInSubGraph = tmpMinTime
                    bestRemainingFlexibleTime = remainingFlexibleTime
                    # BestWindowInSubGraph = tmpSubGraph

        return maxEnergySavingInSubGraph, minTimeInSubGraph, bestRemainingFlexibleTime


# 将 listSubGraph 中 子图 合并 创建 新的融合的SubGraph
# 返回 新的融合的SubGraph
def createMergedSubGraph(listSubGraph):

    print("\ncreateMergedSubGraph: in", flush=True)
    if len(listSubGraph) == 1:
        print("createMergedSubGraph: len(listSubGraph) = {}".format(len(listSubGraph)))
        return listSubGraph[0]
    elif len(listSubGraph) == 0:
        print("createMergedSubGraph: len(listSubGraph) = {}".format(len(listSubGraph)))
        return None

    tmpSetFusedNode = set()
    for tmpSubGraph in listSubGraph:
        tmpSetFusedNode |= set(tmpSubGraph.list_fused_node)
    
    SubGraph0 = listSubGraph[0]
    tmpListFusedNode = list(tmpSetFusedNode)
    newSubGraph = SUB_GRAPH("MultiIOMap", tmpListFusedNode, SubGraph0.NetworkMap, SubGraph0.io_energy_discount, SubGraph0.ratio_time_dla, True)

    newSubGraph.idle_power = 0.0 # 静态功率
    newSubGraph.gpu_time = 0.0 # 使用 GPU 时 运行时间
    newSubGraph.gpu_energy = 0.0 # 使用 GPU 时 总能耗

    newSubGraph.dla_time = 0.0 # 使用 DLA 时 运行时间
    newSubGraph.dla_energy = 0.0 # 使用 DLA 时 总能耗
    newSubGraph.bubble_time = 0.0 # 灵活 GPU 时间使用量
    newSubGraph.in4dla_time = 0.0 # 一个 GPU stream 中为 DLA 准备 输入/输出 时间加和
    newSubGraph.out4dla_time = 0.0 # 一个 GPU stream 中为 DLA 准备 输入/输出 时间加和
    newSubGraph.io4dla_time = 0.0 # 一个 GPU stream 中为 DLA 准备 输入/输出 时间加和
    newSubGraph.time_dla_node = 0.0 # 使用 DLA 时 一次推理 DLA 所有 node 运行时间加和
    newSubGraph.energy_saving = -1e9

    newSubGraph.energy_saving_with_dla = 0.0
    newSubGraph.energy_with_dla = 0.0
    newSubGraph.exe_time_with_dla = 0.0
    newSubGraph.bubble_time_used = 0.0

    print("createMergedSubGraph: list_entry_fused_node:")
    for tmpFusedNode in newSubGraph.list_entry_fused_node:
        print("{}".format(tmpFusedNode.getName(newSubGraph.NetworkMap.useless_prefix)))
    print("createMergedSubGraph: list_exit_fused_node:")
    for tmpFusedNode in newSubGraph.list_exit_fused_node:
        print("{}".format(tmpFusedNode.getName(newSubGraph.NetworkMap.useless_prefix)))
    print("")

    # 这里循环处理输入节点
    for tmpFusedNode in newSubGraph.list_entry_fused_node:
        # print("createMergedSubGraph: input FusedNode: {}".format(tmpFusedNode.getName(newSubGraph.NetworkMap.useless_prefix)))
        # 加 gpu 输入 时间
        newSubGraph.gpu_time += newSubGraph.NumDLA * tmpFusedNode.gpu_profiling.avg_input_time
        # 加 gpu 输入 能耗
        newSubGraph.gpu_energy += (newSubGraph.NumGPU * tmpFusedNode.gpu_profiling.power_dynamic * newSubGraph.io_energy_discount + newSubGraph.idle_power) * newSubGraph.NumDLA * tmpFusedNode.gpu_profiling.avg_input_time
        # 加 dla 输入 动态能耗
        newSubGraph.dla_energy += newSubGraph.num_inferences * tmpFusedNode.dla_profiling.avg_input_time * tmpFusedNode.dla_profiling.power_dynamic * newSubGraph.io_energy_discount
        # 加 为 DLA 准备 输入/输出 的时间
        newSubGraph.in4dla_time += tmpFusedNode.dla_profiling.avg_input_time
        newSubGraph.io4dla_time += tmpFusedNode.dla_profiling.avg_input_time

        # print("createMergedSubGraph: input gpu energy: {}".format((newSubGraph.NumGPU * tmpFusedNode.gpu_profiling.power_dynamic * newSubGraph.io_energy_discount + newSubGraph.idle_power) * newSubGraph.NumDLA * tmpFusedNode.gpu_profiling.avg_input_time))
        # print("createMergedSubGraph: input dla energy: {}".format(newSubGraph.num_inferences * tmpFusedNode.dla_profiling.avg_input_time * tmpFusedNode.dla_profiling.power_dynamic * newSubGraph.io_energy_discount))
    # end for

    # 这里循环处理输出节点
    for tmpFusedNode in newSubGraph.list_exit_fused_node:
        # print("createMergedSubGraph: output FusedNode: {}".format(tmpFusedNode.getName(newSubGraph.NetworkMap.useless_prefix)))
        # 加 gpu 输出 时间
        newSubGraph.gpu_time += newSubGraph.NumDLA * tmpFusedNode.gpu_profiling.avg_output_time
        # 加 gpu 输出 能耗
        newSubGraph.gpu_energy += (newSubGraph.NumGPU * tmpFusedNode.gpu_profiling.power_dynamic * newSubGraph.io_energy_discount + newSubGraph.idle_power) * newSubGraph.NumDLA * tmpFusedNode.gpu_profiling.avg_output_time
        # 加 dla 输出 动态能耗
        newSubGraph.dla_energy += newSubGraph.num_inferences * tmpFusedNode.dla_profiling.avg_output_time * tmpFusedNode.dla_profiling.power_dynamic * newSubGraph.io_energy_discount
        # 加 为 DLA 准备 输入/输出 的时间
        newSubGraph.out4dla_time += tmpFusedNode.dla_profiling.avg_output_time
        newSubGraph.io4dla_time += tmpFusedNode.dla_profiling.avg_output_time

        # print("createMergedSubGraph: output gpu energy: {}".format((newSubGraph.NumGPU * tmpFusedNode.gpu_profiling.power_dynamic * newSubGraph.io_energy_discount + newSubGraph.idle_power) * newSubGraph.NumDLA * tmpFusedNode.gpu_profiling.avg_output_time))
        # print("createMergedSubGraph: output dla energy: {}".format(newSubGraph.num_inferences * tmpFusedNode.dla_profiling.avg_output_time * tmpFusedNode.dla_profiling.power_dynamic * newSubGraph.io_energy_discount))
    # end for

    # 循环累加 各个子图 的数据
    for tmpSubGraph in listSubGraph:
        if tmpSubGraph.gpu_profiling.avg_exe_time > 0.0:
            # 加 gpu 时间
            newSubGraph.gpu_time += tmpSubGraph.NumDLA * tmpSubGraph.gpu_profiling.avg_exe_time
            # 加 gpu 能耗
            newSubGraph.gpu_energy += tmpSubGraph.num_inferences * tmpSubGraph.gpu_profiling.energy_dynamic + tmpSubGraph.NumDLA * tmpSubGraph.gpu_profiling.energy_static

            # 加 dla 动态能耗
            newSubGraph.dla_energy += tmpSubGraph.num_inferences * tmpSubGraph.dla_profiling.energy_dynamic
            # 加 使用 DLA 时 DLA node 运行时间
            newSubGraph.time_dla_node += tmpSubGraph.dla_profiling.avg_exe_time
            print("createMergedSubGraph: tmpSubGraph.dla_profiling.avg_exe_time = {}\n".format(tmpSubGraph.dla_profiling.avg_exe_time), flush=True)
            # print("createMergedSubGraph: tmpSubGraph gpu_energy: {}".format(tmpSubGraph.num_inferences * tmpSubGraph.gpu_profiling.energy_dynamic + tmpSubGraph.NumDLA * tmpSubGraph.gpu_profiling.energy_static))
            # print("createMergedSubGraph: tmpSubGraph dla_energy: {}".format(tmpSubGraph.num_inferences * tmpSubGraph.dla_profiling.energy_dynamic))
        
        else:
            for tmpFusedNode in tmpSubGraph.list_fused_node:
                # 加 各个 节点 gpu 执行时间
                newSubGraph.gpu_time += tmpSubGraph.NumDLA * tmpFusedNode.gpu_profiling.avg_exe_time
                # 加 各个 节点 gpu 能耗
                newSubGraph.gpu_energy += tmpSubGraph.num_inferences * tmpFusedNode.gpu_profiling.energy_dynamic + tmpSubGraph.NumDLA * tmpFusedNode.gpu_profiling.energy_static

                # 加 使用 DLA 时 DLA node 运行时间
                newSubGraph.time_dla_node += tmpFusedNode.dla_profiling.avg_exe_time
                # 加 各个 节点 dla 动态能耗
                newSubGraph.dla_energy += tmpSubGraph.num_inferences * tmpFusedNode.dla_profiling.energy_dynamic
                
                print("createMergedSubGraph: tmpFusedNode.dla_profiling.avg_exe_time = {}\n".format(tmpFusedNode.dla_profiling.avg_exe_time), flush=True)
                # print("createMergedSubGraph: tmpFusedNode gpu_energy: {}".format(tmpSubGraph.num_inferences * tmpFusedNode.gpu_profiling.energy_dynamic + tmpSubGraph.NumDLA * tmpFusedNode.gpu_profiling.energy_static))
                # print("createMergedSubGraph: tmpFusedNode dla_energy: {}".format(tmpSubGraph.num_inferences * tmpFusedNode.dla_profiling.energy_dynamic))
            # end for
        # end if
    # end for

    # TODO: 这里直接减去 单独测量的 重复节点 可能不太合理
    # 单独测量可能导致额外的开销
    # 或许可以计算折扣系数, 按比例扣减 重复节点数据

    # 扣除重复的节点的数据
    for i in range(len(listSubGraph)-1):
        SubGraph0 = listSubGraph[i]
        SubGraph1 = listSubGraph[i+1]

        tmpSetFusedNode = set(SubGraph0.list_fused_node) & set(SubGraph1.list_fused_node)
        tmpListFusedNode = list(tmpSetFusedNode)

        for tmpFusedNode in tmpListFusedNode:
            # print("createMergedSubGraph: repetitive FusedNode: {}".format(tmpFusedNode.getName(newSubGraph.NetworkMap.useless_prefix)))
            # 减 重复 节点 gpu 执行时间
            newSubGraph.gpu_time -= 1.0 * newSubGraph.NumDLA * tmpFusedNode.gpu_profiling.avg_exe_time
            # 减 重复 节点 gpu 能耗
            newSubGraph.gpu_energy -= 1.0 * newSubGraph.num_inferences * tmpFusedNode.gpu_profiling.energy_dynamic + newSubGraph.NumDLA * tmpFusedNode.gpu_profiling.energy_static

            # 减 使用 DLA 时 DLA node 运行时间
            newSubGraph.time_dla_node -= 1.0 * tmpFusedNode.dla_profiling.avg_exe_time
            # 减 重复 节点 dla 动态能耗
            newSubGraph.dla_energy -= 1.0 * newSubGraph.num_inferences * tmpFusedNode.dla_profiling.energy_dynamic

            # TODO: 这里的系数如何确定, 如果考虑减少 stage数量 的奖励

            # print("createMergedSubGraph: repetitive delta gpu_time = {}\n".format(newSubGraph.NumDLA * tmpFusedNode.gpu_profiling.avg_exe_time), flush=True)
            # print("createMergedSubGraph: repetitive delta gpu_energy = {}\n".format(newSubGraph.num_inferences * tmpFusedNode.gpu_profiling.energy_dynamic + newSubGraph.NumDLA * tmpFusedNode.gpu_profiling.energy_static), flush=True)
            # print("createMergedSubGraph: repetitive delta time_dla_node = {}\n".format(tmpFusedNode.dla_profiling.avg_exe_time), flush=True)
            # print("createMergedSubGraph: repetitive delta dla_energy = {}\n".format(newSubGraph.num_inferences * tmpFusedNode.dla_profiling.energy_dynamic), flush=True)
        # end for
    # end for

    newSubGraph.calculateBubbleTime()

    # exit(0)

    return newSubGraph


# 判断节点是否 既可以使用 DLA, 又可以使用 GPU
def canUseDLA_GPU(FusedNode):
    if FusedNode.isCanUseDLA == False or FusedNode.isCanUseGPU == False \
        or FusedNode.gpu_profiling.avg_exe_time <= 0.0 or FusedNode.dla_profiling.avg_exe_time <= 0.0:
        return False
    return True