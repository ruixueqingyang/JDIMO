# coding=utf-8
# 从文件中提取层融合信息
# 知道 layer 是否能在 DLA 上运行比较简单
# 关键是节点聚合方式 全 gpu 和 全 dla 可能不同
# 就按照 全gpu 方式融合
# 使用 全dla log 确定是否可以在 dla 上运行
# 也可以 通过尝试确定 是否可以运行在 dla 上

import re, json

isGPU = "GPU"
isDLA = "DLA"

def extrat_layer_name(strFusedLayerName):

    # listLayerName = ["layer1", "layer2"]
    listLayerName = []

    strTmp = strFusedLayerName
    # PWN(PWN(TFNodes/yolo_evaluation_layer_1/sub_3, PWN(TFNodes/yolo_evaluation_layer_1/truediv_14/y:0_clone_1 + (Unnamed Layer* 481) [Shuffle], TFNodes/yolo_evaluation_layer_1/truediv_11)), TFNodes/yolo_evaluation_layer_1/truediv_12)

    if strTmp.find("Reformatting CopyNode") >= 0:
        return listLayerName

    # 删除无关字符
    strTmp = re.sub("ForeignNode", "", strTmp)
    strTmp = re.sub("\[trainStation\d*\]", "", strTmp)
    strTmp = re.sub("\[LoopOutput\]", "", strTmp)
    strTmp = re.sub("PWN", "", strTmp)
    # strTmp = re.sub("(Unnamed Layer* \d*) \[Shuffle\]", "", strTmp)
    # strTmp = re.sub("(Unnamed Layer* \d*)", "", strTmp)
    strTmp = re.sub("\[Shuffle\]", "", strTmp)
    strTmp = re.sub("\[Constant\]", "", strTmp)
    # strTmp = re.sub("_clone_\d*", "", strTmp)
    # strTmp = re.sub("_copy_output", "", strTmp)
    # strTmp = re.sub(" copy", "", strTmp)
    # strTmp = re.sub(" copy (\d*)", "", strTmp)
    strTmp = re.sub("\(|\)", "", strTmp)
    strTmp = re.sub("\[|\]", "", strTmp)
    strTmp = re.sub("\{|\}", "", strTmp)
    strTmp = re.sub("\.\.\.", "+", strTmp)

    # TFNodes/yolo_evaluation_layer_1/sub_3 + TFNodes/yolo_evaluation_layer_1/truediv_14/y:0_clone_1 + + TFNodes/yolo_evaluation_layer_1/truediv_11 + TFNodes/yolo_evaluation_layer_1/truediv_12

    # 替换字符
    strTmp = re.sub("\r", "", strTmp)
    strTmp = re.sub("\n", "", strTmp)
    strTmp = re.sub(",", "+", strTmp)
    strTmp = re.sub(" ", "", strTmp)
    strTmp = re.sub("\+\+*", "+", strTmp)

    # strTmp = re.sub("\|\|", "+", strTmp)
    # 在 "||" 处 分割字符串, 每个子串分别处理, 之后每个字串分别融合成一个节点
    # 即不融合在网络结构上彼此并行的节点/层
    listStr = strTmp.split("||")

    listlistLayerName = []
    for subStr in listStr:
        listLayerName = subStr.split("+")
        listlistLayerName.append(listLayerName)

    # if len(strTmp) > 0:
    #     listLayerName = strTmp.split("+")

    return listlistLayerName

# 提取一行记录中的 layer 名
# [12/23/2023-20:06:53] [I] [TRT] [GpuLayer] POINTWISE: PWN(PWN(TFNodes/yolo_evaluation_layer_1/sub_3, PWN(TFNodes/yolo_evaluation_layer_1/truediv_14/y:0_clone_1 + (Unnamed Layer* 481) [Shuffle], TFNodes/yolo_evaluation_layer_1/truediv_11)), TFNodes/yolo_evaluation_layer_1/truediv_12)
# [12/23/2023-20:40:31] [I] [TRT] [DlaLayer] {ForeignNode[TFNodes/yolo_evaluation_layer_1/sub_3]}
def extrat_layer_name_from_log_line(strFusedLayerName):

    # listlistLayerName = [["layer1", "layer2"],["layer3", "layer4"]]
    # listlistLayerName = [["layer1", "layer2"]]
    listlistLayerName = []
    device_type = ""

    pos0 = strFusedLayerName.find("[GpuLayer]")
    if pos0 >= 0:
        device_type = isGPU
    else:
        pos0 = strFusedLayerName.find("[DlaLayer]")
        if pos0 >= 0:
            device_type = isDLA

    if pos0 < 0:
        return listlistLayerName, device_type

    pos1 = strFusedLayerName.find(": ", pos0)

    if pos1 > 0:
        strTmp = strFusedLayerName[pos1+len(": "):]
    elif device_type == isGPU:
        strTmp = strFusedLayerName[pos0+len("[GpuLayer] "):]
    elif device_type == isDLA:
        strTmp = strFusedLayerName[pos0+len("[DlaLayer] "):]
    # PWN(PWN(TFNodes/yolo_evaluation_layer_1/sub_3, PWN(TFNodes/yolo_evaluation_layer_1/truediv_14/y:0_clone_1 + (Unnamed Layer* 481) [Shuffle], TFNodes/yolo_evaluation_layer_1/truediv_11)), TFNodes/yolo_evaluation_layer_1/truediv_12)
        
    # [01/18/2024-17:29:21] [I] [TRT] [DlaLayer] {ForeignNode[StatefulPartitionedCall/model/conv2d_2/Conv2D...StatefulPartitionedCall/model/conv2d_6/Conv2D]}

    if strTmp.find("Reformatting CopyNode") >= 0:
        return listlistLayerName, device_type

    listlistLayerName = extrat_layer_name(strTmp)

    return listlistLayerName, device_type

def extrat_layer_from_log(log_file_dir):

    listDLALayerNames = []
    listGPULayerNames = []

    with open(log_file_dir, "r") as file:
        for line in file.readlines():
            pos = line.find("=== Trace details ===")
            if pos > 0:
                break
            listlistLayerName, device_type = extrat_layer_name_from_log_line(line)
            for listLayerName in listlistLayerName:
                if len(listLayerName) > 0:
                    if device_type == isDLA:
                        listDLALayerNames.append(listLayerName)
                    elif device_type == isGPU:
                        listGPULayerNames.append(listLayerName)

    return listDLALayerNames, listGPULayerNames

def extrat_layer_from_json(json_file_dir):

    listDLALayerNames = []
    listGPULayerNames = []
    listDLALayerTime = []
    listGPULayerTime = []

    with open(json_file_dir, "r") as file:
        listJson = json.load(file)
        invalid = listJson[0]["count"]
        for i in range(1, len(listJson), 1):
            dictTmp = listJson[i]
            name = dictTmp["name"]
            exe_time = dictTmp["averageMs"] / 1000

            # 跳过额外层
            pattern = "Reformatting CopyNode for "
            result = re.match(pattern, name, flags=0)
            if result != None:
                continue
            # 跳过额外层
            pattern = "reshape_before_"
            result = re.match(pattern, name, flags=0)
            if result != None:
                continue
            # 跳过额外层
            pattern = "reshape_after_"
            result = re.match(pattern, name, flags=0)
            if result != None:
                continue

            # 判断是在 GPU/DLA 上
            device = "GPU"
            pattern = "{ForeignNode\[.*\]}"
            result = re.match(pattern, name, flags=0)
            if result != None:
                device = "DLA"

            listlistLayerName = extrat_layer_name(name)
            listExeTime = []
            countLayer = 0
            for listLayerName in listlistLayerName:
                countLayer += len(listLayerName)
            for listLayerName in listlistLayerName:
                listExeTime.append(exe_time*len(listLayerName)/countLayer)

            for i in range(len(listlistLayerName)):
                listLayerName = listlistLayerName[i]
                ExeTime = listExeTime[i]
                if len(listLayerName) > 0:
                    if device == "DLA":
                        listDLALayerNames.append(listLayerName)
                        listDLALayerTime.append(ExeTime)
                    elif device == "GPU":
                        listGPULayerNames.append(listLayerName)
                        listGPULayerTime.append(ExeTime)

    return listDLALayerNames, listDLALayerTime, listGPULayerNames, listGPULayerTime

def count_ForeignNode(json_file_dir):
    count = 0
    with open(json_file_dir, "r") as file:
        listJson = json.load(file)
        invalid = listJson[0]["count"]
        for i in range(1, len(listJson), 1):
            dictTmp = listJson[i]
            name = dictTmp["name"]
            pattern = "{ForeignNode\[.*\]}"
            result = re.match(pattern, name, flags=0)
            if result != None:
                count += 1

    return count