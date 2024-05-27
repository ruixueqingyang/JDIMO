# coding=utf-8
##############################################################################################
# 移动平台 CNN 能效-温度 优化 主文件
##############################################################################################

import sys, os, math, time
import numpy as np
import torch
import copy
import network_type

if __name__ == "__main__":
    # model_name = "vgg16"
    # model_name = "vgg19"
    # model_name = "mobilenetv2-7"
    # model_name = "retinanet-9"
    model_name = "yolov4"
    
    

    time_start = time.time()
    time_begin = time.time()
    
    if os.path.exists("D:\\cloud\\study\\Coding"):
        WorkDir = "D:\\cloud\\study\\Coding"
    elif os.path.exists("/home/wfr/Coding"):
        WorkDir = "/home/wfr/Coding"
    else:
        print("预设工作路径不存在!")
        exit(1)

    origin_stdout = sys.stdout
    origin_stderr = sys.stderr
    std2file = False

    BatchSize = 1
    listUnknownDims = []
    dictInputTensor = {}
    dictTensorShape = {}
    listInputTensor = []
    useless_prefix = ""

    if model_name == "densenet-12":
        # densenet-12 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "densenet-12")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "densenet-12", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "densenet-12.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "densenet-12_profiling.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data_0"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data_0"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        useless_prefix = ""

    elif model_name == "shufflenet-v2-10":
        # shufflenet-v2-10 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "shufflenet-v2-10")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "shufflenet-v2-10", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "shufflenet-v2-10.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "shufflenet-v2-10_profiling.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["input"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["input"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        useless_prefix = ""

    elif model_name == "retinanet-9":
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

    elif model_name == "inception-v2-9":
        # inception-v2-9 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "inception-v2-9")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "inception-v2-9", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "inception-v2-9.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "inception-v2-9_profiling.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data_0"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data_0"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        useless_prefix = ""
    
    elif model_name == "squeezenet1.1":
        # squeezenet1.1 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "squeezenet1.1")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "squeezenet1.1", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "squeezenet1.1.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "squeezenet1.1_profiling.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        useless_prefix = ""

    elif model_name == "bvlcalexnet-12":
        # bvlcalexnet-12 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "bvlcalexnet-12")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "bvlcalexnet-12", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "bvlcalexnet-12.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "bvlcalexnet-12_profiling.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data_0"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data_0"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        useless_prefix = ""

    elif model_name == "ssd-12":
        # ssd-12 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "ssd-12")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "ssd-12", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "ssd-12.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "ssd-12_profiling.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["image"] = np.random.random([1,3,1200,1200]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["image"]]
        listInputShape = [[1,3,1200,1200]]
        listRange = [[0, 255]]

        useless_prefix = ""
        dictTensorShape["bboxes"] = [1, 200, 4]
        dictTensorShape["labels"] = [1, 200]
        dictTensorShape["scores"] = [1, 200]

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

    elif model_name == "googlenet-12":
        # googlenet-12 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "googlenet-12")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "googlenet-12", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "googlenet-12.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "googlenet-12_profiling.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = []
        dictInputTensor["data_0"] = np.random.random([1,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data_0"]]
        listInputShape = [[1,3,224,224]]
        listRange = [[0, 255]]

        useless_prefix = ""

    elif model_name == "resnet50-v2-7":
        # resnet50-v2-7 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet50-v2-7")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet50-v2-7", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "resnet50-v2-7.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "resnet50-v2-7_profiling.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = [BatchSize]
        dictInputTensor["data"] = np.random.random([BatchSize,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[BatchSize,3,224,224]]
        listRange = [[0, 255]]

        useless_prefix = "resnetv24_"

    elif model_name == "yolov3-tiny":
        # tiny-yolov3 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "tiny-yolov3")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "tiny-yolov3", "19_dla_layers")
        onnx_file_dir = os.path.join(input_folder_dir, "yolov3-tiny.onnx")
        # "D:\cloud\study\Coding\DLA\onnx_model_zoo\tiny-yolov3\yolov3-tiny.onnx"
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "yolov3-tiny_profiling.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
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
        useless_prefix = "TFNodes/yolo_evaluation_layer_1/"

    elif model_name == "yolov3":
        # yolov3 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov3")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "yolov3", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "yolov3.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "yolov3_profiling.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
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
        useless_prefix = "TFNodes/yolo_evaluation_layer_1/"

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

    elif model_name == "resnet18-v2-7":
        # resnet18-v2-7 相关路径
        input_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet18-v2-7")
        output_folder_dir = os.path.join(WorkDir, "DLA", "onnx_model_zoo", "resnet18-v2-7", "opt")
        onnx_file_dir = os.path.join(input_folder_dir, "resnet18-v2-7.onnx")
        if std2file == True:
            sys.stdout = open(os.path.join(input_folder_dir, "resnet18-v2-7_profiling.log"), "w", encoding='utf-8')
            sys.stderr = sys.stdout
        BatchSize = 1
        listUnknownDims = [BatchSize]
        dictInputTensor["data"] = np.random.random([BatchSize,3,224,224]).astype(np.float32) * 255
        listInputTensor = [dictInputTensor["data"]]
        listInputShape = [[BatchSize,3,224,224]]
        listRange = [[0, 255]]

        useless_prefix = "resnetv22_"

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

    NumLayers = len(NetworkMap.dictNetworkNode)
    NumBranchNodes = 0
    for NetworkNode in NetworkMap.dictNetworkNode.values():
        if len(NetworkNode.list_parent) > 1 or len(NetworkNode.list_child) > 1:
            NumBranchNodes += 1

    print("{}:".format(model_name))
    print("层数量: {}".format(NumLayers))
    print("分支节点数量: {}".format(NumBranchNodes))



    sys.stdout = origin_stdout
    sys.stderr = origin_stderr
