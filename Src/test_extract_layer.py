# coding=utf-8
import os, json
import extract_layer

# log_dir = "D:\\cloud\\study\\Coding\\DLA\\onnx_model_zoo\\yolov4\\StatefulPartitionedCall-model-lambda_71-Exp+StatefulPartitionedCall-model-lambda_71-add+StatefulPartitionedCall-model-lambda_71-Log+StatefulPartitionedCall-model-lambda_71-Tanh+StatefulPartitionedCall-model-lambda_71-mul_trtexec_perf_dla.log"
log_dir = "D:\\cloud\\study\\Coding\\DLA\\onnx_model_zoo\\yolov4\\yolov4_trtexec_verbose_gpu.log"
json_file_dir = "D:\\cloud\\study\\Coding\\DLA\\onnx_model_zoo\\yolov4\\yolov4_layer_information_gpu.json"
if os.path.exists(log_dir):
    # listDLALayerNames, listGPULayerNames = extract_layer.extrat_layer_from_log(log_dir)

    sum_exe_time = 0
    with open(json_file_dir, "r") as file:
        listJson = json.load(file)
        invalid = listJson[0]["count"]
        for i in range(1, len(listJson), 1):
            dictTmp = listJson[i]
            name = dictTmp["name"]
            exe_time = dictTmp["averageMs"] / 1000

            sum_exe_time += exe_time