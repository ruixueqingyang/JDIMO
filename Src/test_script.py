# coding=utf-8
import sys, os, math, time, copy
import numpy as np
from test_profiling import ETOpt

algorithm = "subgraph"

model_name = "vgg16"
ETOpt(model_name, algorithm)

# model_name = "vgg19"
# ETOpt(model_name, algorithm)

# model_name = "mobilenetv2-7"
# ETOpt(model_name, algorithm)
model_name = "vgg16"
ETOpt(model_name, algorithm)

# model_name = "retinanet-9"
# ETOpt(model_name, algorithm)

# model_name = "yolov4"
# ETOpt(model_name, algorithm)



