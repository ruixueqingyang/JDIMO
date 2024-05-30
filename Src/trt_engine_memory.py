# coding=utf-8
import os
import tensorrt as trt

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
import pycuda.driver as cuda
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

# from calibrator import VOID_CALIBRATOR

try:
    # Sometimes python does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger()
# TRT_LOGGER = trt.Logger(trt.tensorrt.ILogger.Severity.VERBOSE)
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, use_managed_memory, name=""):
        self.host = host_mem
        self.device = device_mem
        self.use_managed_memory = use_managed_memory
        self.name = name

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device) + "\nuse_managed_memory:\n" + str(self.use_managed_memory)

    def __repr__(self):
        return self.__str__()

# managed memory 内存分配

# pagelocked / pinned memory 内存分配

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, use_managed_memory=False, create_new_stream=True, dictTensorShape={}):
    print("allocate_buffers:")
    print("dictTensorShape: {}".format(dictTensorShape))
    inputs = []
    outputs = []
    bindings = []
    if create_new_stream == True:
        stream = cuda.Stream()
    else:
        stream = None
    for binding in engine:
        print("binding name = {}".format(binding))
        if binding in dictTensorShape.keys():
            shape = dictTensorShape[binding]
        else:
            shape = engine.get_binding_shape(binding)
            # shape = tuple(shape[i] for i in range(len(shape)))
            shape = tuple([*shape])

        print("binding shape = {}".format(shape))
        # size = trt.volume(engine.get_binding_shape(binding))
        np_dtype = trt.nptype(engine.get_binding_dtype(binding))

        if use_managed_memory == False:
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(shape, np_dtype) # 分配页面锁定的 numpy.ndarray
            # host_mem = np.empty(shape, dtype = np_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
        else:
            host_mem = cuda.managed_empty(shape, np_dtype, mem_flags=cuda.mem_attach_flags.GLOBAL) # 分配 cuda 统一内存的 numpy.ndarray
            # device_mem = host_mem.ctypes.data # 获得 numpy array类型变量 数据区首地址
            device_mem = int(host_mem.base.get_device_pointer())
        
        # Append the device buffer to device bindings.
        # print("type(device_mem): {}".format(type(device_mem)))
        bindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, use_managed_memory, binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, use_managed_memory, binding))
    return inputs, outputs, bindings, stream

# Allocates input buffers required for an engine, i.e. host/device inputs.
def allocate_input_buffers(engine, buf_len=1, use_managed_memory=False, dictTensorShape = {}):
    # print("allocate_input_buffers:")

    # 分配多份缓冲区域, 一遍进行 DLA / GPU 流水执行
    list_inputs = []
    list_input_bindings = []

    # if create_new_stream == True:
    #     stream = cuda.Stream()
    # else:
    #     stream = None
    for i in range(buf_len):
        inputs = []
        bindings = []
        
        for binding in engine:
            if not engine.binding_is_input(binding):
                continue

            if binding in dictTensorShape.keys():
                shape = dictTensorShape[binding]
            else:
                shape = engine.get_binding_shape(binding)
                shape = tuple([*shape])

            print("allocate_input_buffers: {}".format(binding))
            print("binding shape = {}".format(shape))
            # print("type shape: {}".format(type(shape)))
            size = trt.volume(engine.get_binding_shape(binding))
            np_dtype = trt.nptype(engine.get_binding_dtype(binding))

            if use_managed_memory == False:
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(shape, np_dtype) # 分配页面锁定的 numpy.ndarray
                device_mem = cuda.mem_alloc(host_mem.nbytes)
            else:
                host_mem = cuda.managed_empty(shape, np_dtype, mem_flags=cuda.mem_attach_flags.GLOBAL) # 分配 cuda 统一内存的 numpy.ndarray
                # device_mem = host_mem.ctypes.data # 获得 numpy array类型变量 数据区首地址
                device_mem = int(host_mem.base.get_device_pointer())
            
            # Append the device buffer to device bindings.
            # print("type(device_mem): {}".format(type(device_mem)))
            bindings.append(int(device_mem))

            # Append to the appropriate list.
            inputs.append(HostDeviceMem(host_mem, device_mem, use_managed_memory, binding))
        
        list_inputs.append(inputs)
        list_input_bindings.append(bindings)

    return list_inputs, list_input_bindings

# Allocates output buffers required for an engine, i.e. host/device outputs.
def allocate_output_buffers(engine, buf_len=1, use_managed_memory=False, dictTensorShape = {}):

    # print("allocate_output_buffers:")

    # 分配多份缓冲区域, 一遍进行 DLA / GPU 流水执行
    list_outputs = []
    list_output_bindings = []

    # if create_new_stream == True:
    #     stream = cuda.Stream()
    # else:
    #     stream = None
    for i in range(buf_len):
        outputs = []
        bindings = []
        
        for binding in engine:
            if engine.binding_is_input(binding):
                continue

            if binding in dictTensorShape.keys():
                shape = dictTensorShape[binding]
            else:
                shape = engine.get_binding_shape(binding)
                # shape = tuple(shape[i] for i in range(len(shape)))
                shape = tuple([*shape])

            print("allocate_output_buffers: {}".format(binding))
            print("binding shape = {}".format(shape))
            size = trt.volume(engine.get_binding_shape(binding))
            np_dtype = trt.nptype(engine.get_binding_dtype(binding))

            if use_managed_memory == False:
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(shape, np_dtype) # 分配页面锁定的 numpy.ndarray
                device_mem = cuda.mem_alloc(host_mem.nbytes)
            else:
                host_mem = cuda.managed_empty(shape, np_dtype, mem_flags=cuda.mem_attach_flags.GLOBAL) # 分配 cuda 统一内存的 numpy.ndarray
                # device_mem = host_mem.ctypes.data # 获得 numpy array类型变量 数据区首地址
                device_mem = int(host_mem.base.get_device_pointer())
            
            # Append the device buffer to device bindings.
            # print("type(device_mem): {}".format(type(device_mem)))
            bindings.append(int(device_mem))

            # Append to the appropriate list.
            outputs.append(HostDeviceMem(host_mem, device_mem, use_managed_memory, binding))
        
        list_outputs.append(outputs)
        list_output_bindings.append(bindings)

    return list_outputs, list_output_bindings

def get_engine(onnx_file_path, engine_file_path="", batch_size=1, default_device_type=trt.DeviceType.GPU, dictNameDevice={}, dla_index=0, my_calibrator=None):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser, \
            trt.Runtime(TRT_LOGGER) as runtime:

            # 先不改配置
            # config.max_workspace_size = 1 << 28  # 256MiB # 过时的参数
            # builder.max_batch_size = 1 # 过时的参数

            # config.flags = 1 << int(trt.BuilderFlag.FP16) | 1 << int(trt.BuilderFlag.INT8)
            config.flags = 1 << int(trt.BuilderFlag.INT8)
            # config.flags = 1 << int(trt.BuilderFlag.FP16)
            config.default_device_type = default_device_type
            # if default_device_type == trt.DeviceType.DLA:
            config.flags = config.flags | 1 << int(trt.BuilderFlag.GPU_FALLBACK)
            # config.default_device_type = trt.DeviceType.GPU # GPU DLA
            config.DLA_core = dla_index # 0 / 1

            runtime.DLA_core = dla_index
            runtime.max_threads = 8

            # 设置 TensorRT 搜索最优配置时 使用的内存池的大小
            memory_pool_size = 8 * (1024**3) # 6GB / 8GB
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_pool_size)
            memory_pool_size = config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)
            print("memory_pool_size = {}".format(memory_pool_size), flush=True)

            # managed_memory_pool_size = 1 * (1024**2) # 128MB
            # config.set_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM , managed_memory_pool_size)
            # managed_memory_pool_size = config.get_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM)
            # print("managed_memory_pool_size = {}".format(managed_memory_pool_size), flush=True)

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print("ONNX file {} not found.".format(onnx_file_path))
                exit(0)
            print("Loading ONNX file from path {} ...".format(onnx_file_path), flush=True)
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing", flush=True)
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.", flush=True)
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            

            # 显式指定 batch_size, 并
            # 这里未知维度(onnx模型中是"N"或"?"的维度)的默认值似乎是 -1
            listInputShape = []
            print("\n")
            # print("batch_size = {}".format(batch_size))
            for i in range(network.num_inputs):
                print("input {}: {}".format(i, network.get_input(i).shape))
                tmpShape = network.get_input(0).shape
                if tmpShape[0] > 0:
                    pass
                else:
                    tmpShape[0] = batch_size
                    network.get_input(0).shape = tmpShape
                    print("input {}: {}".format(i, network.get_input(i).shape))
                tmp_list = [*tmpShape]
                listInputShape.append(tmp_list)


            # 配置 int8 校准
            # tmp = os.path.split(onnx_file_path)
            # onnx_file_folder_path = tmp[0]
            # onnx_file_name = os.path.splitext(tmp[1])[0]
            # cache_file_dir = os.path.join(onnx_file_folder_path, onnx_file_name+"_calibration.cache")
            # print("cache_file_dir = {}".format(cache_file_dir))
            # list_range = [0,255]
            # num_batches = 32
            # my_calibrator = VOID_CALIBRATOR(listInputShape[0], list_range, num_batches, cache_file_dir)
            if my_calibrator != None:
                config.int8_calibrator = my_calibrator


            # 尝试在这里获得 CNN 层数/每层信息, 并进行配置使用 GPU/DLA
            # config.set_device_type(self: tensorrt.tensorrt.IBuilderConfig, layer: tensorrt.tensorrt.ILayer, device_type: tensorrt.tensorrt.DeviceType)→ None
            # config.can_run_on_DLA(self: tensorrt.tensorrt.IBuilderConfig, layer: tensorrt.tensorrt.ILayer)→ bool
            # for i in range(len(network)):
            #     tmpLayer = network[i]
            #     config.set_device_type(tmpLayer, trt.DeviceType.GPU)
            #     # if True == config.can_run_on_DLA(tmpLayer):
            #     #     config.set_device_type(tmpLayer, trt.DeviceType.DLA)
            #     # else:
            #     #     config.set_device_type(tmpLayer, trt.DeviceType.GPU)

            # 根据设定的配置 设置使用 DLA / GPU
            for i in range(len(network)):
                tmpLayer = network[i]
                if tmpLayer.name in dictNameDevice.keys():
                    config.set_device_type(tmpLayer, dictNameDevice[tmpLayer.name])
                else:
                    config.set_device_type(tmpLayer, default_device_type)

            print("Completed parsing of ONNX file", flush=True)
            print("Building an engine from file {}; this may take a while ...".format(onnx_file_path), flush=True)

            # engine = builder.build_engine(network, config) # 如果不需要保存 trt 模型, 直接使用这个即可, "serialized" 是为了能够输出保存到文件的 trt 格式

            serialized_engine = builder.build_serialized_network(network, config)
            print("build_serialized_network complete")
            engine = runtime.deserialize_cuda_engine(serialized_engine)

            print("Completed creating Engine")

            with open(engine_file_path, "wb") as f:
                f.write(serialized_engine)

            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            runtime.DLA_core = dla_index
            runtime.max_threads = 8
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

# 获得能在 DLA 上运行的层 的 层名
def getDLALayerNames(onnx_file_path):
    setDLALayerName = set()

    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        builder.create_builder_config() as config, \
        trt.OnnxParser(network, TRT_LOGGER) as parser, \
        trt.Runtime(TRT_LOGGER) as runtime:

        # Parse model file
        if not os.path.exists(onnx_file_path):
            print("ONNX file {} not found.".format(onnx_file_path))
            exit(0)
        # print("Loading ONNX file from path {} ...".format(onnx_file_path), flush=True)
        with open(onnx_file_path, "rb") as model:
            # print("Beginning ONNX file parsing", flush=True)
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.", flush=True)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None


        # 尝试在这里获得 CNN 层数/每层信息, 并进行配置使用 GPU/DLA
        # config.set_device_type(self: tensorrt.tensorrt.IBuilderConfig, layer: tensorrt.tensorrt.ILayer, device_type: tensorrt.tensorrt.DeviceType)→ None
        # config.can_run_on_DLA(self: tensorrt.tensorrt.IBuilderConfig, layer: tensorrt.tensorrt.ILayer)→ bool
        
        for i in range(len(network)):
            tmpLayer = network[i]
            LayerName = tmpLayer.name
            canRunOnDLA = config.can_run_on_DLA(tmpLayer)
            # print("getDLALayerNames: {}: LayerName: {}; isCanUseDLA: {}".format(i, LayerName, canRunOnDLA))
            if canRunOnDLA == True:
                setDLALayerName.add(LayerName)

    return setDLALayerName
