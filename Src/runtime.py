# coding=utf-8
##############################################################################################
# 移动平台 CNN 能效-温度 优化 运行时
##############################################################################################

import os, math
from time import time
# from queue import Queue
from multiprocessing import Queue, Process, Manager, Value, shared_memory
# from threading import Thread
from typing import Any
import onnx
import onnx.version_converter
import onnxruntime
import sys, getopt
import numpy as np
import torch
import copy
from enum import Enum
import tensorrt as trt
from trt_engine_memory import get_engine, allocate_buffers, allocate_input_buffers, allocate_output_buffers, HostDeviceMem

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
import pycuda.driver as cuda
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

def least_common_multiple(num):  # 求任意多个数的最小公倍数
    minimum = 1
    for i in num:
        minimum = int(i)*int(minimum) / math.gcd(int(i), int(minimum))
    return int(minimum)

class trt_config_index:
    def __init__(self, listIndex, device) -> None:
        self.listIndex = listIndex
        self.device = device # trt.DeviceType.DLA / trt.DeviceType.GPU

class trt_config_name:
    def __init__(self, listName, device) -> None:
        self.listName = listName
        self.device = device # trt.DeviceType.DLA / trt.DeviceType.GPU

# trt engine 及其 stream
class ENGINE_STREAM:
    def __init__(self, trt_engine) -> None:
        print("ENGINE_STREAM: in")
        self.trt_engine = trt_engine
        print("ENGINE_STREAM: create_execution_context")
        self.trt_context = self.trt_engine.create_execution_context()
        print("ENGINE_STREAM: create_execution_context done")
        self.stream = cuda.Stream()

# 将缓冲区组织成环形缓冲区
class RING_BUFFER:
    def __init__(self, list_pipeline_stage_engine, num_pipeline_buffers, use_managed_memory=False) -> None:
        list_stage_streams = [len(list_pipeline_stage_engine[i]) for i in range(len(list_pipeline_stage_engine))]
        lcm = least_common_multiple(list_stage_streams)

        # 共享变量
        num_pipeline_buffers = max(num_pipeline_buffers, 2)
        self.ring_len = Value('i', num_pipeline_buffers * lcm) # 定义共享变量
        self.dictStatus = {'empty': 0, 'writting': 1, 'full': 2, 'reading': 3}
        self.manager = Manager()
        self.list_status = self.manager.list([self.dictStatus['empty']] * self.ring_len.value)
        self.occupy_count = Value('i', 0) # 定义共享变量

        self.index_writting_first = Value('i', 0) # 定义共享变量
        self.index_writting_last = Value('i', 0) # 定义共享变量
        self.index_reading_first = Value('i', 0) # 定义共享变量
        self.index_reading_last = Value('i', 0) # 定义共享变量

        # 根据变量名来索引内存
        num_stages = len(list_pipeline_stage_engine)

        # 收集变量名, 初始化缓冲字典
        listVarName = []
        dictName_Shape = {}
        dictName_np_dtype = {}
        dictName_listMemPair = {}
        dictName_listMemDevice = {}
        for i in range(num_stages):
            tmpEngine = list_pipeline_stage_engine[i][0].trt_engine
            for tmpName in tmpEngine: # 每一个 输入/输出变量 的 变量名
                if not tmpName in listVarName:
                    # print("VarName: {}".format(tmpName))
                    listVarName.append(tmpName)
                    shape = tmpEngine.get_binding_shape(tmpName)
                    # print("shape: {}".format(shape))
                    shape = tuple([*shape])
                    if len(shape) == 0:
                        shape = tuple([1,])
                    # print("shape: {}".format(shape))
                    # print("get_binding_dtype: {}".format(tmpEngine.get_binding_dtype(tmpName)))
                    # print("nptype: {}".format(trt.nptype(tmpEngine.get_binding_dtype(tmpName))))
                    dictName_Shape[tmpName] = shape
                    dictName_np_dtype[tmpName] = trt.nptype(tmpEngine.get_binding_dtype(tmpName))
                    dictName_listMemPair[tmpName] = []
                    dictName_listMemDevice[tmpName] = []

        # 分配内存
        for i in range(self.ring_len.value):
            # 保证变量访问先后顺序, 有利于局部性
            for tmpName in listVarName:
                tmpShape = dictName_Shape[tmpName]
                tmp_np_dtype = dictName_np_dtype[tmpName]
                if use_managed_memory == False:
                    # Allocate host and device buffers
                    host_mem = cuda.pagelocked_empty(tmpShape, tmp_np_dtype) # 分配页面锁定的 numpy.ndarray
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                else:
                    host_mem = cuda.managed_empty(tmpShape, tmp_np_dtype, mem_flags=cuda.mem_attach_flags.GLOBAL) # 分配 cuda 统一内存的 numpy.ndarray
                    # device_mem = host_mem.ctypes.data # 获得 numpy array类型变量 数据区首地址
                    device_mem = int(host_mem.base.get_device_pointer())
                
                MemPair = HostDeviceMem(host_mem, device_mem, use_managed_memory, tmpName)

                dictName_listMemPair[tmpName].append(MemPair)
                dictName_listMemDevice[tmpName].append(device_mem)

        self.list_stage_events = []

        # list_stage_input_mem_pairs[i][j][k]
        # i: stage id;    j: buffer组 id;    k: 输入变量 id
        self.list_stage_input_mem_pairs = []
        self.list_stage_output_mem_pairs = []
        self.list_stage_input_mem_devices = []
        self.list_stage_output_mem_devices = []
        
        for i in range(num_stages):
            tmpEngine = list_pipeline_stage_engine[i][0].trt_engine
            # list_inputs[j][k]
            # j: 缓冲组id
            # k: 各个输入/输出变量
            stage_input_mem_pairs = []
            stage_output_mem_pairs = []
            stage_input_mem_devices = []
            stage_output_mem_devices = []
            for j in range(self.ring_len.value):
                input_mem_pairs = []
                output_mem_pairs = []
                input_mem_devices = []
                output_mem_devices = []
                for tmpName in tmpEngine:
                    if tmpEngine.binding_is_input(tmpName):
                        input_mem_pairs.append(dictName_listMemPair[tmpName][j])
                        input_mem_devices.append(dictName_listMemDevice[tmpName][j])
                    else:
                        output_mem_pairs.append(dictName_listMemPair[tmpName][j])
                        output_mem_devices.append(dictName_listMemDevice[tmpName][j])

                stage_input_mem_pairs.append(input_mem_pairs)
                stage_output_mem_pairs.append(output_mem_pairs)
                stage_input_mem_devices.append(input_mem_devices)
                stage_output_mem_devices.append(output_mem_devices)

            self.list_stage_input_mem_pairs.append(stage_input_mem_pairs)
            self.list_stage_output_mem_pairs.append(stage_output_mem_pairs)
            self.list_stage_input_mem_devices.append(stage_input_mem_devices)
            self.list_stage_output_mem_devices.append(stage_output_mem_devices)
            self.list_stage_events.append([cuda.Event() for _ in range(self.ring_len.value)])
        
        # 分配共享内存
        self.list_output_shms = []
        self.list_outputs = []
        for i in range(self.ring_len.value):
            mem_pairs = self.list_stage_output_mem_pairs[-1][i]
            self.list_output_shms.append([])
            self.list_outputs.append([])
            for mem_pair in mem_pairs:
                shm = shared_memory.SharedMemory(create=True, size=mem_pair.host.nbytes)
                output = np.ndarray(mem_pair.host.shape, dtype=mem_pair.host.dtype, buffer=shm.buf)
                self.list_output_shms[-1].append(shm)
                self.list_outputs[-1].append(output)

        # pycuda.driver.event_flags.DEFAULT # 等待 event 时, 当前线程会循环查询 event 状态, 可能会有很高 CPU 占用
        # pycuda.driver.event_flags.DISABLE_TIMING # 不计时
        # pycuda.driver.event_flags.BLOCKING_SYNC # 等待 event 时, 当前线程被挂起
        # pycuda.driver.event_flags.INTERPROCESS # 用于进程间 event 同步
        # class pycuda.driver.event_flags
        # DEFAULT
        # BLOCKING_SYNC
        # DISABLE_TIMING
        # INTERPROCESS
        
        # 缓冲区状态枚举类型
        # self.status = Enum('status', ('empty', 'writting', 'full', 'reading'))
        # # self.list_status = [self.status.empty] * self.ring_len.value
        # # self.index_writting = 0 # 最近正在写 / 刚写完
        # self.index_reading = 0 # 最近正在读 / 刚读完
        # self.occupy_count = 0
    
    def free_shm(self):
        # 释放共享内存
        for i in range(len(self.list_output_shms)):
            for j in range(len(self.list_output_shms[i])):
                self.list_output_shms[i][j].close()
                self.list_output_shms[i][j].unlink()

    def copyOutput(self, idx_writting):
        # print("copyOutput: in", flush=True)
        num_outputs = len(self.list_outputs[idx_writting])
        for idx_output in range(num_outputs):
            self.list_outputs[idx_writting][idx_output][:] = self.list_stage_output_mem_pairs[-1][idx_writting][idx_output].host[:]
        # print("copyOutput: done", flush=True)

    def getEmptyIndex(self):
        tmp_index = (self.index_writting_last.value + 1) % self.ring_len.value
        if self.list_status[tmp_index] == self.dictStatus["empty"]:
            self.list_status[tmp_index] = self.dictStatus["writting"]
            self.index_writting_last.value = tmp_index
            self.occupy_count.value += 1
            # 处理 申请第一个 写缓存 的情况
            if self.list_status[self.index_writting_first.value] != self.dictStatus["writting"]:
                self.index_writting_first.value = (self.index_writting_first.value + 1) % self.ring_len.value
            return self.index_writting_last.value
        else:
            return None

    def setReading(self, index_reading):
        self.list_status[index_reading] = self.dictStatus["reading"]
        
        # self.index_reading_last.value = (self.index_reading_last.value + 1) % self.ring_len.value
        for _ in range(self.ring_len.value):
            tmp_index = (self.index_reading_last.value + 1) % self.ring_len.value
            if self.list_status[tmp_index] == self.dictStatus["reading"]:
                self.index_reading_last.value = tmp_index
            else:
                break
        # 处理 设置第一个 读缓存 的情况
        if self.list_status[self.index_reading_first.value] != self.dictStatus["reading"]:
            self.index_reading_first.value = (self.index_reading_first.value + 1) % self.ring_len.value
        
        # self.index_writting_first.value = (self.index_writting_first.value + 1) % self.ring_len.value
        for _ in range(self.ring_len.value):
            if self.list_status[self.index_writting_first.value] != self.dictStatus["writting"]:
                self.index_writting_first.value = (self.index_writting_first.value + 1) % self.ring_len.value
            else:
                break
    
    def setEmpty(self, index_released):
        self.list_status[index_released] = self.dictStatus["empty"]
        self.occupy_count.value -= 1

        # self.index_reading_first.value = (self.index_reading_first.value + 1) % self.ring_len.value
        for _ in range(self.ring_len.value):
            tmp_index = (self.index_reading_first.value + 1) % self.ring_len.value
            if self.list_status[tmp_index] != self.dictStatus["reading"]:
                self.index_reading_first.value = tmp_index
            else:
                break

    def getReadingIndex(self):
        if self.list_status[self.index_reading_first.value] != self.dictStatus["reading"]:
            return None
        else:
            return self.index_reading_first.value

# 流水线化的 trt engine, 分别映射到 DLA/GPU
class ENGINE_PIPELINE():
    def __init__(self, list_pipeline_stage_engine, num_pipeline_buffers=4, gpu_id=0) -> None:
        self.device = cuda.Device(gpu_id)
        self.context = self.device.retain_primary_context()
        self.list_pipeline_stage_engine = list_pipeline_stage_engine
        self.num_pipeline_buffers = num_pipeline_buffers
        self.use_managed_memory = False
        self.num_pipeline_stages = len(self.list_pipeline_stage_engine)

        self.ring_buf = RING_BUFFER(self.list_pipeline_stage_engine, self.num_pipeline_buffers, self.use_managed_memory)

        num_input_streams = len(self.list_pipeline_stage_engine[0])
        self.list_input_stream = [cuda.Stream() for _ in range(num_input_streams)]
        self.list_input_event = [cuda.Event() for _ in range(self.ring_buf.ring_len.value)]

        num_output_streams = len(self.list_pipeline_stage_engine[-1])
        self.list_output_stream = [cuda.Stream() for _ in range(num_output_streams)]
        self.list_output_event = [cuda.Event(cuda.event_flags.BLOCKING_SYNC) for _ in range(self.ring_buf.ring_len.value)]
                
        # 初始化输入, 暂时不使用真实输入
        for idx_buf in range(len(self.ring_buf.list_stage_input_mem_pairs[0])):
            for idx_input in range(len(self.ring_buf.list_stage_input_mem_pairs[0][idx_buf])):
                # tmpInput = 9 * np.ones(self.ring_buf.list_stage_input_mem_pairs[0][idx_buf][idx_input].host.shape, dtype=self.ring_buf.list_stage_input_mem_pairs[0][idx_buf][idx_input].host.dtype)
                # np.random.seed(0)
                tmpInput = 255 * np.random.random(self.ring_buf.list_stage_input_mem_pairs[0][idx_buf][idx_input].host.shape).astype(self.ring_buf.list_stage_input_mem_pairs[0][idx_buf][idx_input].host.dtype)
                np.copyto(self.ring_buf.list_stage_input_mem_pairs[0][idx_buf][idx_input].host, tmpInput) # 将 numpy array 拷贝到 cuda 分配的锁页 numpy array / managed_memory
    
    def fillInputBuf(self, list_stage0_input):
        for idx_buf in range(len(self.ring_buf.list_stage_input_mem_pairs[0])):
            for idx_input in range(len(self.ring_buf.list_stage_input_mem_pairs[0][idx_buf])):
                # print("idx_buf = {}, idx_input = {}".format(idx_buf, idx_input))
                np.copyto(self.ring_buf.list_stage_input_mem_pairs[0][idx_buf][idx_input].host, list_stage0_input[idx_buf][idx_input]) # 将 numpy array 拷贝到 cuda 分配的锁页 numpy array / managed_memory

    def __del__(self):
        self.ring_buf.free_shm()

    def getEmptyBuffer(self, is_print_stage_time=False):
        # print("getEmptyBuffer: in", flush = True)

        # 申请一组空的缓存, 即 一次完整流水 的各个 stage 所需的全部缓存
        idx_buf = None
        while idx_buf == None:
            # print("getEmptyBuffer: idx_buf = {}".format(idx_buf), flush = True)
            idx_buf = self.ring_buf.getEmptyIndex()
            # print("getEmptyBuffer: idx_buf = {}".format(idx_buf), flush = True)
            if idx_buf != None:
                break

            index_writting_first = self.ring_buf.index_writting_first.value
            index_writting_last = self.ring_buf.index_writting_last.value
            idx_writting = index_writting_first
            
            # 先进行同步(等待数据拷贝完成)再进行数据拷贝及结果输出
            event_writting = self.list_output_event[idx_writting]
            # 等待一次完整流水完成
            event_writting.synchronize()
            # 如果使用 managed memory
            if self.use_managed_memory == True:
                # 先进行同步 再进行内存拷贝
                self.context.synchronize()
            
            
            # 调试打印各个 event 间的时间
            if is_print_stage_time == True:
                event_curr = self.list_input_event[idx_writting]
                for i in range(len(self.ring_buf.list_stage_events)):
                    event_prev = event_curr
                    event_curr = self.ring_buf.list_stage_events[i][idx_writting]
                    duration = event_curr.time_since(event_prev)
                    # duration = event_prev.time_till(event_curr)
                    print("getEmptyBuffer: Stage {} duration: {:.4f} ms".format(i, duration), flush = True)
                event_prev = event_curr
                event_curr = self.list_output_event[idx_writting]
                duration = event_curr.time_since(event_prev)
                print("getEmptyBuffer: Stage out duration: {:.4f} ms".format(duration), flush = True)

            
            # 这里进行内存拷贝
            self.ring_buf.copyOutput(idx_writting)
            self.ring_buf.setReading(idx_writting) # 设置流水完成标识
            # print("put1", flush = True)
            try:
                self.queuePost.put(idx_writting, block=True, timeout=None)
                # print("put1: {}".format(idx_writting), flush = True)
            except:
                pass
            # self.ring_buf.setEmpty(idx_writting) # 设置读取数据完标识
            # # TODO: 这里应该使用另一个线程/进程来读取数据

        # print("getEmptyBuffer: done", flush = True)
        return idx_buf

    def flushOutput(self, isWait=False, is_print_stage_time=False):
        # print("flushOutput: isWait = {}".format(isWait), flush = True)
        index_writting_first = self.ring_buf.index_writting_first.value
        index_writting_last = self.ring_buf.index_writting_last.value
        idx_writting = index_writting_first
        while True:
            event_writting = self.list_output_event[idx_writting]

            if isWait == False:
                if event_writting.query() == False:
                    break
            else:
                # 等待一次完整流水完成
                event_writting.synchronize()
            
            # 如果使用 managed memory
            if self.use_managed_memory == True:
                # 先进行同步 再进行内存拷贝
                self.context.synchronize()
            # print("getEmptyBuffer: idx_writting = {}".format(idx_writting), flush = True)
                

            # 调试打印各个 event 间的时间
            if is_print_stage_time == True:
                event_curr = self.list_input_event[idx_writting]
                for i in range(len(self.ring_buf.list_stage_events)):
                    event_prev = event_curr
                    event_curr = self.ring_buf.list_stage_events[i][idx_writting]
                    duration = event_curr.time_since(event_prev)
                    # duration = event_prev.time_till(event_curr)
                    print("flushOutput: Stage {} duration: {:.4f} ms".format(i, duration), flush = True)
                event_prev = event_curr
                event_curr = self.list_output_event[idx_writting]
                duration = event_curr.time_since(event_prev)
                print("flushOutput: Stage out duration: {:.4f} ms".format(duration), flush = True)


            # 这里进行内存拷贝
            self.ring_buf.copyOutput(idx_writting)
            self.ring_buf.setReading(idx_writting) # 设置流水完成标识
            # print("put2", flush = True)
            try:
                self.queuePost.put(idx_writting, block=True, timeout=None)
                # print("put2: {}".format(idx_writting), flush = True)
            except:
                pass
            
            # 关闭后处理进程后, 加入这里以释放 buffer
            # idx_tmp = self.queuePost.get(block=True, timeout=None)
            # self.ring_buf.setEmpty(idx_writting) # 设置读取数据完标识
            # TODO: 这里应该使用另一个线程/进程来读取数据
            
            if idx_writting == index_writting_last:
                break
            idx_writting = (idx_writting + 1) % self.ring_buf.ring_len.value
        
        # print("flushOutput: done", flush = True)

    def run(self, num_inferences, is_print_stage_time=False):

        self.queuePost = Queue(maxsize=self.ring_buf.ring_len.value)
        
        # 启动后处理进程
        post_process = POST_PROCESS(self.queuePost, self.ring_buf)
        post_process.start()

        # 开始推理
        for idx_infer in range(num_inferences):
            # print("idx_infer = {}".format(idx_infer), flush = True)

            idx_buf = self.getEmptyBuffer(is_print_stage_time)

            # TODO: 处理实际数据时, 这里需要从 输入缓冲区 拷贝数据
            # 如果没有输入, 就等待输入

            # 向 device memory 拷贝输入数据
            event_curr = self.list_input_event[idx_buf]
            if self.use_managed_memory == False:
                idx_input_stream = idx_infer % len(self.list_input_stream)
                input_stream = self.list_input_stream[idx_input_stream]
                # 不使用 managed memory 时, 显式数据拷贝
                for mem_pair in self.ring_buf.list_stage_input_mem_pairs[0][idx_buf]:
                    cuda.memcpy_dtoh_async(mem_pair.host, mem_pair.device, input_stream)
            else:
                idx_input_stream = idx_infer % len(self.list_pipeline_stage_engine[0])
                input_stream = self.list_pipeline_stage_engine[0][idx_input_stream].stream
            # 等待数据拷贝完成
            event_curr.record(input_stream)

            # 流水执行 trt engine
            # print("idx_buf = {}; 流水执行 trt engine".format(idx_buf), flush = True)
            for idx_stage in range(self.num_pipeline_stages):

                # print("idx_stage = {}".format(idx_stage))
                idx_stream = idx_infer % len(self.list_pipeline_stage_engine[idx_stage])
                tmp_stream = self.list_pipeline_stage_engine[idx_stage][idx_stream].stream
                # tmp_engine = self.list_pipeline_stage_engine[idx_stage][idx_stream].trt_engine
                tmp_context = self.list_pipeline_stage_engine[idx_stage][idx_stream].trt_context
                # print("idx_stream = {}".format(idx_stream))
                
                input_bindings = self.ring_buf.list_stage_input_mem_devices[idx_stage][idx_buf]
                output_bindings = self.ring_buf.list_stage_output_mem_devices[idx_stage][idx_buf]

                bindings = input_bindings + output_bindings

                event_prev = event_curr
                event_curr = self.ring_buf.list_stage_events[idx_stage][idx_buf]

                # 按 stage 将 trt engine 和 event 放入 stream 执行
                tmp_stream.wait_for_event(event_prev) # 等待前一个 stage 完成
                tmp_context.execute_async_v2(bindings, tmp_stream.handle, None) # input_consumed: capsule = None, 似乎支持在 输入buffer 用完之后发出一个 event
                # 该 cuda event 表示上边的 trt engine context 执行完成
                event_curr.record(tmp_stream)

                # print("idx_buf = {}; idx_stage = {}; stream = {}; event record".format(idx_buf, idx_stage, tmp_stream))
            
            # 向 host memory 拷贝输出数据
            event_prev = self.ring_buf.list_stage_events[-1][idx_buf]
            event_curr = self.list_output_event[idx_buf]
            if self.use_managed_memory == False:
                # 不使用 managed memory 时, 显式数据拷贝

                # 先等待推理的最后一阶段执行完成, 使用 输出专用 stream
                idx_output_stream = idx_infer % len(self.list_output_stream)
                output_stream = self.list_output_stream[idx_output_stream]
                output_stream.wait_for_event(event_prev)

                # 然后显式数据拷贝, 使用 输出专用 stream
                for mem_pair in self.ring_buf.list_stage_output_mem_pairs[-1][idx_buf]:
                    cuda.memcpy_htod_async(mem_pair.device, mem_pair.host, output_stream)

                # 最后 插入 输出 event, 使用 输出专用 stream
                event_curr.record(output_stream)

            else:
                # 使用 managed memory 时
                # 在推理的最后一阶段之后 插入 输出 event, 使用最后一阶段的 stream
                idx_output_stream = idx_infer % len(self.list_pipeline_stage_engine[-1])
                output_stream = self.list_pipeline_stage_engine[-1][idx_output_stream].stream
                event_curr.record(output_stream)

            self.flushOutput(False, is_print_stage_time)
            
        self.flushOutput(True, is_print_stage_time)

        # 至此所有推理都完成了, 流水线空了
        
        # 通知后处理线程: 处理完缓冲区中现有数据, 然后 结束后处理线程
        self.queuePost.put(None, block=True, timeout=None)

        # 等待后处理线程执行完毕
        post_process.join()

class POST_PROCESS(Process):
    # 消息队列, 共享变量, 共享输出缓冲 np array
    def __init__(self, queuePost, ring_buf):
        Process.__init__(self) # 必须步骤
        self.queuePost = queuePost
        self.ring_len = ring_buf.ring_len
        self.dictStatus = ring_buf.dictStatus
        self.list_status = ring_buf.list_status

        self.occupy_count = ring_buf.occupy_count
        self.list_outputs = ring_buf.list_outputs # 输出缓冲

        self.index_writting_first = ring_buf.index_writting_first
        self.index_writting_last = ring_buf.index_writting_last
        self.index_reading_first = ring_buf.index_reading_first
        self.index_reading_last = ring_buf.index_reading_last
    
    def setEmpty(self, index_released):
        self.list_status[index_released] = self.dictStatus["empty"]
        self.occupy_count.value -= 1

        # self.index_reading_first.value = (self.index_reading_first.value + 1) % self.ring_len.value
        for _ in range(self.ring_len.value):
            tmp_index = (self.index_reading_first.value + 1) % self.ring_len.value
            if self.list_status[tmp_index] != self.dictStatus["reading"]:
                self.index_reading_first.value = tmp_index
            else:
                break

    def getReadingIndex(self):
        if self.list_status[self.index_reading_first.value] != self.dictStatus["reading"]:
            return None
        else:
            return self.index_reading_first.value

    def run(self):  # 入口是名字为run的方法
        # print("开始做一个任务啦")
        # time.sleep(1)  # 用time.sleep模拟任务耗时
        # print("这个任务结束啦")

        idx_reading = None
        while True:

            # 是否使用队列, 开销如何
            # print("get", flush = True)
            idx_tmp = self.queuePost.get(block=True, timeout=None)
            # print("get complete", flush = True)
            # print("idx_tmp = {}".format(idx_tmp), flush = True)
            if idx_tmp == None:
                break
                
            # idx_reading = self.getReadingIndex()
            # if idx_reading == None:
            #     continue


            # TODO: 这里读取数据并进行后处理


            # self.setEmpty(idx_reading)
            self.setEmpty(idx_tmp)
