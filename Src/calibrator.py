# coding=utf-8
import tensorrt as trt
import os, copy

import pycuda.driver as cuda
import pycuda.autoinit
# from PIL import Image
import numpy as np

class VOID_CALIBRATOR(trt.IInt8EntropyCalibrator2):

    # 生成给定 shape, 给定范围 的 随机数
    # 输入: 
    # 1. 网络输入数据的 shape, shape[0] 是 batch_size
    # 2. 生成随机数范围 list_range
    # 3. 生成 batch 的个数
    # 4. 校准缓存文件路径 xxx.cache
    def __init__(self, list_input_shape, list_range, batch_size, num_batches, cache_file_dir):

        # print("VOID_CALIBRATOR: in", flush=True)

        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_index = 0
        self.cache_file = cache_file_dir
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.data = []
        self.device_input = []
        for i in range(len(list_input_shape)):
            if isinstance(list_range[i], list):
                list_tmp_input = []
                for _ in range(num_batches):
                    list_tmp_input.append(np.random.random(list_input_shape[i]).astype(np.float32) * (list_range[i][1] - list_range[i][0]) + list_range[i][0])
                    # print("VOID_CALIBRATOR: input {} shape: {}".format(i, list_input_shape[i]), flush=True)
                self.data.append(list_tmp_input)
            else:
                list_tmp_input = []
                for _ in range(num_batches):
                    list_tmp_input.append(copy.deepcopy(list_range[i]))
                    # print("VOID_CALIBRATOR: input {} = {}".format(i, list_range[i]), flush=True)
                self.data.append(list_tmp_input)

            self.device_input.append(cuda.mem_alloc(self.data[-1][0].nbytes))

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        # 生成给定范围的随机数
        # tmp_data_shape = [*input_shape]
        # tmp_data_shape[0] = tmp_data_shape[0] * num_batches
        # self.data = np.random.random(tmp_data_shape).astype(np.float32) * (list_range[1] - list_range[0]) + list_range[0]
        # self.batch_size = input_shape[0]
        

        # Allocate enough memory for a whole batch.
        # self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        # print("get_batch_size: in", flush=True)
        return self.batch_size
    
    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        # print("get_batch: in", flush=True)
        if self.batch_index >= self.num_batches:
            return None

        # list_device_input = []
        for i in range(len(self.data)):
            batch = self.data[i][self.batch_index]
            cuda.memcpy_htod(self.device_input[i], batch)
            # print("get_batch: input {} shape: {}".format(i, batch.shape))

        self.batch_index += 1
        return self.device_input

        # current_batch = int(self.current_index / self.batch_size)
        # # if current_batch % 10 == 0:
        # #     print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        # batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        # cuda.memcpy_htod(self.device_input, batch)
        # self.current_index += self.batch_size
        # return [self.device_input]
    
    def read_calibration_cache(self):
        # print("read_calibration_cache: in", flush=True)
        # return None
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        # print("write_calibration_cache: in", flush=True)
        # return None
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# # Returns a numpy buffer of shape (num_images, 1, 28, 28)
# def load_mnist_data(filepath):
#     with open(filepath, "rb") as f:
#         raw_buf = np.fromstring(f.read(), dtype=np.uint8)
#     # Make sure the magic number is what we expect
#     assert raw_buf[0:4].view(">i4")[0] == 2051
#     num_images = raw_buf[4:8].view(">i4")[0]
#     image_c = 1
#     image_h = raw_buf[8:12].view(">i4")[0]
#     image_w = raw_buf[12:16].view(">i4")[0]
#     # Need to scale all values to the range of [0, 1]
#     return np.ascontiguousarray((raw_buf[16:] / 255.0).astype(np.float32).reshape(num_images, image_c, image_h, image_w))

# # Returns a numpy buffer of shape (num_images)
# def load_mnist_labels(filepath):
#     with open(filepath, "rb") as f:
#         raw_buf = np.fromstring(f.read(), dtype=np.uint8)
#     # Make sure the magic number is what we expect
#     assert raw_buf[0:4].view(">i4")[0] == 2049
#     num_labels = raw_buf[4:8].view(">i4")[0]
#     return np.ascontiguousarray(raw_buf[8:].astype(np.int32).reshape(num_labels))

# MNISTEntropyCalibrator
# class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
#     def __init__(self, training_data, cache_file, batch_size=64):
#         # Whenever you specify a custom constructor for a TensorRT class,
#         # you MUST call the constructor of the parent explicitly.
#         trt.IInt8EntropyCalibrator2.__init__(self)

#         self.cache_file = cache_file

#         # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
#         self.data = load_mnist_data(training_data)
#         self.batch_size = batch_size
#         self.current_index = 0

#         # Allocate enough memory for a whole batch.
#         self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

#     def get_batch_size(self):
#         return self.batch_size

#     # TensorRT passes along the names of the engine bindings to the get_batch function.
#     # You don't necessarily have to use them, but they can be useful to understand the order of
#     # the inputs. The bindings list is expected to have the same ordering as 'names'.
#     def get_batch(self, names):
#         if self.current_index + self.batch_size > self.data.shape[0]:
#             return None

#         current_batch = int(self.current_index / self.batch_size)
#         if current_batch % 10 == 0:
#             print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

#         batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
#         cuda.memcpy_htod(self.device_input, batch)
#         self.current_index += self.batch_size
#         return [self.device_input]


#     def read_calibration_cache(self):
#         # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "rb") as f:
#                 return f.read()

#     def write_calibration_cache(self, cache):
#         with open(self.cache_file, "wb") as f:
#             f.write(cache)


