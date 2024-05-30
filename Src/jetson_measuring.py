# coding=utf-8
import numpy as np
from timeit import default_timer as timer # 实际上是调用了 time.perf_counter() 返回高精度时间
from multiprocessing import Process, Lock
import multiprocessing
import time
from pylibi2c import I2CDevice


class orin_nx_measuring:

    def __init__(self, sample_interval = 0.1) -> None:
        self.SHT40 = I2CDevice('/dev/i2c-7', 0x44, iaddr_bytes=0)
        self.cmd = bytes([0xFD])
        self.lockMeasure = Lock()

        try:
            self.lockMeasure.release()
        except ValueError:
            pass
        
        # 控制是否重启测量, 会重置测量数据
        self.isRestart = multiprocessing.Value('b', True)
        # 共享标识变量 控制测量停止
        self.isMeasuring = multiprocessing.Value('b', True)
        # 共享标识变量 控制是否写回测量结果数据
        self.isGetData = multiprocessing.Value('b', False)
        # 启动阶段跳过一段时间
        self.skip_time = multiprocessing.Value('f', 0)

        # 分配共享变量
        # 采样间隔 0.1s
        self.sample_interval = sample_interval
        # 测量次数
        self.measurement_count = multiprocessing.Value('i', 0)
        # 测量持续时间 s
        self.measurement_duration = multiprocessing.Value('f', 0)
        # VDD 功率 W
        self.power_vdd = multiprocessing.Value('f', 0)
        # 总能耗 J
        self.energy_vdd = multiprocessing.Value('f', 0)
        # 环境温度 ℃
        self.temp_ambient = multiprocessing.Value('f', -274)
        # SoC 不同部件最高瞬时温度 ℃
        self.temp_max = multiprocessing.Value('f', -274)
        # CPU 温度 ℃
        self.temp_cpu = multiprocessing.Value('f', -274)
        # GPU 温度 ℃
        self.temp_gpu = multiprocessing.Value('f', -274)
        # CV0 温度 ℃
        self.temp_cv0 = multiprocessing.Value('f', -274)
        # CV1 温度 ℃
        self.temp_cv1 = multiprocessing.Value('f', -274)
        # CV2 温度 ℃
        self.temp_cv2 = multiprocessing.Value('f', -274)
        # SOC0 温度 ℃
        self.temp_soc0 = multiprocessing.Value('f', -274)
        # SOC1 温度 ℃
        self.temp_soc1 = multiprocessing.Value('f', -274)
        # SOC2 温度 ℃
        self.temp_soc2 = multiprocessing.Value('f', -274)
        # tj 温度 ℃
        self.temp_tj = multiprocessing.Value('f', -274)
        
        # CPU 频率 MHz
        self.freq_cpu0 = multiprocessing.Value('f', 0)
        self.freq_cpu1 = multiprocessing.Value('f', 0)
        self.freq_cpu2 = multiprocessing.Value('f', 0)
        self.freq_cpu3 = multiprocessing.Value('f', 0)
        self.freq_cpu4 = multiprocessing.Value('f', 0)
        self.freq_cpu5 = multiprocessing.Value('f', 0)
        self.freq_cpu6 = multiprocessing.Value('f', 0)
        self.freq_cpu7 = multiprocessing.Value('f', 0)
        # GPU 频率 MHz
        self.freq_gpu = multiprocessing.Value('f', 0)
        # 内存频率 MHz
        self.freq_mem = multiprocessing.Value('f', 0)

        # "/sys/class/devfreq/17000000.ga10b/cur_freq"
        # "/sys/class/devfreq/17000000.ga10b/min_freq"
        # "/sys/class/devfreq/17000000.ga10b/max_freq"

        # "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"
        # "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
        # "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
        # "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq"

        # "/sys/kernel/nvpmodel_emc_cap/emc_iso_cap"
        # "/sys/kernel/debug/bpmp/debug/clk/emc"

        # 测量进程
        # self.MeasurementProcess = Process(target=self.measuring, args=(self.lockMeasure,)) # python multiprocessing.Process
        self.MeasurementProcess = Process(target=self.measuring) # python multiprocessing.Process
        # self.MeasurementProcessOS = None # python psutil.Process

    def __enter__(self):
        return self

    def init_data(self):
        self.measurement_count.value = 0
        self.measurement_start_time = 0
        self.measurement_end_time = 0
        self.measurement_duration.value = 0
        self.temp_ambient.value = -274
        self.temp_max.value = -274
        self.temp_cpu.value = -274
        self.temp_gpu.value = -274
        self.temp_cv0.value = -274
        self.temp_cv1.value = -274
        self.temp_cv2.value = -274
        self.temp_soc0.value = -274
        self.temp_soc1.value = -274
        self.temp_soc2.value = -274
        self.temp_tj.value = -274
        self.power_vdd.value = 0
        self.energy_vdd.value = 0
        self.freq_cpu0.value = 0
        self.freq_cpu1.value = 0
        self.freq_cpu2.value = 0
        self.freq_cpu3.value = 0
        self.freq_cpu4.value = 0
        self.freq_cpu5.value = 0
        self.freq_cpu6.value = 0
        self.freq_cpu7.value = 0
        self.freq_gpu.value = 0
        self.freq_mem.value = 0

    def measuring(self):
        measurement_count = 0
        power_vdd_sum = 0
        temp_ambient_sum = 0
        temp_max = 0
        temp_cpu_sum = 0
        temp_gpu_sum = 0
        temp_cv0_sum = 0
        temp_cv1_sum = 0
        temp_cv2_sum = 0
        temp_soc0_sum = 0
        temp_soc1_sum = 0
        temp_soc2_sum = 0
        temp_tj_sum = 0

        freq_cpu0_sum, freq_cpu1_sum, freq_cpu2_sum, freq_cpu3_sum, freq_cpu4_sum, freq_cpu5_sum, freq_cpu6_sum, freq_cpu7_sum, = 0, 0, 0, 0, 0, 0, 0, 0
        freq_gpu_sum = 0
        freq_mem_sum = 0

        once_end_time = timer()
        # measurement_begin_time = once_end_time
        while self.isMeasuring.value == True:

            self.lockMeasure.acquire(block=True, timeout=None)
            self.lockMeasure.release()

            # 如果重启测量 则 重置数据
            if self.isRestart.value == True:
                once_end_time = timer()
                # measurement_begin_time = once_end_time
                measurement_count = 0
                power_vdd_sum = 0
                temp_ambient_sum = 0
                temp_max = 0
                temp_cpu_sum = 0
                temp_gpu_sum = 0
                temp_cv0_sum = 0
                temp_cv1_sum = 0
                temp_cv2_sum = 0
                temp_soc0_sum = 0
                temp_soc1_sum = 0
                temp_soc2_sum = 0
                temp_tj_sum = 0

                freq_cpu0_sum, freq_cpu1_sum, freq_cpu2_sum, freq_cpu3_sum, freq_cpu4_sum, freq_cpu5_sum, freq_cpu6_sum, freq_cpu7_sum, = 0, 0, 0, 0, 0, 0, 0, 0
                freq_gpu_sum, freq_mem_sum = 0, 0

                self.isRestart.value = False

                # 启动阶段跳过一段时间
                if self.skip_time.value > 0.0:
                    time.sleep(self.skip_time.value)

            measurement_count += 1

            temp_max_new, temp_cpu, temp_gpu, temp_cv0, temp_cv1, temp_cv2, temp_soc0, temp_soc1, temp_soc2, temp_tj = self.get_curr_temp()

            temp_max = np.max([temp_max, temp_max_new])
            temp_cpu_sum += temp_cpu
            temp_gpu_sum += temp_gpu
            temp_cv0_sum += temp_cv0
            temp_cv1_sum += temp_cv1
            temp_cv2_sum += temp_cv2
            temp_soc0_sum += temp_soc0
            temp_soc1_sum += temp_soc1
            temp_soc2_sum += temp_soc2
            temp_tj_sum += temp_tj

            power_vdd = self.get_curr_power_vdd()
            power_vdd_sum += power_vdd

            freq_cpu0, freq_cpu1, freq_cpu2, freq_cpu3, freq_cpu4, freq_cpu5, freq_cpu6, freq_cpu7, freq_gpu, freq_mem = self.get_curr_freq()
            freq_cpu0_sum += freq_cpu0
            freq_cpu1_sum += freq_cpu1
            freq_cpu2_sum += freq_cpu2
            freq_cpu3_sum += freq_cpu3
            freq_cpu4_sum += freq_cpu4
            freq_cpu5_sum += freq_cpu5
            freq_cpu6_sum += freq_cpu6
            freq_cpu7_sum += freq_cpu7
            freq_gpu_sum += freq_gpu
            freq_mem_sum += freq_mem

            size = self.SHT40.write(0x0, self.cmd)
            time.sleep(1e-2) # 等待 10 ms, 给 SHT40 测量时间
            TempHumiData = self.SHT40.ioctl_read(0x0, 6) # 读取温湿度数据
            Temperature = (float(TempHumiData[0])*256+TempHumiData[1])*175/65535-45
            # Humidity = (float(TempHumiData[3])*256+TempHumiData[4])*125/65535-6
            temp_ambient_sum += Temperature

            # print("measurement_count = {}".format(measurement_count))

            if self.isGetData.value == True:
                # measurement_duration = timer() - measurement_begin_time
                # self.measurement_duration.value = measurement_duration
                self.measurement_count.value = measurement_count
                self.temp_max.value = temp_max
                self.temp_cpu.value = temp_cpu_sum / measurement_count
                self.temp_gpu.value = temp_gpu_sum / measurement_count
                self.temp_cv0.value = temp_cv0_sum / measurement_count
                self.temp_cv1.value = temp_cv1_sum / measurement_count
                self.temp_cv2.value = temp_cv2_sum / measurement_count
                self.temp_soc0.value = temp_soc0_sum / measurement_count
                self.temp_soc1.value = temp_soc1_sum / measurement_count
                self.temp_soc2.value = temp_soc2_sum / measurement_count
                self.temp_tj.value = temp_tj_sum / measurement_count
                self.power_vdd.value = power_vdd_sum / measurement_count
                self.freq_cpu0.value = freq_cpu0_sum / measurement_count
                self.freq_cpu1.value = freq_cpu1_sum / measurement_count
                self.freq_cpu2.value = freq_cpu2_sum / measurement_count
                self.freq_cpu3.value = freq_cpu3_sum / measurement_count
                self.freq_cpu4.value = freq_cpu4_sum / measurement_count
                self.freq_cpu5.value = freq_cpu5_sum / measurement_count
                self.freq_cpu6.value = freq_cpu6_sum / measurement_count
                self.freq_cpu7.value = freq_cpu7_sum / measurement_count
                self.freq_gpu.value = freq_gpu_sum / measurement_count
                self.freq_mem.value = freq_mem_sum / measurement_count
                self.temp_ambient.value = temp_ambient_sum / measurement_count

                # self.energy_vdd.value = power_vdd_sum / measurement_count * measurement_duration
                self.isGetData.value = False

            # 根据测量间隔延时一段时间
            once_start_time = once_end_time
            once_end_time = timer()  # 获取当前时间
            remaining_idle_time = self.sample_interval - (once_end_time - once_start_time) # 计算剩余时间
            time.sleep(np.max([0, remaining_idle_time-1*1e-6])) # 休眠

    # 关闭测量进程
    def __del__(self):
        if self.MeasurementProcess == None:
            return

        self.isMeasuring.value = False
        try:
            self.lockMeasure.release()
        except ValueError:
            pass

        if self.MeasurementProcess.is_alive() == True:
            self.MeasurementProcess.join()
        # self.MeasurementProcess.close()
        # del self.MeasurementProcess

    def __exit__(self, exc_type, exc_val, exc_tb):
        # print(exc_type)
        # print(exc_val)
        # print(exc_tb)

        if self.MeasurementProcess == None:
            return

        self.isMeasuring.value = False
        try:
            self.lockMeasure.release()
        except ValueError:
            pass

        if self.MeasurementProcess.is_alive() == True:
            self.MeasurementProcess.join()

    def exit(self):
        if self.MeasurementProcess == None:
            return

        self.isMeasuring.value = False
        try:
            self.lockMeasure.release()
        except ValueError:
            pass

        if self.MeasurementProcess.is_alive() == True:
            self.MeasurementProcess.join()
        self.MeasurementProcess.close()
        del self.MeasurementProcess
        self.MeasurementProcess = None

    # 启动新进程 作为测量进程, 异步测量 Jetson Orin NX 平台 能耗/温度 等 状态
    def start(self, skip_time = 0.0):
        self.init_data()
        self.skip_time.value = skip_time
        self.isRestart.value = True
        self.isMeasuring.value = True
        self.isGetData.value = False

        try:
            self.lockMeasure.release() # 重新启动测量
            self.measurement_start_time = timer()
        except ValueError:
            pass

        if self.MeasurementProcess == None:
            self.MeasurementProcess = Process(target=self.measuring) # python multiprocessing.Process

        if self.MeasurementProcess.is_alive() == False:
            self.MeasurementProcess.start() # 第一次启动测量
            self.measurement_start_time = timer()
        
    # 关闭测量进程 停止测量 计算均值等统计值
    def stop(self):
        self.measurement_end_time = timer()
        self.isGetData.value = True
        while self.isGetData.value == True:
            time.sleep(0.01)
            
        self.lockMeasure.acquire(block=True, timeout=None)

        self.measurement_duration.value = self.measurement_end_time - self.measurement_start_time
        self.energy_vdd.value = self.power_vdd.value * self.measurement_duration.value

    # 读取最高温度 (℃)
    def get_curr_temp(self):
        MaxTemp = -274.0
        
        FilePath = "/sys/devices/virtual/thermal/thermal_zone0/temp"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        TempCPU = float(strOutput)/1000 # CPU温度
        MaxTemp = np.max([TempCPU, MaxTemp])

        
        FilePath = "/sys/devices/virtual/thermal/thermal_zone1/temp"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        TempGPU = float(strOutput)/1000 # GPU 温度
        MaxTemp = np.max([TempGPU, MaxTemp])

        FilePath = "/sys/devices/virtual/thermal/thermal_zone2/temp"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        TempCV0 = float(strOutput)/1000 # CV0 温度
        MaxTemp = np.max([TempCV0, MaxTemp])

        FilePath = "/sys/devices/virtual/thermal/thermal_zone3/temp"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        TempCV1 = float(strOutput)/1000 # CV1 温度
        MaxTemp = np.max([TempCV1, MaxTemp])

        FilePath = "/sys/devices/virtual/thermal/thermal_zone4/temp"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        TempCV2 = float(strOutput)/1000 # CV2 温度
        MaxTemp = np.max([TempCV2, MaxTemp])

        FilePath = "/sys/devices/virtual/thermal/thermal_zone5/temp"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        TempSOC0 = float(strOutput)/1000 # SOC0 温度
        MaxTemp = np.max([TempSOC0, MaxTemp])

        FilePath = "/sys/devices/virtual/thermal/thermal_zone7/temp"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        TempSOC1 = float(strOutput)/1000 # SOC1 温度
        MaxTemp = np.max([TempSOC1, MaxTemp])

        FilePath = "/sys/devices/virtual/thermal/thermal_zone7/temp"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        TempSOC2 = float(strOutput)/1000 # SOC2 温度
        MaxTemp = np.max([TempSOC2, MaxTemp])

        FilePath = "/sys/devices/virtual/thermal/thermal_zone8/temp"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        Temp_tj = float(strOutput)/1000 # tj 温度
        MaxTemp = np.max([Temp_tj, MaxTemp])

        return MaxTemp, TempCPU, TempGPU, TempCV0, TempCV1, TempCV2, TempSOC0, TempSOC1, TempSOC2, Temp_tj
    
    # 读取核心板总功率 (W)
    def get_curr_power_vdd(self):

        FilePath = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        VDD = float(strOutput)/1000 # 电压 mV

        FilePath = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/curr1_input"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        IDD = float(strOutput)/1000 # 电压 mA

        PowerVDD = VDD * IDD
        return PowerVDD
    
    # 读取频率
    def get_curr_freq(self):
        
        # 读取 CPU 0-7 频率 MHz
        FilePath = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        freq_cpu0 = float(strOutput) / 1000

        FilePath = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_cur_freq"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        freq_cpu1 = float(strOutput) / 1000

        FilePath = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_cur_freq"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        freq_cpu2 = float(strOutput) / 1000

        FilePath = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_cur_freq"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        freq_cpu3 = float(strOutput) / 1000

        FilePath = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_cur_freq"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        freq_cpu4 = float(strOutput) / 1000

        FilePath = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_cur_freq"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        freq_cpu5 = float(strOutput) / 1000

        FilePath = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_cur_freq"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        freq_cpu6 = float(strOutput) / 1000

        FilePath = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_cur_freq"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        freq_cpu7 = float(strOutput) / 1000


        # 读取 GPU 频率 MHz
        FilePath = "/sys/class/devfreq/17000000.ga10b/cur_freq"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        freq_gpu = float(strOutput) / 1000000


        # 读取 内存 频率 MHz
        FilePath = "/sys/kernel/debug/bpmp/debug/clk/emc/rate"
        with open(FilePath, "r") as OutFile:
            strOutput = OutFile.readline()
            OutFile.close()
        freq_mem = float(strOutput) / 1000000

        return freq_cpu0, freq_cpu1, freq_cpu2, freq_cpu3, freq_cpu4, freq_cpu5, freq_cpu6, freq_cpu7, freq_gpu, freq_mem

