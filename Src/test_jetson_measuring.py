###############################################################################################
# 测试 jetson_measuring.orin_nx_measuring
###############################################################################################
import time

from jetson_measuring import orin_nx_measuring

if __name__ == "__main__":
    tmp_time = 2

    # 会自动处理测量过程中的子进程, 避免僵尸进程 或 程序不能结束
    with orin_nx_measuring(0.05) as OrinNXMeasuring:

        for i in range(3):

            print("开始测量")
            OrinNXMeasuring.start()
            
            print("延时 {} s".format(tmp_time))
            time.sleep(tmp_time)

            print("停止测量")
            OrinNXMeasuring.stop()

            print("measurement_count = {}".format(OrinNXMeasuring.measurement_count.value))
            print("measurement_duration = {} s".format(OrinNXMeasuring.measurement_duration.value))
            print("power_vdd = {} W".format(OrinNXMeasuring.power_vdd.value))
            print("energy_vdd = {} J".format(OrinNXMeasuring.energy_vdd.value))
            print("temp_ambient = {} ℃".format(OrinNXMeasuring.temp_ambient.value))
            print("temp_max = {} ℃".format(OrinNXMeasuring.temp_max.value))
            print("temp_cpu = {} ℃".format(OrinNXMeasuring.temp_cpu.value))
            print("temp_gpu = {} ℃".format(OrinNXMeasuring.temp_gpu.value))
            print("temp_cv0 = {} ℃".format(OrinNXMeasuring.temp_cv0.value))
            print("temp_cv1 = {} ℃".format(OrinNXMeasuring.temp_cv1.value))
            print("temp_cv2 = {} ℃".format(OrinNXMeasuring.temp_cv2.value))
            print("temp_soc0 = {} ℃".format(OrinNXMeasuring.temp_soc0.value))
            print("temp_soc1 = {} ℃".format(OrinNXMeasuring.temp_soc1.value))
            print("temp_soc2 = {} ℃".format(OrinNXMeasuring.temp_soc2.value))
            print("temp_tj = {} ℃".format(OrinNXMeasuring.temp_tj.value))
            
            print("freq_cpu0 = {} MHz".format(OrinNXMeasuring.freq_cpu0.value))
            print("freq_cpu1 = {} MHz".format(OrinNXMeasuring.freq_cpu1.value))
            print("freq_cpu2 = {} MHz".format(OrinNXMeasuring.freq_cpu2.value))
            print("freq_cpu3 = {} MHz".format(OrinNXMeasuring.freq_cpu3.value))
            print("freq_cpu4 = {} MHz".format(OrinNXMeasuring.freq_cpu4.value))
            print("freq_cpu5 = {} MHz".format(OrinNXMeasuring.freq_cpu5.value))
            print("freq_cpu6 = {} MHz".format(OrinNXMeasuring.freq_cpu6.value))
            print("freq_cpu7 = {} MHz".format(OrinNXMeasuring.freq_cpu7.value))
            print("freq_gpu = {} MHz".format(OrinNXMeasuring.freq_gpu.value))
            print("freq_mem = {} MHz".format(OrinNXMeasuring.freq_mem.value))

            print("")
