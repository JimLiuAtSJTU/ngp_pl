import time

from pynvml import *

import datetime




idle_time_limit=10

scan_interval=2

inspecting_GPU_INDEX=1


def wait():
    nvmlInit()

    print(f'nvml check gpu:{inspecting_GPU_INDEX}')

    idle_time=0
    while True:
        print(datetime.datetime.now())
        h = nvmlDeviceGetHandleByIndex(inspecting_GPU_INDEX)
        info = nvmlDeviceGetMemoryInfo(h)

        t,f,u=info.total/1024**5,info.free/1024**5,info.used/1024**5
        print(f'total    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')

        ratio =  u / t
        assert ratio>=0 and ratio<=1
        print(f'memory used ratio {ratio}')
        if ratio<0.10: # almost no gpu memory usage
            idle_time=idle_time+1

            print(f'find GPU idle. idle time {idle_time}')

        else:
            idle_time=0 # gpu memory usage >10%, not idle
            print('gpu occupied.')
        if idle_time>idle_time_limit:
            print(f'find gpu constantly idle. exit.')
            break
        time.sleep(60*scan_interval) # wait for 1 minutes

if __name__=='__main__':
    wait()