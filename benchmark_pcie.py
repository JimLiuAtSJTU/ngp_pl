

import torch

import time


# pcie 3.0 x4 -> x 16   1.5x faster for data transfer.



def main():

    A=torch.randn((5000,5000*100))

    t0=time.time()
    A0=A.to(0)
    t1=time.time()

    A1=A.to(1)

    t2=time.time()
    print('------------gpu0-----------')
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
    print('------------gpu1-----------')

    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(1) / 1024 / 1024 / 1024))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(1) / 1024 / 1024 / 1024))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(1) / 1024 / 1024 / 1024))



    print(f'time: {t1-t0},{t2-t1}, speed ratio={1/((t1-t0)/(t2-t1))}')







if __name__=="__main__":
    main()