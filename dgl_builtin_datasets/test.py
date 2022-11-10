# from sklearn.metrics import roc_auc_score
# import random
# import torch

# a = torch.tensor([0.925363139387929, 0.5, 0.9253069649024872, 0.9252646238617597, 0.8328786894026335, 0.9271960325986272, 0.9243159246947011, 0.921751922466676, 0.7734648086337816, 0.9260898155887178, 0.9253914177003281])

# for i in a:
#     if i < 0.5:
#         i = 1 - i
#     print((i * 100).item())
    
import numpy as np
import cupy as cp
import time
time0=time.time()
x=np.ones((1024,512,4,4))*1024.
y=np.ones((1024,512,4,4))*512.3254
time1=time.time()
print('time for cpu:',time1 - time0)
for i in range(50):
    z=x*y
print('average time for 20 times cpu:',(time.time()-time1)/20.)

time0=time.time()
x=cp.ones((1024,512,4,4))*1024.
y=cp.ones((1024,512,4,4))*512.3254
time1=time.time()
print('time for gpu:',time1 - time0)
for i in range(50):
    z=x*y
print('average time for 20 times gpu:',(time.time()-time1)/20.)


