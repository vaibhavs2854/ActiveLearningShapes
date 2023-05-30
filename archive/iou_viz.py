#Test
#show histogram
import numpy as np
from matplotlib import pyplot as plt

iou_arr = np.load("/usr/xtmp/vs196/mammoproj/Code/ActiveLearning/iouhist_1.npy")
print(np.min(iou_arr))
print(np.max(iou_arr))
print(np.mean(iou_arr))
iou_arr_aslist = iou_arr.tolist()
plt.plot(iou_arr_aslist)
plt.show()

#Run cell in jupyter notebook