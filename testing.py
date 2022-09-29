import numpy as np


def smooth_step(arr, width):
    width = width-1
    # arr = np.hstack((np.zeros(width).astype(float), arr, np.zeros(width).astype(float)))
    steps = np.diff(arr)
    locs = np.where(np.abs(steps) == 1)[0]
    for loc in locs:
        if steps[loc] == 1:
            arr = np.hstack((np.zeros(width).astype(float), arr))
            print(f"{len(arr)=}")
            arr[(loc-width+width):(loc+width+1+width)] = 1 / (1 + np.exp(-3*(np.arange(0, width*2+1)-1.5)))
            print(f"{len(arr)=}")
            arr = np.delete(arr, np.arange(width), None)
            print(f"{len(arr)=}")
        else:
            arr[(loc-width):(loc+width+1)] = -1 / (1 + np.exp(-3*(np.arange(0, width*2+1)-1.5)))+1

    # np.delete(arr, len(arr)-np.arange(width)-1, None)
    return arr



a = np.array([0,1,1,1,1,1,1,1,1,1,1,1,1]).astype(float)
print(a, len(a))
b = smooth_step(a, 6)
print(b, len(b))
# b[-3:] = 0
# print(b)
# c = smooth_step(b, 2)
# print(c)