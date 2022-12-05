import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import json
import os
import csv
from matplotlib.widgets import RectangleSelector
from datetime import datetime
from scipy.optimize import curve_fit

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}

mpl.rc('font', **font)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

# path = r'C:\Users\HOEKHJ\Dev\InterferometryPython\export\PROC_20221201163125\powerlawFittingData.json'
# path = r'C:\Users\HOEKHJ\Dev\InterferometryPython\export\PROC_20221201164649\powerlawFittingData.json'
path = r'C:\Users\HOEKHJ\Dev\InterferometryPython\export\PROC_20221202113132\powerlawFittingData.json'

with open(path) as f:
    data = json.load(f)


analyzeImages = [int(p) for p in data['analyzeImages'].split(',')]

colors = plt.cm.viridis(np.linspace(0, 1, len(analyzeImages)))



fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for idx in range(len(analyzeImages)):
    x = data['data'][str(idx)]['x_full']
    y = data['data'][str(idx)]['y_full']
    xnew = data['data'][str(idx)]['xnew']
    xrange = data['data'][str(idx)]['xrange']
    ynew = data['data'][str(idx)]['ynew']

    y = y[0:find_nearest(x, xrange[-1])[1]]
    x = x[0:find_nearest(x, xrange[-1])[1]]

    y = [yi - ynew[-1] for yi in y]
    ynew = [yi - ynew[-1] for yi in ynew]
    x = [xi - xnew[0] for xi in x]
    xnew = [xi - xnew[0] for xi in xnew]
    time = int(data['data'][str(idx)]['timeFromStart'])

    # print(f"{xrange[-1]=}")
    # print(find_nearest(x, xrange[-1]))

    ax.scatter(x, y, label=f"t={time}s", s=2, color=colors[idx])
    ax.plot(xnew, ynew, '-.', linewidth=2, color=colors[idx])

plt.legend()
ax.set_xlabel('Distance to contact line [um]')
ax.set_ylabel('Height [um]')
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(path), f"allprofiles_powerlaw.png"), dpi=300)




fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for idx in range(len(analyzeImages)):
    x = data['data'][str(idx)]['x_full']
    y = data['data'][str(idx)]['y_full']
    xnew = data['data'][str(idx)]['xnew']
    xrange = data['data'][str(idx)]['xrange']
    ynew = data['data'][str(idx)]['ynew']

    # remove noisy data in end: beyond fitting range
    y = y[0:find_nearest(x, xrange[-1])[1]]
    x = x[0:find_nearest(x, xrange[-1])[1]]

    y = [yi - np.mean(y[-10:-1]) for yi in y]
    x = [xi - xnew[0] for xi in x]

    time = int(data['data'][str(idx)]['timeFromStart'])

    ax.scatter(x, y, label=f"t={time}s", s=2, color=colors[idx])
plt.legend()
ax.set_xlabel('Distance to contact line [um]')
ax.set_ylabel('Height [um]')
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(path), f"allprofiles_standard.png"), dpi=300)

plt.show()