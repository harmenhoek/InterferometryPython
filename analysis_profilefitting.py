import math
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import csv
from matplotlib.widgets import RectangleSelector
from datetime import datetime
from scipy.optimize import curve_fit


'''
Goal: fit power law to data
 
 
1. Load dataset
2. Select data to fit powerlaw to
3. Fit power law f(x) = a*x^b and plot
4. Save profile
5. Repeat for all images.
 
 
NOTE: first datapoint selected is also the alignment point!

'''

class Highlighter(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.x, self.y = x, y
        self.mask = np.zeros(x.shape, dtype=bool)

        self._highlight = ax.scatter([], [], s=200, color='yellow', zorder=10)

        self.selector = RectangleSelector(ax, self, useblit=True)

    def __call__(self, event1, event2):
        self.mask |= self.inside(event1, event2)
        xy = np.column_stack([self.x[self.mask], self.y[self.mask]])
        self._highlight.set_offsets(xy)
        self.canvas.draw()

    def inside(self, event1, event2):
        """Returns a boolean mask of the points inside the rectangle defined by
        event1 and event2."""
        # Note: Could use points_inside_poly, as well
        x0, x1 = sorted([event1.xdata, event2.xdata])
        y0, y1 = sorted([event1.ydata, event2.ydata])
        mask = ((self.x > x0) & (self.x < x1) &
                (self.y > y0) & (self.y < y1))
        return mask

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

procStatsJsonPath = r'C:\Users\HOEKHJ\Dev\InterferometryPython\export\PROC_20221202113132\PROC_20221202113132_statistics.json'

print(os.path.join(os.path.dirname(procStatsJsonPath), f"powerLawFitting.csv"))


csvPathAppend = r''
flipData = True
# analyzeImages = np.concatenate((np.arange(140, 160, 2), np.arange(160, 500, 10), np.arange(500, 914, 70)))

analyzeTimes = np.linspace(0, 57604, 12)


# analyzeImages = np.array([60, 155, 160, 165, 180, 195, 210, 330, 630, 870, 1556])
# actual times: 60, 296, 596, 896, 1796, 2696, 3596, 10796, 28796, 43196, 84356

haloLength = 2000


with open(procStatsJsonPath, 'r') as f:
    procStats = json.load(f)

data = {}
data['jsonPath'] = procStatsJsonPath
data['processDatetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data['csvPathAppend'] = csvPathAppend
data['data'] = {}

deltaTime = procStats["deltatime"]
timeFromStart = np.cumsum(deltaTime)
print(timeFromStart)

analyzeImages = np.array([find_nearest(timeFromStart, t)[1] for t in analyzeTimes])
print(f"{analyzeTimes=}")
print(f"{analyzeImages=}")
data['analyzeImages'] = ','.join(analyzeImages.astype(str))


def func(x, a, b):
    return a * x ** b

# try:
for idx, imageNumber in enumerate(analyzeImages):
    print(f'Analyzing image {idx}/{len(analyzeImages)}.')
    originalPath = procStats["analysis"][str(imageNumber)]["wrappedPath"]
    dataPath = os.path.join(os.path.dirname(originalPath), csvPathAppend, os.path.basename(originalPath))
    print(dataPath)

    conversionZ = procStats["conversionFactorZ"]
    y = np.loadtxt(dataPath, delimiter=",") * conversionZ  # y is in um
    if flipData:
        y = -y + max(y)

    conversionXY = procStats["conversionFactorXY"]
    x = np.arange(0, len(y)) * conversionXY  # x is now in um

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    highlighter = Highlighter(ax, x, y)
    plt.show()
    selected_regions = highlighter.mask
    xrange1, yrange1 = x[selected_regions], y[selected_regions]

    # TODO do fitting here
    popt, pcov = curve_fit(func, xrange1, yrange1)
    a = popt[0]
    b = popt[1]

    print(f"{a=}")
    print(f"{b=}")

    x_plot = np.arange(x[np.where(selected_regions)[0][0]], x[np.where(selected_regions)[0][0]]+haloLength)  # get the first nonFalse, i.e. first x of fit, than add haloLength datapoints. So all graphs will have same halo length in the end
    y_fit = func(x_plot, *popt)


    # TODO do plotting here
    fig, ax = plt.subplots()
    ax.scatter(x, y, label=f'Raw data {os.path.basename(originalPath)}')
    # ax.scatter(xrange1, yrange1, color='green', label='Selected data line 1')
    ax.plot(x_plot, y_fit, color='red', linewidth=3, label='Linear fit 1')
    # ax.set_title(f"{angleDeg=}")
    ax.set_xlabel("[um]")
    ax.set_ylabel("[um]")
    # ax.set_xlim([x[0], x[-1]])
    # ax.set_ylim([y[0], y[-1]])


    fig.savefig(os.path.join(os.path.dirname(originalPath),
                     f"PowerLawFitting_{os.path.splitext(os.path.basename(originalPath))[0]}.png"), dpi=300)
    print(os.path.join(os.path.dirname(originalPath), f"PowerLawFitting_{os.path.splitext(os.path.basename(originalPath))[0]}.png"))

    data['data'][idx] = {}
    data['data'][idx]['timeFromStart'] = timeFromStart[imageNumber]
    data['data'][idx]['xrange'] = xrange1.tolist()
    data['data'][idx]['yrange'] = yrange1.tolist()
    data['data'][idx]['a'] = a
    data['data'][idx]['b'] = b
    data['data'][idx]['xnew'] = x_plot.tolist()
    data['data'][idx]['ynew'] = y_fit.tolist()
    data['data'][idx]['x_full'] = x.tolist()
    data['data'][idx]['y_full'] = y.tolist()

    # plt.show()
    plt.close('all')
# except Exception as e:
#     print("Something went wrong, still saving data.")
#     print(e)

with open(os.path.join(os.path.dirname(procStatsJsonPath), f"powerlawFittingData.json"), 'w') as f:
    json.dump(data, f, indent=4)

