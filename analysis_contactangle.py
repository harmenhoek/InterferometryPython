import math

import numpy as np
from matplotlib import pyplot as plt
import json
import os
import csv
from matplotlib.widgets import RectangleSelector
from datetime import datetime


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


procStatsJsonPath = '/Users/harmenhoek/Dev/InterferometryPython/export/PROC_20221104130005/PROC_20221104130005_statistics.json'
csvPathAppend = r''
flipData = True

# 1 slice: Contact angle = -1.6494950309356011 degrees.
# 11 slices: -1.650786783947852 degrees.

analyzeImages = np.array([0])


with open(procStatsJsonPath, 'r') as f:
    procStats = json.load(f)


'''
20 periods = 226 pix = 134um
1pi = lambda / 4n = 532 / (4*1.434) = 92.74 um
20 periods = 20*2pi = 40pi = 40*92.74 = 3709.6nm
1.53

237pix = 237/1687 = 0.1405 mm



'''


data = {}
data['jsonPath'] = procStatsJsonPath
data['processDatetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data['analyzeImages'] = ','.join(analyzeImages.astype(str))
data['csvPathAppend'] = csvPathAppend
data['data'] = {}


timeFromStart = procStats["deltatime"]


angleDegAll = np.zeros_like(timeFromStart, dtype='float')

try:
    for idx, imageNumber in enumerate(analyzeImages):
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
        # plt.draw()
        # plt.waitforbuttonpress(0)  # this will wait for indefinite time
        # plt.close(fig)
        selected_regions = highlighter.mask
        xrange1, yrange1 = x[selected_regions], y[selected_regions]

        # fig, ax = plt.subplots()
        # ax.scatter(x, y)
        # highlighter = Highlighter(ax, x, y)
        # plt.show()
        # selected_regions = highlighter.mask
        # xrange2, yrange2 = x[selected_regions], y[selected_regions]

        # print(xrange1, yrange1)
        # print(xrange2, yrange2)

        coef1 = np.polyfit(xrange1, yrange1, 1)
        poly1d_fn1 = np.poly1d(coef1)

        # coef2 = np.polyfit(xrange2, yrange2, 1)
        # poly1d_fn2 = np.poly1d(coef2)

        # print(coef1, coef2)

        a_horizontal = 0

        # angleRad = math.atan((coef1[0]-coef2[0])/(1+coef1[0]*coef2[0]))
        angleRad = math.atan((coef1[0]-a_horizontal)/(1+coef1[0]*a_horizontal))
        angleDeg = math.degrees(angleRad)

        # print(f"{angleRad=}")
        # print(f"{angleDeg=}")

        fig, ax = plt.subplots()
        ax.scatter(x, y, label=f'Raw data {os.path.basename(originalPath)}')
        ax.scatter(xrange1, yrange1, color='green', label='Selected data line 1')
        # ax.scatter(xrange2, yrange2, color='green', label='Selected data line 2')
        ax.plot(x, poly1d_fn1(x), color='red', linewidth=3, label='Linear fit 1')
        # ax.plot(x, poly1d_fn2(x), color='red', linewidth=3, label='Linear fit 2')
        ax.set_title(f"{angleDeg=}")
        ax.set_xlabel("[um]")
        ax.set_ylabel("[um]")
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([y[0], y[-1]])



        fig.savefig(os.path.join(os.path.dirname(originalPath), f"angleFitting_{os.path.splitext(os.path.basename(originalPath))[0]}.png"), dpi=300)

        data['data'][idx] = {}
        data['data'][idx]['timeFromStart'] = timeFromStart[idx]
        data['data'][idx]['xrange1'] = xrange1.tolist()
        data['data'][idx]['yrange1'] = yrange1.tolist()
        # data['data'][idx]['xrange2'] = xrange2
        # data['data'][idx]['yrange2'] = yrange2
        data['data'][idx]['coef1'] = coef1.tolist()
        # data['data'][idx]['coef2'] = coef2
        data['data'][idx]['angleDeg'] = angleDeg
        data['data'][idx]['angleRad'] = angleRad

        angleDegAll[idx] = angleDeg
        print(f'Contact angle = {angleDeg} degrees.')


        # plt.show()
        plt.close('all')
except:
    print("Something went wrong, still saving data.")

    with open(os.path.join(os.path.dirname(procStatsJsonPath), f"angleFittingData.csv"), 'w') as f:
        json.dump(data, f, indent=4)







