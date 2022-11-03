import numpy as np
from matplotlib import pyplot as plt
import json
import os
import csv
from matplotlib.widgets import RectangleSelector


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


procStatsJsonPath = r'C:\Users\HOEKHJ\Dev\InterferometryPython\export\PROC_20221031162601\PROC_20221031162601_statistics.json'
csvPathAppend = r'csv'
flipData = True

analyzeImages = [500]

with open(procStatsJsonPath, 'r') as f:
    procStats = json.load(f)

for idx in analyzeImages:
    originalPath = procStats["analysis"][str(idx)]["wrappedPath"]
    dataPath = os.path.join(os.path.dirname(originalPath), csvPathAppend, os.path.basename(originalPath))
    print(dataPath)

    y = np.loadtxt(dataPath, delimiter=",")
    if flipData:
        y = -y + max(y)
    x = np.arange(0, len(y))
    print(len(x), len(y))
    print(type(x), type(y))

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    highlighter = Highlighter(ax, x, y)
    plt.show()
    selected_regions = highlighter.mask
    xrange1, yrange1 = x[selected_regions], y[selected_regions]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    highlighter = Highlighter(ax, x, y)
    plt.show()
    selected_regions = highlighter.mask
    xrange2, yrange2 = x[selected_regions], y[selected_regions]

    print(xrange1, yrange1)
    print(xrange2, yrange2)

    coef1 = np.polyfit(xrange1, yrange1, 1)
    poly1d_fn1 = np.poly1d(coef1)

    coef2 = np.polyfit(xrange2, yrange2, 1)
    poly1d_fn2 = np.poly1d(coef2)

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x, poly1d_fn1(x))
    ax.plot(x, poly1d_fn2(x))

    plt.show()








