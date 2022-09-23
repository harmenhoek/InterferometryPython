import numpy as np
from matplotlib import pyplot as plt
from configparser import ConfigParser
from plotting import plot_imunwrapped



filename = r"C:\Users\HOEKHJ\Dev\InterferometryPython\export\PROC_20220923120429\Basler_a2A5328-15ucBAS__40087133__20220914_150439202_0006_small_19_4_False,True_unwrapped.npy"
configFilename = r"C:\Users\HOEKHJ\Dev\InterferometryPython\export\PROC_20220923120429\config_PROC_20220923120429.ini"


config = ConfigParser()
config.read("config.ini")

im_unwrapped = np.load(filename, mmap_mode='r')
fig = plot_imunwrapped(im_unwrapped, config)

plt.show()