import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
import glob
import os
from natsort import os_sorted


images = glob.glob('export/wrapped/*.png')
# images = glob.glob('export/unwrapped/*.png')
images = os_sorted(images)

fig = plt.figure(figsize=(15., 15.))
plt.axis('off')

grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(12, 12),  # creates 12x12 grid of axes
                 axes_pad=0,  # pad between axes in inch.
                 )

for ax, path in zip(grid, images):
    # Iterating over the grid returns the Axes.
    img = cv2.imread(path)
    img = img[176:176+1100, 250:250+1400]

    ax.imshow(img)
    ax.axis('off')

plt.axis('off')
# plt.show()

fig.savefig('grid_wrapped.png', dpi=500, bbox_inches='tight', pad_inches=0)
fig.savefig('grid_wrapped.pdf', dpi=500, bbox_inches='tight', pad_inches=0)
# fig.savefig('grid_unwrapped.png', dpi=500, bbox_inches='tight', pad_inches=0)
# fig.savefig('grid_unwrapped.pdf', dpi=500, bbox_inches='tight', pad_inches=0)

plt.close('all')
# 250, 176
# 1460, 1100