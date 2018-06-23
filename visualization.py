import os
import sys
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

fileDir = os.path.dirname(os.path.realpath(__file__))

scale = lambda inVal, inMin, inMax, outMin, outMax: outMin + (inVal - inMin) / (inMax - inMin) * (outMax - outMin)

template = np.load(os.path.join(fileDir, 'template.npy'))
delaunay = scipy.spatial.Delaunay(template)

weights = np.load('weights.npy')

template[:, 1] = scale(template[:, 1], 0, 1, 1, 0)

figure = plt.figure()
axes = figure.add_subplot(1, 1, 1)

for index, (weight, triangle) in enumerate(sorted(zip(weights, delaunay.simplices))):

    axes.add_patch(plt.Polygon(template[triangle], fill=True if index in [ int(argv) for argv in sys.argv[1:] ] else False))

plt.savefig(os.path.join(fileDir, 'landmarks', ' '.join(sys.argv[1:]) + '.png'))