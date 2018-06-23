import os
import sys
import math
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

fileDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=fileDir, help='image directory')
parser.add_argument('--num', type=int, default=1000, help='num of images')
args = parser.parse_args()

print(args)

scale = lambda inVal, inMin, inMax, outMin, outMax: outMin + (inVal - inMin) / (inMax - inMin) * (outMax - outMin)

images = [ cv2.imread(os.path.join(args.dir, str(index).zfill(3) + '.jpg'), cv2.IMREAD_GRAYSCALE).astype(np.float).reshape(-1) for index in range(args.num) ]

masks = [ np.load(os.path.join(fileDir, 'masks', str(index).zfill(3) + '.npy')) for index in range(111) ]

trianglesList = np.array([ [ image[mask == 1] for image in images ] for mask in masks ])
        
clusters = {}
    
for line in open(os.path.join(fileDir, 'clusters.txt')).readlines():

    first, last, label = line[:-1].split()

    if not label in clusters:

        clusters[label] = []

    clusters[label] += range(int(first), int(last))

diffs = np.array([ sum([ sum([ 
    np.linalg.norm(triangles[cluster].mean(0) - triangles.mean(0)) ** 2 / len(clusters) - 
        np.linalg.norm(triangle - triangles[cluster].mean(0)) ** 2 / len(cluster) 
            for triangle in triangles[cluster] ]) for cluster in clusters.values() ]) for triangles in trianglesList ])

weights = scale(diffs, diffs.min(), diffs.max(), 0, 1)

np.save(os.path.join(fileDir, 'weights.npy'), weights)