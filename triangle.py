import os
import sys
import math
import argparse
import cv2
import numpy as np
import sklearn.decomposition
import sklearn.metrics.pairwise
import sklearn.cluster
import shapely.geometry
import scipy.spatial
import scipy.optimize
import segmentation

fileDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=fileDir, help='image directory')
parser.add_argument('--num', type=int, default=1000, help='num of images')
parser.add_argument('--pca', action='store_true', help='with pca')
parser.add_argument('--var', type=float, default=0.9, help='cumulative contribution rate')
parser.add_argument('--seg', action='store_true', help='with segmentation')
parser.add_argument('--lam', type=float, default=0.2, help='lambda for segmentation')
parser.add_argument('--cls', type=int, default=6, help='num of clusters')
parser.add_argument('--dst', type=str, default='cosine', choices=['cos', 'euc'], help='distance for clustering')
parser.add_argument('--nam', type=str, help='name of file or directory')
args = parser.parse_args()

print(args)

scale = lambda inVal, inMin, inMax, outMin, outMax: outMin + (inVal - inMin) / (inMax - inMin) * (outMax - outMin)

images = [ cv2.imread(os.path.join(args.dir, str(index).zfill(3) + '.jpg'), cv2.IMREAD_GRAYSCALE).astype(np.float).reshape(-1) for index in range(args.num) ]

if args.seg:

    changingIndices = segmentation.Segmentation().segment(sklearn.decomposition.PCA(args.var).fit_transform(images), args.lam)

    changingIndices.insert(0, 0)
    changingIndices.append(args.num)

    images = [ np.mean(images[first:last], 0) for first, last in zip(changingIndices[0:], changingIndices[1:]) for _ in range(first, last) ]

masks = [ np.load(os.path.join(fileDir, 'masks', str(index).zfill(3) + '.npy')) for index in range(111) ]

trianglesList = [ [ image[mask == 1] for image in images ] for mask in masks ]

for index, triangles in enumerate(trianglesList):

    if args.dst == 'cos':

        affinityMatrix = sklearn.metrics.pairwise.cosine_distances(triangles)

    elif args.dst == 'euc':

        affinityMatrix = sklearn.metrics.pairwise.euclidean_distances(triangles)

    affinityMatrix = scale(affinityMatrix, affinityMatrix.min(), affinityMatrix.max(), 1, 0)

    clusterIndices = sklearn.cluster.SpectralClustering(n_clusters=args.cls, affinity='precomputed').fit_predict(affinityMatrix)

    affinityMatrixImage = cv2.cvtColor((affinityMatrix * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    colors = [ (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255) ]

    i = 0

    for j in range(args.num):

        if j + 1 == args.num or clusterIndices[j] != clusterIndices[j + 1]:

            cv2.rectangle(affinityMatrixImage, (i, i), (j - 1, j - 1), colors[clusterIndices[j]], 2)

            i = j

    cv2.imwrite(os.path.join(args.dir, '..', args.nam, str(index).zfill(3) + '.jpg'), affinityMatrixImage)