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
parser.add_argument('--dst', type=str, default='cosine', choices=['cos', 'euc', 'opt'], help='distance for clustering')
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

if args.dst == 'opt':

    masks = [ np.load(os.path.join(fileDir, 'masks', str(index).zfill(3) + '.npy')) for index in range(111) ]

    trianglesList = np.array([ [ image[mask == 1] for image in images ] for mask in masks ])
        
    weights = np.load(os.path.join(fileDir, 'weights.npy'))

    distance = lambda image1, image2: math.sqrt(sum([ np.linalg.norm(triangle1 - triangle2) ** 2 * weight for triangle1, triangle2, weight in zip(image1, image2, weights) ]))

    affinityMatrix = np.array([ [ distance(image1, image2) for image2 in trianglesList.T ] for image1 in trianglesList.T ])

elif args.dst == 'cos':

    affinityMatrix = sklearn.metrics.pairwise.cosine_distances(images)

elif args.dst == 'euc':

    affinityMatrix = sklearn.metrics.pairwise.euclidean_distances(images)

affinityMatrix = scale(affinityMatrix, affinityMatrix.min(), affinityMatrix.max(), 1, 0)

clusterIndices1 = sklearn.cluster.SpectralClustering(n_clusters=args.cls, affinity='precomputed').fit_predict(affinityMatrix)

clusterIndices2 = []
    
for line in open(os.path.join(fileDir, 'clusters.txt')).readlines():

    first, last, label = line[:-1].split()

    for i in range(int(first), int(last)):

        clusterIndices2.append(int(label))

affinityMatrixImage1 = cv2.cvtColor((affinityMatrix * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
affinityMatrixImage2 = cv2.cvtColor((affinityMatrix * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

colors = [ (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255) ]

i1 = 0
i2 = 0

for j in range(args.num):

    if j + 1 == args.num or clusterIndices1[j] != clusterIndices1[j + 1]:

        cv2.rectangle(affinityMatrixImage1, (i1, i1), (j - 1, j - 1), colors[clusterIndices1[j]], 2)

        i1 = j

    if j + 1 == args.num or clusterIndices2[j] != clusterIndices2[j + 1]:

        cv2.rectangle(affinityMatrixImage2, (i2, i2), (j - 1, j - 1), colors[clusterIndices2[j]], 2)

        i2 = j

cv2.imwrite(os.path.join(args.dir, '..', args.nam, 'auto.jpg'), affinityMatrixImage1)
cv2.imwrite(os.path.join(args.dir, '..', args.nam, 'manual.jpg'), affinityMatrixImage2)