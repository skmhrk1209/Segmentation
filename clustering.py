import os
import sys
import argparse
import cv2
import openface
import numpy as np
import sklearn.decomposition
import sklearn.metrics.pairwise
import sklearn.cluster
import shapely.geometry
import scipy.spatial
import segmentation

fileDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=fileDir, help='image directory')
parser.add_argument('--num', type=int, default=1000, help='num of images')
parser.add_argument('--dim', type=int, default=96, help='dim of images')
parser.add_argument('--cls', type=int, default=10, help='num of clusters')
parser.add_argument('--tri', action='store_true', help='apply to triangle')
parser.add_argument('--pca', action='store_true', help='with pca')
parser.add_argument('--var', type=float, default=0.9, help='pca explained variance')
parser.add_argument('--mtd', type=str, default='spectral', choices=['spectral', 'agglomerative'], help='clustering method')
parser.add_argument('--seg', action='store_true', help='with segmentation')
parser.add_argument('--lam', type=float, default=0.2, help='lambda for segmentation')
args = parser.parse_args()

print(args)

images = [ cv2.imread(os.path.join(args.dir, str(index).zfill(3) + '.jpg'), cv2.IMREAD_GRAYSCALE).reshape(-1) for index in xrange(args.num) ]

'''
template = openface.align_dlib.MINMAX_TEMPLATE
delaunay = scipy.spatial.Delaunay(template)

masks = [ np.array([ 1 if shapely.geometry.Point(x, y).intersects(shapely.geometry.Polygon(args.dim * template[triangle])) else 0

    for x in xrange(args.dim) for y in xrange(args.dim) ]) for triangle in delaunay.simplices ]
'''

masks = [ np.load(os.path.join(fileDir, 'masks', str(index).zfill(3) + '.npy')) for index in xrange(111) ]

imagesList = [ sklearn.decomposition.PCA(n_components=args.var).fit_transform(images) if args.pca else images 
    for images in ([ [ image[mask == 1] for image in images ] for mask in masks ] if args.tri else [ images ]) ]

for index, images in enumerate(imagesList):

    if args.seg:

        indices = segmentation.Segmentation().segment(images, args.lam)

        indices.insert(0, 0)
        indices.append(args.num)

        images = [ images[first:last].mean(0) for (first, last) in zip(indices[0:], indices[1:]) ]

    scale = lambda inVal, inMin, inMax, outMin, outMax: outMin + (inVal - inMin) / (inMax - inMin) * (outMax - outMin)

    adjacencyMatrix = scale(sklearn.metrics.pairwise.cosine_similarity(images), -1, 1, 0, 1)

    clusters = sklearn.cluster.SpectralClustering(n_clusters=args.cls, affinity='precomputed').fit_predict(adjacencyMatrix) if args.mtd == 'spectral' else \
        sklearn.cluster.AgglomerativeClustering(n_clusters=args.cls, affinity='cosine', linkage='average').fit_predict(images)

    adjacencyMatrixImage = cv2.cvtColor(cv2.resize((adjacencyMatrix * 255).astype(np.uint8), (adjacencyMatrix.shape[0] * 10, adjacencyMatrix.shape[1] * 10)), cv2.COLOR_GRAY2RGB)

    cv2.imwrite(os.path.join(args.dir, '..', 'adjacency matrices', str(index).zfill(3) + '.jpg'), adjacencyMatrixImage)