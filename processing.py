import os
import sys
import argparse
import cv2
import openface
import numpy as np
import scipy.spatial
import alignment

fileDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=fileDir, help='video directory')
parser.add_argument('--num', type=int, default=1000, help='num of images')
parser.add_argument('--dim', type=int, default=96, help='alignment dimension')
args = parser.parse_args()

print(args)

videoCapture = cv2.VideoCapture(os.path.join(args.dir, '000.mp4'))

if not videoCapture.isOpened(): sys.exit('video not opened')

template = np.load(os.path.join(fileDir, 'template.npy'))
delaunay = scipy.spatial.Delaunay(template)

facePredictor = os.path.join(fileDir, 'openface', 'models', 'dlib', 'shape_predictor_68_face_landmarks.dat')
alignDlib = openface.AlignDlib(facePredictor)
alignment = alignment.Alignment(args.dim, template, delaunay.simplices)

print('processing images...')

for index in range(args.num):

    ret, rawImage = videoCapture.read()

    if not ret: break
    
    boundingBox = alignDlib.getLargestFaceBoundingBox(rawImage)
    landmarks = alignDlib.findLandmarks(rawImage, boundingBox)

    alignedImage = alignment.align(rawImage, landmarks)

    convertedImage = cv2.cvtColor(alignedImage, cv2.COLOR_RGB2GRAY)

    equalizedImage = cv2.equalizeHist(convertedImage)

    markedImage = rawImage.copy()

    for triangle in delaunay.simplices:

        cv2.line(markedImage, landmarks[triangle[0]], landmarks[triangle[1]], (255, 0, 255))
        cv2.line(markedImage, landmarks[triangle[1]], landmarks[triangle[2]], (255, 0, 255))
        cv2.line(markedImage, landmarks[triangle[2]], landmarks[triangle[0]], (255, 0, 255))

    cv2.imwrite(os.path.join(args.dir, '..', 'raw images', str(index).zfill(3) + '.jpg'), 
        rawImage[landmarks[30][1] - 100:landmarks[30][1] + 100, landmarks[30][0] - 100:landmarks[30][0] + 100])
    cv2.imwrite(os.path.join(args.dir, '..', 'marked images', str(index).zfill(3) + '.jpg'), 
        markedImage[landmarks[30][1] - 100:landmarks[30][1] + 100, landmarks[30][0] - 100:landmarks[30][0] + 100])
    cv2.imwrite(os.path.join(args.dir, '..', 'aligned images', str(index).zfill(3) + '.jpg'), equalizedImage)

videoCapture.release()