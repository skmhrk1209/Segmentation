# coding: UTF-8

import cv2
import numpy as np
import shapely.geometry

class Alignment:

    def __init__(self, dim, template, triangles):

        template = np.asarray(template)

        self.dim = dim

        self.template = template

        self.triangles = triangles

        self.masks = [ np.array([ [ [1, 1, 1]

            if shapely.geometry.Point(x, y).intersects(shapely.geometry.Polygon(dim * template[triangle])) else [0, 0, 0]

                for x in xrange(dim) ] for y in xrange(dim) ]) for triangle in triangles ]

    def align(self, image, landmarks):

        landmarks = np.asarray(landmarks)

        alignedImage = np.zeros((self.dim, self.dim, 3), image.dtype)

        for triangle, mask in zip(self.triangles, self.masks):

            minPosition = landmarks[triangle].min(0)
            maxPosition = landmarks[triangle].max(0)

            affineTransform = cv2.getAffineTransform(np.apply_along_axis(
                lambda position: position - minPosition, 1, landmarks[triangle])
                    .astype(self.template.dtype), self.dim * self.template[triangle])
            
            warpedImage = cv2.warpAffine(image[
                minPosition[1]:maxPosition[1]+1, 
                minPosition[0]:maxPosition[0]+1], 
                affineTransform, (self.dim, self.dim))

            alignedImage += warpedImage * mask
        
        return alignedImage