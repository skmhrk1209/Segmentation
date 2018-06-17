import math
import numpy as np

class Segmentation:

    def segment(self, images, lam):

        images = np.asarray(images)

        indices, covariances = [], {}

        num, dim = images.shape[0:2]

        def BIC(first, middle, last):

            def covariance(first, last):

                if not (first, last) in covariances:

                    covariances[(first, last)] = np.cov(images[first:last].T)

                return covariances[(first, last)]

            def regularize(covariance):

                return np.diag(np.diag(covariance))
            
            return (last - first) * math.log(np.linalg.det(regularize(covariance(first, last)))) \
                - (middle - first) * math.log(np.linalg.det(regularize(covariance(first, middle)))) \
                - (last - middle) * math.log(np.linalg.det(regularize(covariance(middle, last)))) \
                - lam * (dim + dim * (dim + 1) / 2) * math.log(last - first)

        first, last = 0, 1

        while last <= num:

            maxBic, maxIndex = 0, 0

            for middle in xrange(first, last):

                if (middle - first) < 2: continue

                if (last - middle) < 2: continue

                bic = BIC(first, middle, last)

                if bic > maxBic:

                    maxBic = bic
                    maxIndex = middle

            if maxBic > 0:

                indices.append(maxIndex)
                first, last = maxIndex, maxIndex + 1

            else:

                last += 1

        return indices
