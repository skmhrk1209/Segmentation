import sys
import cv2
import sklearn.decomposition

image = cv2.cvtColor(cv2.imread('image.jpg'), cv2.COLOR_RGB2GRAY)
pca = sklearn.decomposition.PCA(float(sys.argv[1]))
reduced = pca.fit_transform(image)
restored = pca.inverse_transform(reduced)
cv2.imwrite('restored.jpg', restored)