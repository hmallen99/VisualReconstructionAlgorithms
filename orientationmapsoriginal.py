import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt

##########
"""
img = cv.imread('dave.jpg',0)
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
"""
let src = cv.imread('dave.jpg');
let dstx = new cv.Mat();
let dsty = new cv.Mat();
cv.cvtColor(src, src, cv.COLOR_RGB2GRAY, 0);
// You can try more different parameters
cv.Sobel(src, dstx, cv.CV_8U, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
cv.Sobel(src, dsty, cv.CV_8U, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);
// cv.Scharr(src, dstx, cv.CV_8U, 1, 0, 1, 0, cv.BORDER_DEFAULT);
// cv.Scharr(src, dsty, cv.CV_8U, 0, 1, 1, 0, cv.BORDER_DEFAULT);
cv.imshow('canvasOutputx', dstx);
cv.imshow('canvasOutputy', dsty);
src.delete(); dstx.delete(); dsty.delete();

##########
"""
f(x, y) = z where z is pixel intensity
grad(z) = <del z/ del x, del z, del y> direction of steepest ascent, directional derivatives dot u where u = <x, y, z> and |u| = 1
<1, 0>
<0, 1>
<-1, 0>
<0, -1>
<sqrt(2)/2, sqrt(2)/2>
<-sqrt(2)/2, -sqrt(2)/2>
<sqrt(2)/2, -sqrt(2)/2>
<-sqrt(2)/2, sqrt(2)/2>
"""


############



############




##############
