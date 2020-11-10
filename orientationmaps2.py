##########
img = cv.imread('TestImg.jpg')
laplacian = cv.Laplacian(img,cv.CV_64F)

"""first set of four directions"""
sobelx1 = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely1 = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
sobelx2 = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely2 = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx1,cmap = 'gray')
plt.title('Sobel X1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely1,cmap = 'gray')
plt.title('Sobel Y1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx2,cmap = 'gray')
plt.title('Sobel X2'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely2,cmap = 'gray')
plt.title('Sobel Y2'), plt.xticks([]), plt.yticks([])


"""second set of four directions"""
sobelx3 = cv.Sobel(img,cv.CV_64F,1,1,ksize=5, scale=math.sqrt(2)/2)
#sobely3 = cv.Sobel(img,cv.CV_64F,-1,-1,ksize=5, scale=math.sqrt(2)/2)
#sobelx4 = cv.Sobel(img,cv.CV_64F,1,-1,ksize=5, scale=math.sqrt(2)/2)
#sobely4 = cv.Sobel(img,cv.CV_64F,-1,1,ksize=5, scale=math.sqrt(2)/2)



plt.subplot(2,2,4),plt.imshow(sobelx3,cmap = 'gray')
plt.title('Sobel X3'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobely3,cmap = 'gray')
#plt.title('Sobel Y3'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobelx4,cmap = 'gray')
#plt.title('Sobel X4'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobely4,cmap = 'gray')
#plt.title('Sobel Y4'), plt.xticks([]), plt.yticks([])
plt.show()
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
