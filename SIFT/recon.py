import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
from scipy.misc import imread
from eight_point import triangulate, calc_skew, normalize_points

def get_sift(img):
    sift = cv.xfeatures2d.SIFT_create()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)

    ret_img = cv.drawKeypoints(gray, kp, img)

    cv.imwrite('sift_keypoints.jpg', img)
    return kp

def match_features(img1, img2):
    sift = cv.xfeatures2d.SIFT_create()

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(gray1, None)

    kp2, des2 = sift.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.85*n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 100.0)
    mask = mask.ravel()

    return src_pts[mask==1].T, dst_pts[mask==1].T

def make_homogeneous(X):
    if X.ndim == 1:
        return np.hstack([X, 1])
    return np.array(np.vstack([X, np.ones(X.shape[1])]))

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def plot_epipolar(img1, img2, pts1, pts2, F):
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()

def plot_matches(img1, img2, src, dst):
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img1)
    ax1.plot(src[0], src[1], 'r.')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img2)
    ax2.plot(dst[0], dst[1], 'r.')
    fig.show()

def calc_proj_mats(F):
    v, e = np.linalg.eig(F)
    v_prime, e_prime = np.linalg.eig(F.T)

    e = e[:, np.argmin(np.abs(v))]
    e = e / e[2]
    e_prime = e_prime[:, np.argmin(np.abs(v_prime))]
    e_prime = e_prime / e_prime[2]

    P = np.eye(3, 4)
    S = calc_skew(e_prime)
    SF = S @ F
    P_prime = np.hstack((SF, e_prime.reshape(3, 1)))
    P_prime = P_prime / P_prime[2, 2]
    return P, P_prime

def main():
    img1 = imread("images/viff.001.ppm")
    img2 = imread("images/viff.003.ppm")

    src, dst = match_features(img1, img2)

    src = make_homogeneous(src)
    dst = make_homogeneous(dst)

    pts1 = np.int32(src[:2].T)
    pts2 = np.int32(dst[:2].T)

    F, maskF = cv.findFundamentalMat(pts1, pts2)

    intrinsic = np.array([
        [2360, 0, img1.shape[1] / 2],
        [0, 2360, img1.shape[0] / 2],
        [0, 0, 1]
    ])

    pts1 = np.dot(np.linalg.inv(intrinsic), src)[:2].T
    pts2 = np.dot(np.linalg.inv(intrinsic), dst)[:2].T

    E, mask = cv.findEssentialMat(pts1, pts2)

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    U, S, V = np.linalg.svd(E)

    
    W = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])

    u3 = U[:, -1].reshape(3, 1)

    P_prime_lst = []
    P_prime_lst.append(np.hstack((U @ W @ V, u3)))
    P_prime_lst.append(np.hstack((U @ W @ V, -u3)))
    P_prime_lst.append(np.hstack((U @ W.T @ V, u3)))
    P_prime_lst.append(np.hstack((U @ W.T @ V, -u3)))

    #P, P_primeF = calc_proj_mats(F)
    #H1, H2, t = cv.stereoRectifyUncalibrated(pts1, pts2, F, img1.shape[:2])
    #img1_warp = cv.warpPerspective
    
    pts1n = np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))
    pts2n = np.vstack((pts2.T, np.ones((1, pts2.shape[0]))))

    P = np.eye(3, 4)
    P_prime = P_prime_lst[0]

    X = cv.triangulatePoints(P, P_prime, pts1.T.astype(float), pts2.T.astype(float))
    tri_pts = cv.convertPointsFromHomogeneous(X.T)
    tri_pts = tri_pts.reshape(-1, 3).T

    fig = plt.figure()
    fig.suptitle('3D reconstructed')
    ax = fig.gca(projection='3d')
    ax.plot(tri_pts[0], tri_pts[1], tri_pts[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.show()

if __name__ == "__main__":
    main()