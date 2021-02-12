import numpy as np

def calc_A(left_pts, right_pts):
    A = []
    for i in range(left_pts.shape[0]):
        xl = left_pts[i, 0]
        yl = left_pts[i, 1]
        xr = right_pts[i, 0]
        yr = right_pts[i, 1]
        A_i = np.array([xl*xr, xl*yr, xl, yl*xr, yl*yr, yl, xr, yr, 1])
        A.append(A_i)
    
    return np.array(A)

def normalize_points(points):
    m = points.mean(axis=1)
    average_dist = np.sqrt(np.power(points[0] - m[0], 2) + np.power(points[1] - m[1], 2))
    s = np.sqrt(2) / average_dist.mean()

    T = np.array([[s, 0, -s * m[0]], 
                  [0, s, -s * m[1]],
                  [0, 0,        1]])

    #points = np.dot(T, points)
    return T


def calc_F(left_pts, right_pts):
    left_pts, T = normalize_points(left_pts)
    right_pts, T_prime = normalize_points(right_pts)
    A = calc_A(left_pts, right_pts)
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    F = np.dot(T_prime, np.dot(F, T)) 
    return F / F[2, 2]

def calc_skew(x):
    return np.array([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ])

def calc_epipole(F):
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]

def calc_P(F):
    e = calc_epipole(F.T)
    T = calc_skew(e)
    return np.vstack((np.dot(T, F.T).T, e)).T


def triangulate(left_pts, right_pts, cam1, cam2):
    n = left_pts.shape[0]
    ret_mat = np.zeros((4, n))

    for i in range(n):
        A = np.array([
            left_pts[i, 0] * cam1[2, :] - cam1[0, :],
            left_pts[i, 1] * cam1[2, :] - cam1[1, :],
            right_pts[i, 0] * cam2[2, :] - cam2[0, :],
            right_pts[i, 1] * cam2[2, :] - cam2[1, :]
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        ret_mat[:, i] = X / X[3]

    return ret_mat