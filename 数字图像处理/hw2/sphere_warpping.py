import cv2
import math
import numpy as np

class Trans:
    def __init__(self, d0, r0, src_shape):
        self.d0 = d0
        self.r0 = r0
        self.shape = src_shape
    def __call__(self, theta, phi):
        d = 2. / math.pi * self.d0 * phi
        r = d * math.sin(theta)
        c = d * math.cos(theta)
        return (c + self.shape[1] / 2., r + self.shape[0] / 2.)

def get_color(src, trans, x, y):
    def _in_range(_x, _y):
        return _x >= 0 and _x < src.shape[1] and _y >= 0 and _y < src.shape[0]

    res = trans(x, y)
    x = res[0]
    y = res[1]
    x = min(max(x, 0), src.shape[1] - 1)
    y = min(max(y, 0), src.shape[0] - 1)
    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    y1 = int(np.floor(y))
    y2 = int(np.ceil(y))
    q11 = src[y1, x1] if _in_range(x1, y1) else np.array([0,0,0], dtype=np.uint8)
    q21 = src[y1, x2] if _in_range(x2, y1) else np.array([0,0,0], dtype=np.uint8)
    q12 = src[y2, x1] if _in_range(x1, y2) else np.array([0,0,0], dtype=np.uint8)
    q22 = src[y2, x2] if _in_range(x2, y2) else np.array([0,0,0], dtype=np.uint8)

    if x1 != x2:
        y_1 = ((x2 - x) * q11 + (x - x1) * q21)/ (x2 - x1)
        y_2 = ((x2 - x) * q12 + (x - x1) * q22) / (x2 - x1)
    else:
        y_1 = src[y1, x1] if _in_range(x1, y1) else np.array([0,0,0], dtype=np.uint8)
        y_2 = src[y2, x1] if _in_range(x1, y2) else np.array([0,0,0], dtype=np.uint8)
    if y1 != y2:
        a = ((y2 - y) * y_1 + (y - y1) * y_2) / (y2 - y1)
    else:
        a = y_1
    return np.round(a).astype(np.uint8)

if __name__ == "__main__":
    src = cv2.imread("sphere_warp_src.jpg")

    height = 500
    width = 500
    res = np.zeros((height, width, 3), np.uint8)
    r_0 = 1./2 * min(height, width)
    trans = Trans(1. / 2 * max(src.shape[0], src.shape[1]), 1. / 2 * min(height, width), src.shape)
    for j in range(height):
        for i in range(width):
            dis = math.sqrt((i - width / 2.)**2 + (j - height / 2.)**2)
            if dis <= r_0:
                phi = math.asin(dis / r_0)
                theta = math.atan2(j - height / 2., i - width / 2.)
                res[j, i] = get_color(src, trans, theta, phi)

    cv2.imshow('Result', res)
    cv2.imwrite('sphere_warp_res_2.png', res)
    cv2.waitKey()
