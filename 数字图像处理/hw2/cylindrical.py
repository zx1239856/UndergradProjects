import cv2
import math
import numpy as np

class Trans:
    def __init__(self, src_shape, flip=False):
        self.W = src_shape[1]
        self.H = src_shape[0]
        self.flip = flip
    def __call__(self, x, y):
        A = self.W / math.pi
        if not self.flip:
            x0 = A * (math.pi - math.acos(x / A - 1))
            return (x0, y)
        else:
            y0 = A * (math.pi - math.acos(y / A - 1))
            return (x, y0)

class Conformal:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x, y):
        x -= self.shape[1] / 2.
        y -= self.shape[0] / 2.
        if abs(x) <= 1 and abs(y) <= 1:
            return (x + self.shape[1] / 2., y + self.shape[0] / 2.)
        else:
            return ((x * x - y * y + 1) / max(abs(x), abs(y)) + self.shape[1] / 2., 2 * x * y / max(abs(x), abs(y)) + self.shape[0] / 2.)

def get_color(src, trans, x, y):
    def _in_range(_x, _y):
        return _x >= 0 and _x < src.shape[1] and _y >= 0 and _y < src.shape[0]

    x, y = trans(x, y)
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
    src = cv2.imread("final_720p.png")

    # trans = Trans(src.shape)
    # height = src.shape[0]
    # width = int(math.floor(src.shape[1] * 2. / math.pi))
    # res = np.zeros((height, width, 3), np.uint8)
    # for j in range(height):
    #     for i in range(width):
    #         res[j, i] = get_color(src, trans, i, j)
    # cv2.imshow('Result', res)
    # cv2.imwrite('final_720p_res.png', res)

    src_ = src
    # src = res
    # trans = Trans(src.shape, True)
    # width = src.shape[1]
    # height = int(math.floor(src.shape[0] * 2. / math.pi))
    # res2 = np.zeros((height, width, 3), np.uint8)
    # for j in range(height):
    #     for i in range(width):
    #         res2[j, i] = get_color(src, trans, i, j)
    # cv2.imshow('Result2', res2)
    # cv2.imwrite('final_720p_res2.png', res2)

    res3 = np.zeros(src_.shape, np.uint8)
    trans = Conformal(src_.shape)
    for j in range(src_.shape[0]):
        for i in range(src_.shape[1]):
            res3[j, i] = get_color(src_, trans, i, j)
    cv2.imshow('Conformal', res3)
    cv2.waitKey()
