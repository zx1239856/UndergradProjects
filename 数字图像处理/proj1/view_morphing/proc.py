from argparse import ArgumentParser
import cv2
import numpy as np
import scipy as sp
import os.path as osp
from itertools import product
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R


def rotation_matrix(u, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - np.cos(theta)
    x = u[0]
    y = u[1]
    return np.array([[t*x*x + c, t*x*y, s*y],
                     [t*x*y, t*y*y + c, -s*x],
                     [-s*y, s*x, c]])


def rotation_matrix_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def warpKeyPoints(points, H):
    points = np.array(points)  # [pnts, ndim]
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = points.dot(H.T)
    points /= points[:, [2]]
    return points[:, :2]


def get_prewarp(F):

    def get_e(mat):
        val, vec = np.linalg.eig(mat)
        return vec[:, np.argmin(np.abs(val))]

    e0 = get_e(F)
    e1 = get_e(F.T)
    d0 = np.array([-e0[1], e0[0], 0])
    Fd0 = F.dot(d0)
    d1 = np.array([-Fd0[1], Fd0[0], 0])
    theta0 = np.arctan(e0[2]/(d0[1]*e0[0] - d0[0]*e0[1]))
    theta1 = np.arctan(e1[2]/(d1[1]*e1[0] - d1[0]*e1[1]))

    R0 = rotation_matrix(d0, theta0)
    R0_inv = rotation_matrix(d0, -theta0)
    R1 = rotation_matrix(d1, theta1)
    n_e0 = R0.dot(e0)
    n_e1 = R1.dot(e1)

    phi0 = -np.arctan(n_e0[1] / n_e0[0])
    phi1 = -np.arctan(n_e1[1] / n_e1[0])

    R_phi0 = rotation_matrix_z(phi0)
    R_phi1 = rotation_matrix_z(phi1)
    R_phi0_inv = rotation_matrix_z(-phi0)

    # n_F = R_phi1.dot(R1).dot(F).dot(R0_inv).dot(R_phi0_inv)
    # T = np.zeros((3, 3))
    # T[0, 0] = 1
    # T[1, 1] = -n_F[1, 2] # -a
    # T[1, 2] = -n_F[2, 2] # -c
    # T[2, 2] = -n_F[2, 1] # b
    H0 = R_phi0.dot(R0)
    # H1 = T.dot(R_phi1).dot(R1)
    H1 = R_phi1.dot(R1)
    return H0, H1


def read_points(file):
    points = []
    with open(file, 'r') as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            if not line:
                break
            pts = [int(i) for i in line.strip().split()]
            points.append((pts[0], pts[1]))
    return points


def append_boundary(img, points, disable_mid=False):
    h, w = img.shape[0], img.shape[1]
    points.extend([(0, 0), (w - 1, 0), 
                   (w - 1, h - 1), (0, h - 1)])
    if not disable_mid:
        points.extend([(w // 2, 0), (w - 1, h // 2),
        (w // 2, h - 1), (0, h // 2)])
    return points


def in_triangle(p0, p1, p2, p):
    p0 = np.array(p0).reshape((-1,1))
    p1 = np.array(p1).reshape((-1,1))
    p2 = np.array(p2).reshape((-1,1))
    p = np.array(p)
    pa = p0 - p
    pb = p1 - p
    pc = p2 - p
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    c1 = cross(pa, pb)
    c2 = cross(pb, pc)
    c3 = cross(pc, pa)
    return (c1 * c2 >= 0) * (c2 * c3 >= 0)


if __name__ == '__main__':
    parser = ArgumentParser('View morphing')
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--ratio', type=int, default=50)
    parser.add_argument('--manual', help='Manually select postwarp', action='store_true')
    parser.add_argument('--no_auto_boundary', help='Disable auto boundary points', action='store_true')
    args = parser.parse_args()

    points0 = read_points(args.src + '.txt')
    points1 = read_points(args.dst + '.txt')

    src = cv2.imread(args.src)
    dst = cv2.imread(args.dst)
    ratio = args.ratio / 100.

    assert len(points1) == len(points0)

    F = cv2.findFundamentalMat(np.array(points0), np.array(
        points1), method=cv2.FM_8POINT)[0]
    H0, H1 = get_prewarp(F)
    H1 /= H1[2, 2]

    np.set_printoptions(precision=10, suppress=True)
    if not args.no_auto_boundary:
        append_boundary(src, points0)
        append_boundary(dst, points1)
    points0 = np.array(points0)
    points1 = np.array(points1)

    # delaunay and visualize it
    tri = Delaunay(points0)
    src_delaunay = src.copy()
    dst_delaunay = dst.copy()
    for i in range(tri.simplices.shape[0]):
        face = tri.simplices[i]
        a, b, c = face[0], face[1], face[2]
        cv2.line(src_delaunay, tuple(points0[a]), tuple(
            points0[b]), (0, 255, 255))
        cv2.line(src_delaunay, tuple(points0[a]), tuple(
            points0[c]), (0, 255, 255))
        cv2.line(src_delaunay, tuple(points0[c]), tuple(
            points0[b]), (0, 255, 255))
        cv2.line(dst_delaunay, tuple(points1[a]), tuple(
            points1[b]), (0, 255, 255))
        cv2.line(dst_delaunay, tuple(points1[a]), tuple(
            points1[c]), (0, 255, 255))
        cv2.line(dst_delaunay, tuple(points1[c]), tuple(
            points1[b]), (0, 255, 255))
    cv2.imshow('Delaunay_src', src_delaunay)
    cv2.imshow('Delaunay_dst', dst_delaunay)
    cv2.waitKey()
    cv2.destroyAllWindows()

    new_size = int(
        np.sqrt(np.power(src.shape[0], 2) + np.power(src.shape[1], 2)))
    prewarp_0 = cv2.warpPerspective(src, H0, (new_size, new_size))
    prewarp_1 = cv2.warpPerspective(dst, H1, (new_size, new_size))

    # visualize prewarps
    pw_points0 = warpKeyPoints(points0, H0)
    pw_points1 = warpKeyPoints(points1, H1)
    # for i in range(pw_points0.shape[0]):
    #     pnt = pw_points0[i]
    #     cv2.circle(prewarp_0, (int(round(pnt[0])), int(round(pnt[1]))), 1, (255, 255, 0))
    # for i in range(pw_points1.shape[0]):
    #     pnt = pw_points1[i]
    #     cv2.circle(prewarp_1, (int(round(pnt[0])), int(round(pnt[1]))), 1, (255, 255, 0))

    cv2.imshow('Prewarped I0', prewarp_0)
    cv2.imshow('Prewarped I1', prewarp_1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # actual morphing

    affine_trans0 = []  # intermediate --> warped I0
    affine_trans1 = []  # intermediate --> warped I1
    intermediate_points = []

    # calculate intermediate warped points

    for i in range(pw_points0.shape[0]):
        pnt0 = pw_points0[i]
        pnt1 = pw_points1[i]
        inter = pnt0 * (1 - ratio) + pnt1 * ratio
        intermediate_points.append(np.round(inter))
    intermediate_points = np.array(intermediate_points)
    H0_inv = np.linalg.inv(H0)
    H1_inv = np.linalg.inv(H1)

    offset = np.min(intermediate_points, axis=0)
    wh = np.ceil(np.max(intermediate_points, axis=0) - offset).astype(np.int) + 1
    offset = 0 - np.floor(offset).astype(np.int)

    # output image

    out_shape = (wh[1], wh[0])  # h x w
    if len(src.shape) == 3:
        out_shape += (src.shape[2], )
    out = np.zeros(out_shape, dtype=np.uint8)

    for i in range(tri.simplices.shape[0]):
        face = tri.simplices[i]
        a, b, c = tuple(face)  # vertex idx

        pnts = intermediate_points[[a, b, c]]
        # cache transformations
        trans0 = cv2.getAffineTransform(pnts.astype(
            np.float32), pw_points0[[a, b, c]].astype(np.float32))
        trans1 = cv2.getAffineTransform(pnts.astype(
            np.float32), pw_points1[[a, b, c]].astype(np.float32))
        affine_trans0.append(trans0)
        affine_trans1.append(trans1)

    def update_warp():
        for i in range(tri.simplices.shape[0]):
            a, b, c = tuple(tri.simplices[i])
            pnts = intermediate_points[[a, b, c]]
            xmin, ymin = tuple(np.min(pnts, axis=0).astype(int))
            xmax, ymax = tuple(np.max(pnts, axis=0).astype(int))
            p = np.array(list(product(range(xmin, xmax + 1), range(ymin, ymax + 1))))
            # N x 2 points
            p = np.concatenate([p, np.ones((p.shape[0], 1))], axis=1).T
            p0 = affine_trans0[i].dot(p)
            p1 = affine_trans1[i].dot(p)
            p0 = np.vstack([p0, np.ones(p0.shape[1])])
            p1 = np.vstack([p1, np.ones(p1.shape[1])])
            src_pnts = H0_inv.dot(p0)
            src_pnts /= src_pnts[2]
            src_pnts = np.round(src_pnts[:2]).astype(int)
            dst_pnts = H1_inv.dot(p1)
            dst_pnts /= dst_pnts[2]
            dst_pnts = np.round(dst_pnts[:2]).astype(int)
            # clip 
            src_pnts[0, :] = np.clip(src_pnts[0, :], 0, src.shape[1] - 1)
            src_pnts[1, :] = np.clip(src_pnts[1, :], 0, src.shape[0] - 1)
            dst_pnts[0, :] = np.clip(dst_pnts[0, :], 0, dst.shape[1] - 1)
            dst_pnts[1, :] = np.clip(dst_pnts[1, :], 0, dst.shape[0] - 1)

            if (np.min(src_pnts, axis=1) == np.max(src_pnts, axis=1)).all() and (np.min(dst_pnts, axis=1) == np.max(dst_pnts, axis=1)).all():
                continue

            color = src[src_pnts[1, :], src_pnts[0, :]] * (1 - ratio) + dst[dst_pnts[1, :], dst_pnts[0, :]] * ratio
            py = p[1, :].astype(int) + int(offset[1])
            px = p[0, :].astype(int) + int(offset[0])

            mask = in_triangle(pnts[0], pnts[1], pnts[2], p[:2]).reshape((-1, 1))
            out[py, px] = color * mask + out[py, px] * (1 - mask)

    # display initial warped version
    update_warp()

    out_display = out.copy()
    corners = []

    def on_sel_pnt(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            corners.append((x, y))
            cv2.circle(out_display, (x, y), 2, (0, 0, 255))
            if len(corners) > 1:
                cv2.line(out_display, corners[-2], corners[-1], (0, 255, 255), thickness=2)
            if len(corners) == 4:
                cv2.line(out_display, corners[0], corners[-1], (0, 255, 255), thickness=2)
            cv2.imshow(sel_win_name, out_display)

    # automatically detect corners
    if not args.manual:
        out_grey = cv2.cvtColor(out_display, cv2.COLOR_BGR2GRAY)
        out_grey[out_grey > 0] = 255
        contours, _ = cv2.findContours(out_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        poly_approx = cv2.approxPolyDP(max_contour, 100, True)
        if poly_approx.shape[0] == 4:
            tmp_corners = []
            for i in range(poly_approx.shape[0]):
                corner = poly_approx[i][0]
                tmp_corners.append((corner[0], corner[1]))
            tmp_corners = sorted(tmp_corners, key=lambda pair : pair[1]) # sorted y
            if tmp_corners[1][1] != tmp_corners[2][1]:
                tmp_upper = sorted(tmp_corners[:2], key=lambda pair : pair[0])
                tmp_lower = sorted(tmp_corners[2:], key=lambda pair : pair[0])
                corners.extend(tmp_upper)
                corners.extend(reversed(tmp_lower))
    
    if len(corners) != 4:
        print("Failed to automatically extract corners")
        sel_win_name = 'Double Click to Select 4 points (Clockwise, start from LeftUpper corner, "c" to clear)'
        cv2.namedWindow(sel_win_name)
        cv2.setMouseCallback(sel_win_name, on_sel_pnt)
        cv2.imshow(sel_win_name, out_display)

        while True:
            ky = cv2.waitKey()
            if ky & 0xFF == ord('c'):
                corners = []
                out_display = out.copy()
                cv2.imshow(sel_win_name, out_display)
            else:
                cv2.destroyAllWindows()
                break

    h, w = src.shape[:2]
    dst_corners = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]).astype(np.float32)
    T = cv2.getPerspectiveTransform(np.array(corners).astype(np.float32), dst_corners)
    final_res = cv2.warpPerspective(out, T, (w, h))

    cv2.imshow('Result Demo ("q" to quit, "s" to save)', final_res)
    save_name = osp.splitext(args.src)[0]
    
    while True:
        ky = cv2.waitKey() & 0xFF
        if ky == ord('q'):
            cv2.destroyAllWindows()
            break
        elif ky == ord('s'):
            cv2.imwrite(save_name + '_{}.jpg'.format(int(ratio * 100)), final_res)
