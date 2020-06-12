import numpy as np
import cv2, argparse
from tqdm import tqdm
from numba import jit

MASK_THRES = 240.
PROTECT_MASK_ENERGY = 1E6


def rotate_image(img, cw):
    return np.rot90(img, 1 if cw else 3)


def visualize_seam(img, seam_idxs, rotate, alpha=0.5):
    img = img.astype(np.float64)
    if rotate:
        img = rotate_image(img, True)
    rows = range(img.shape[0])
    for seam in seam_idxs:
        img[rows, seam] *= (1 - alpha)
        img[rows, seam] += alpha * np.array([0., 255., 255.])
    img = np.round(img).astype(np.uint8)
    if rotate:
        img = rotate_image(img, False)
    return img

@jit
def fast_argmin_axis_0(a):
    matches = np.nonzero((a == np.min(a, axis=0)).ravel())[0]
    rows, cols = np.unravel_index(matches, a.shape)
    argmin_array = np.empty(a.shape[1], dtype=np.intp)
    argmin_array[cols] = rows
    return argmin_array


@jit
def get_forward_energy(img):
    """
    Forward energy defined in "Improved Seam Carving for Video Retargeting" by Rubinstein, Shamir, Avidan.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    energy = np.zeros(img.shape[:2])
    m = np.zeros_like(energy)

    U = np.roll(img, 1, axis=0)  # i - 1
    L = np.roll(img, 1, axis=1)  # j - 1
    R = np.roll(img, -1, axis=1)  # j + 1

    cU = np.abs(R - L)
    cR = cU + np.abs(U - R)
    cL = cU + np.abs(U - L)

    # dp
    for i in range(1, img.shape[0]):
        mU = m[i - 1]  # M(i-1, j)
        mL = np.roll(mU, 1)  # M(i-1, j-1)
        mR = np.roll(mU, -1)  # M(i-1, j+1)

        m_all = np.array([mU, mL, mR])
        c_all = np.array([cU[i], cL[i], cR[i]])
        m_all += c_all

        argmins = fast_argmin_axis_0(m_all)
        m[i] = np.choose(argmins, m_all)
        energy[i] = np.choose(argmins, c_all)

    return energy


@jit
def get_min_seam(img, protect_mask=None, remove_mask=None):
    h, w = img.shape[:2]
    M = get_forward_energy(img)
    if protect_mask is not None:
        M[np.where(protect_mask > MASK_THRES)] = PROTECT_MASK_ENERGY
    if remove_mask is not None:
        M[np.where(remove_mask > MASK_THRES)] = -100 * PROTECT_MASK_ENERGY

    dp = np.zeros_like(M, dtype=np.int32)
    for i in range(1, h):
        for j in range(0, w):
            left = max(j - 1, 0)
            idx = np.argmin(M[i - 1, left: j + 2])
            dp[i, j] = idx + left
            M[i, j] += M[i - 1, idx + left]

    seam_idx = []
    j = np.argmin(M[-1])
    for i in range(h - 1, -1, -1):
        seam_idx.append(j)
        j = dp[i, j]

    seam_idx.reverse()
    return np.array(seam_idx)


@jit
def _remove_seam_impl(img, seam_idx):
    h, w = img.shape[:2]
    bin_mask = np.ones((h, w), dtype=np.int32)
    bin_mask[np.arange(h), seam_idx] = 0
    new_shape = list(img.shape)
    new_shape[1] -= 1
    if len(img.shape) == 3:
        # color img
        bin_mask = np.stack([bin_mask] * 3, axis=2)
    return img[bin_mask > 0].reshape(new_shape)


@jit
def _add_seam_impl(img, seam_idx):
    new_shape = list(img.shape)
    new_shape[1] += 1
    output = np.zeros(new_shape)
    h, w = img.shape[:2]
    for row in range(h):
        col = seam_idx[row]
        left = max(col - 1, 0)
        output[row, :col] = img[row, :col]
        output[row, col] = np.average(img[row, left: col + 2], axis=0)
        output[row, col + 1:] = img[row, col:]
    return np.round(output).astype(np.uint8)


@jit
def manip_seam(img, delta, mask=None):
    seams = []
    temp_img = img.copy()
    temp_mask = mask.copy() if mask is not None else None
    last_seam = None
    for _ in tqdm(range(abs(delta))):
        seam_idx = get_min_seam(temp_img, temp_mask)
        temp_img = _remove_seam_impl(temp_img, seam_idx)
        if temp_mask is not None:
            temp_mask = _remove_seam_impl(temp_mask, seam_idx)
        if last_seam is not None:
            seam_idx[seam_idx > last_seam] += 1
        last_seam = seam_idx
        seams.append(last_seam)

    if delta < 0:
        return temp_img, temp_mask, seams
    else:
        seams.reverse()

        for i in tqdm(range(delta)):
            seam = seams[i]
            img = _add_seam_impl(img, seam)
            if mask is not None:
                mask = _add_seam_impl(mask, seam)
            for j in range(delta):
                seams[j][np.where(seams[j] > seam)] += 1
        return img, mask, seams


def seam_carve(img, dx, dy, mask=None, vis=False):
    h, w = img.shape[:2]
    assert w + dx > 0 and h + dy > 0 and dx < w and dy < h
    if mask is not None:
        m_h, m_w = mask.shape[:2]
        assert h == m_h and w == m_w
    output = img.copy()
    seams_x = None
    seams_y = None
    if dx != 0:
        output_x, mask, seams_x = manip_seam(output, dx, mask)
    else:
        output_x = output
    if dy != 0:
        output = rotate_image(output_x, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask, seams_y = manip_seam(output, dy, mask)
        output = rotate_image(output, False)
    else:
        output = output_x
    if vis:
        if seams_x is not None:
            vis_dst = output_x if dx > 0 else img
            vis_dst = visualize_seam(vis_dst, seams_x, False)
            cv2.imshow('Seams_X', vis_dst)
            cv2.imwrite('seams_x.jpg', vis_dst)
            cv2.waitKey(1)
        if seams_y is not None:
            vis_dst = output if dy > 0 else output_x
            vis_dst = visualize_seam(vis_dst, seams_y, True)
            cv2.imshow('Seams_Y', vis_dst)
            cv2.imwrite('seams_y.jpg', vis_dst)
            cv2.waitKey(1)
    return output


def object_removal(img, remove_mask, protect_mask=None, remove_horiz=False):
    h, w = img.shape[:2]
    output = img
    if remove_horiz:
        output = rotate_image(output, True)
        remove_mask = rotate_image(remove_mask, True)
        if protect_mask is not None:
            protect_mask = rotate_image(protect_mask, True)

    last = -1
    while True:
        remain = len(np.where(remove_mask > MASK_THRES)[0])
        if remain == last:
            break
        last = remain
        print("Remaining: {}".format(remain))
        if remain == 0:
            break
        seam_idx = get_min_seam(output, protect_mask, remove_mask)
        output = _remove_seam_impl(output, seam_idx)
        remove_mask = _remove_seam_impl(remove_mask, seam_idx)
        if protect_mask is not None:
            protect_mask = _remove_seam_impl(protect_mask, seam_idx)

    cv2.imshow('Partial res', output)
    cv2.imwrite('partial.jpg', output)
    cv2.waitKey()
    num_enlarge = (h if remove_horiz else w) - output.shape[1]
    print("Start to restore")
    output, _, __ = manip_seam(output, num_enlarge, protect_mask)
    print("Done")
    if remove_horiz:
        output = rotate_image(output, False)

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Seam carve")
    parser.add_argument('--src', help="Source image", required=True)
    parser.add_argument('--op', help="Operation (seam, remove)", default="seam")
    parser.add_argument('--mask', help="Protective mask of the image", default="")
    parser.add_argument('--rm_mask', help="Mask of the part to remove (only in REMOVE op)", default="")
    parser.add_argument('--out', help="Output file (optional)", default="")
    parser.add_argument('--disp', help="Display result", action="store_true", default=True)
    parser.add_argument('--dx', help="Number of cols to add/remove", type=int, default=0)
    parser.add_argument('--dy', help="Number of rows to add/remove", type=int, default=0)
    parser.add_argument('--vis_seam', help="Visualize seam", action="store_true", default=False)

    args = parser.parse_args()

    src = cv2.imread(args.src)
    assert src is not None
    mask = cv2.imread(args.mask, 0) if args.mask else None

    if args.op == 'seam':
        output = seam_carve(src, args.dx, args.dy, mask, args.vis_seam)
    elif args.op == 'remove':
        rm_mask = cv2.imread(args.rm_mask, 0)
        output = object_removal(src, rm_mask, mask)
    else:
        raise NotImplementedError("Unsupported type of op: {}".format(args.op))
    if args.out:
        cv2.imwrite(args.out, output)
    if args.disp:
        cv2.imshow('Result', output)
        cv2.waitKey(0)
