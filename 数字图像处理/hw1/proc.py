import cv2
import argparse
import numpy as np

def get_cdf(hist):
    cdf = hist.cumsum()
    return cdf / float(cdf.max())

def get_lut(src_cdf, ref_cdf):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = np.searchsorted(ref_cdf, src_cdf[i])
    return lut

def match_channel_histogram(src, ref):
    h_src, _ = np.histogram(src, 256, [0, 256])
    h_ref, _ = np.histogram(ref, 256, [0, 256])
    s_cdf = get_cdf(h_src)
    r_cdf = get_cdf(h_ref)
    return cv2.LUT(src, get_lut(s_cdf, r_cdf))

def equalize_histogram(channel):
    hist, _ = np.histogram(channel, 256, [0, 256])
    cdf = get_cdf(hist)
    return cv2.LUT(channel, np.round(255 * cdf).astype(np.uint8))

def match_histogram(src, ref):
    sb, sg, sr = cv2.split(src)
    rb, rg, rr = cv2.split(ref)

    b_res = match_channel_histogram(sb, rb)
    g_res = match_channel_histogram(sg, rg)
    r_res = match_channel_histogram(sr, rr)

    return cv2.convertScaleAbs(cv2.merge([b_res, g_res, r_res]))

def match_b_w_gr(src):
    sb, sg, sr = cv2.split(src)
    sb = match_channel_histogram(sb, (sg + sr) / 2)
    return cv2.convertScaleAbs(cv2.merge([sb, sg, sr]))

def match_g_w_br(src):
    sb, sg, sr = cv2.split(src)
    sg = match_channel_histogram(sg, (sb + sr) / 2)
    return cv2.convertScaleAbs(cv2.merge([sb, sg, sr]))

def match_r_w_bg(src):
    sb, sg, sr = cv2.split(src)
    sr = match_channel_histogram(sr, (sb + sg) / 2)
    return cv2.convertScaleAbs(cv2.merge([sb, sg, sr]))

def equalize_all_channels(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, 256, [0, 256])
    lut = np.round(255 * get_cdf(hist)).astype(np.uint8)
    res = cv2.LUT(src, lut)
    return cv2.convertScaleAbs(res)

def adjust_gamma(image, gamma=1.0):
    return np.array(np.power(image / 255., 1. / gamma) * 255).astype("uint8")

def logistic_lut(image):
    return np.array(255. / (1. + 0.2 * np.exp(-12. * (image / 255.) + 6))).astype("uint8")

def alter_saturation(image, sat):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype('float')
    img[:, :, 1] *= np.clip(sat, 0, 500) / 50.
    img[img < 0] = 0
    img[img > 255] = 255
    return cv2.cvtColor(np.round(img).astype('uint8'), cv2.COLOR_HSV2BGR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('hist_matching')
    parser.add_argument('--src', required=True)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--saturation', default=50.0, type=float)
    #parser.add_argument('--ref', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    src = cv2.imread(args.src)
    out = src
    #ref = cv2.imread(args.ref)

    #out = logistic_lut(src)
    out = alter_saturation(out, args.saturation)
    out = adjust_gamma(out, args.gamma)
    cv2.imshow('Source: ', cv2.resize(src, (640, 480)))
    cv2.imshow('Out: ', cv2.resize(out, (640, 480)))
    cv2.imwrite(args.out, out)

    cv2.waitKey(0)
