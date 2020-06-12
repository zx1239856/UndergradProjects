import argparse
import cv2
import json
import signal
from collections import OrderedDict

target = None
target_copy = None
sel_points = None

def sig_handler(sig, frame):
    cv2.destroyAllWindows()
    exit(0)

def mouse_cb(event, x, y, flags, param):
    global target, target_copy, sel_points
    if event == cv2.EVENT_LBUTTONDOWN:
        pnt = (x, y)
        sel_points.append(pnt)
        if len(sel_points) > 2:
            cv2.circle(target, pnt, 1, (0, 0, 255))
        else:
            cv2.circle(target, pnt, 1, (0, 255, 255))
        if len(sel_points) == 2:
            cv2.rectangle(target, sel_points[0], sel_points[1], (0, 255, 255))
        cv2.imshow('Label', target)
    elif event == cv2.EVENT_RBUTTONDOWN and len(sel_points) > 0:
        sel_points.pop(-1)
        target = target_copy.copy()
        if len(sel_points) > 2:
            for pnt in sel_points[2:]:
                cv2.circle(target, pnt, 1, (0, 0, 255))
        for pnt in sel_points[:2]:
            cv2.circle(target, pnt, 1, (0, 255, 255))
        if len(sel_points) >= 2:
            cv2.rectangle(target, sel_points[0], sel_points[1], (0, 255, 255))
        cv2.imshow('Label', target)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Manual label tool for face')
    parser.add_argument('-r', '--ref', type=str, help='Reference file', required=True)
    parser.add_argument('-f', '--file', type=str, help='File to label', required=True)

    args = parser.parse_args()

    signal.signal(signal.SIGINT, sig_handler)

    ref = cv2.imread(args.ref)
    target = cv2.imread(args.file)

    note = OrderedDict(json.load(open(args.ref+'.json')))
    note = note['faces'][0]
    rec = note['face_rectangle']
    cv2.namedWindow('Label')
    cv2.imshow('Label', target)

    cv2.setMouseCallback('Label', mouse_cb)
    cv2.rectangle(ref, (rec['left'], rec['top']), (rec['left'] + rec['width'], rec['top'] + rec['height']), (255, 255, 255))
    last_pnt = None

    sel_points = []
    target_copy = target.copy()
    dup = set()

    for i in note['landmark']:
        t = note['landmark'][i]
        if (t['x'], t['y']) in dup:
            print("Dup entry, ignore")
            continue
        else:
            dup.add((t['x'], t['y']))
        if last_pnt is not None:
            cv2.circle(ref, last_pnt, 1, (0, 0, 255))
        last_pnt = (t['x'], t['y'])
        cv2.circle(ref, last_pnt, 1, (0, 255, 255))
        print("Labeling: {}".format(i))
        cv2.imshow('Reference', ref)
        cv2.waitKey()
    
    ## save
    with open(args.file+'.txt', 'w') as f:
        top = min(sel_points[0][1], sel_points[1][1])
        left = min(sel_points[0][0], sel_points[1][0])
        width = abs(sel_points[1][0] - sel_points[0][0])
        height = abs(sel_points[1][1] - sel_points[0][1])
        f.write('{:d} {:d} {:d} {:d}\n'.format(top, left, width, height))
        for i in range(2, len(sel_points)):
            t = sel_points[i]
            f.write('{:d} {:d}\n'.format(t[1], t[0]))