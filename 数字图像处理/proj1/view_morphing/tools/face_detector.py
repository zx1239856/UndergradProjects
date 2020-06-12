import dlib
from imutils import face_utils
import numpy as np
import cv2
from argparse import ArgumentParser

def get_face_landmarks(img_path):
    img = cv2.imread(img_path)
    path = './shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path)

    rects = detector(img, 0)

    if len(rects) > 1 or len(rects) <= 0:
        print("Invalid number of faces in picture. Expected 1 only.")
        exit(0)

    shape = predictor(img, rects[0])
    shape = face_utils.shape_to_np(shape)

    x1 = shape[:, 0][np.newaxis].T
    y1 = shape[:, 1][np.newaxis].T

    with open(img_path + '.txt', 'w') as fp:
        fp.write('0 0 0 0\n') # ignores bbox here
        for idx in range(shape.shape[0]):
            pnt = shape[idx]
            fp.write('{} {}\n'.format(pnt[0], pnt[1]))
            cv2.circle(img, (pnt[0], pnt[1]), 1, (0, 255, 255))
    
    cv2.imshow('Result', img)
    cv2.waitKey()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img', required=True, type=str)
    args = parser.parse_args()
    get_face_landmarks(args.img)