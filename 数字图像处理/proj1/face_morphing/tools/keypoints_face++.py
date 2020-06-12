import argparse
import json
import os
from random import randint
from collections import OrderedDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get 106 keypoints of face keypoints via Face++ API")
    parser.add_argument('-f', '--file', type=str, help="Filename of image", required=True)
    parser.add_argument('-k', '--key', type=str, help="API key", required=True)
    parser.add_argument('-s', '--sec', type=str, help="API Secret", required=True)
    args = parser.parse_args()
    if not os.path.exists(args.file + '.json'):
        os.system('curl -X POST "https://api-cn.faceplusplus.com/facepp/v3/detect" -F "api_key={api_key}" -F "api_secret={api_sec}" -F "image_file=@{fname}" -F "return_landmark=2" > {fname}.json'.format(fname=args.file, api_key=args.key, api_sec=args.sec))
    res = OrderedDict(json.load(open(args.file+'.json')))
    dup = set()
    with open(args.file+'.txt', 'w') as f:
        rec = res['faces'][0]['face_rectangle']
        f.write('{:d} {:d} {:d} {:d}\n'.format(rec['top'], rec['left'], rec['width'], rec['height']))
        for i in res['faces'][0]['landmark']:
            t = res['faces'][0]['landmark'][i]
            if (t['y'], t['x']) in dup:
                print("Found duplicate entry {}, ignore it".format(i))
            else:
                dup.add((t['y'], t['x']))
                f.write('{:d} {:d}\n'.format(t['y'], t['x']))
