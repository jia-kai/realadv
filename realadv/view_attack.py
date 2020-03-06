import cv2
import numpy as np

import argparse
import pickle
import pprint

def cv2_imshow(name, img: np.ndarray):
    assert img.ndim == 4 and img.shape[:2] == (1, 3) and img.dtype == np.float32
    img = (np.transpose(np.squeeze(img, 0), (1, 2, 0)) * 255).astype(np.uint8)
    cv2.imshow(name, img)

def main():
    parser = argparse.ArgumentParser(description='view attack results')
    parser.add_argument('results', help='output files from attack', nargs='+')
    args = parser.parse_args()

    for fpath in args.results:
        if fpath.endswith('.json') or fpath.endswith('.finished'):
            print(f'skip {fpath}')
            continue
        print(f'display {fpath}')
        with open(fpath, 'rb') as fin:
            record = pickle.load(fin)

        for i in ['adv_out_score', 'adv_out_score_mm', 'inp_out_score']:
            print(f'{i}: {np.argmax(record[i])} {record[i]}')
        print('verifier result:', pprint.pformat(record['verify']))
        print('linf:', np.abs(record['inp'] - record['adv_inp']).max())
        cv2_imshow('input', record['inp'])
        cv2_imshow('adv', record['adv_inp'])
        if chr(cv2.waitKey(-1) & 0xff) == 'q':
            break
