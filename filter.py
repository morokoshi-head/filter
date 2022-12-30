import argparse
import cv2
import glob
import numpy as np
import os
import shutil
from tqdm import tqdm

TRANS_CH_IDX = 0
TRANS_H_IDX = 1
TRANS_W_IDX = 2

KER_SZ = 3
SGM = 0.85

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', help='input directory')
    parser.add_argument('out_dir', help='output directory')

    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    paths = glob.glob(os.path.join(in_dir, '*'))

    # Generate gaussian kernel
    ker = gen_gauss_ker(KER_SZ, SGM)

    for path in tqdm(paths):
        img = cv2.imread(path)
        ch_num = img.shape[2]

        conv_chs = []
        for i in range(ch_num):
            conv_ch = conv(img[:, :, i], ker)
            conv_chs.append(conv_ch)

        conv_img = np.array(conv_chs).transpose(TRANS_H_IDX, TRANS_W_IDX, TRANS_CH_IDX)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), conv_img)

def gen_gauss_ker(ker_sz=3, sgm=0.85):
    ker = np.zeros((ker_sz, ker_sz), dtype=np.float64)
    ker_d = int((ker_sz-1) / 2)

    for y in range(-ker_d, ker_d+1):
        for x in range(-ker_d, ker_d+1):
            ker[y+ker_d, x+ker_d] = 1 / (2*np.pi*(sgm**2)) * np.exp(-(x**2+y**2)/(2*sgm**2))

    return ker / ker.sum()

def conv(img, ker):
    img_h, img_w = img.shape[:2]
    ker_sz = ker.shape[0]

    ker_d = int((ker_sz-1) / 2)
    trg_img = np.zeros((ker_sz, ker_sz), dtype=np.uint8)
    conv_img = np.zeros((img_h, img_w), dtype=np.uint8)

    for y in range(img_h):
        for x in range(img_w):

            for j in range(ker_sz):
                for i in range(ker_sz):
                    trg_img[j, i] = img[min(abs(y-ker_d+j), img_h-1), \
                                        min(abs(x-ker_d+i), img_w-1)]

            conv_img[y, x] = np.round((trg_img * ker).sum())

    return conv_img

if __name__ == '__main__':
    main()