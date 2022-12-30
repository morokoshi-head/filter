import argparse
import cv2
import glob
import numpy as np
import os
import shutil
from tqdm import tqdm

KER_SZ = 3
SGM = 0.85

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('output_dir', help='output directory')

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    paths = glob.glob(os.path.join(input_dir, '*'))

    # Generate gaussian kernel
    ker = gen_gauss_ker(KER_SZ, SGM)

    for path in tqdm(paths):
        img = cv2.imread(path)

        # Convolute filter
        conv_img_b = conv_ker(img[:, :, 0], ker)
        conv_img_g = conv_ker(img[:, :, 1], ker)
        conv_img_r = conv_ker(img[:, :, 2], ker)

        conv_img = cv2.merge((conv_img_b, conv_img_g, conv_img_r))

        cv2.imwrite(os.path.join(output_dir, os.path.basename(path)), conv_img)

def gen_gauss_ker(ker_sz=3, sgm=0.85):
    ker = np.zeros((ker_sz, ker_sz), dtype=np.float64)
    ker_d = int((ker_sz-1) / 2)

    for y in range(-ker_d, ker_d+1):
        for x in range(-ker_d, ker_d+1):
            ker[y+ker_d, x+ker_d] = 1 / (2*np.pi*(sgm**2)) * np.exp(-(x**2+y**2)/(2*sgm**2))

    return ker / ker.sum()

def conv_ker(img, ker):
    img_h, img_w = img.shape[:2]
    ker_sz = ker.shape[0]

    ker_d = int((ker_sz-1) / 2)
    trg_img = np.zeros((ker_sz, ker_sz), dtype=np.uint8)
    conv_img = np.zeros((img_h, img_w), dtype=np.float64)

    for y in range(img_h):
        for x in range(img_w):

            for j in range(ker_sz):
                for i in range(ker_sz):
                    trg_img[j, i] = img[min(abs(y-ker_d+j), img_h-1), \
                                    min(abs(x-ker_d+i), img_w-1)]

            conv_img[y, x] = (trg_img * ker).sum()

    return conv_img

if __name__ == '__main__':
    main()