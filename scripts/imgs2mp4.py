import click
import os
import numpy as np
import cv2 as cv
from glob import glob
from os.path import join as pjoin

@click.command()
@click.option('--data_dir', type=str)
@click.option('--suffix', type=str, default='*.png')
@click.option('--fps', type=int, default=30)
def hello(data_dir, suffix, fps):
    image_list = sorted(glob(pjoin(data_dir, suffix)))

    imgs = []
    for img_path in image_list:
        imgs.append(cv.imread(img_path))

    height, width, layers = imgs[-1].shape
    size = (width, height)

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(pjoin(data_dir, 'output.mp4'), fourcc, fps, size, True)
    for img in imgs:
        out.write(img)

    out.release()


if __name__ == '__main__':
    hello()

