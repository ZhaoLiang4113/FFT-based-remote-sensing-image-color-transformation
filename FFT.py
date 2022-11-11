# -*- coding: utf-8 -*-
# @Time    : 2022/10/26 15:35
# @Email   : zhaoliang4113@gmail.com

import os
import cv2
import numpy as np
from tqdm import tqdm


# 对影像A进行快速fft变换，并保存至B_fft文件夹中
# 建议对裁剪后的小样本进行此操作，大面积的图像色彩范围更广，效果也更差


def style_transfer(source_image, target_image):
    # 快速fft变换
    h, w, c = source_image.shape
    out = []
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:, :, i])
        source_image_fshift = np.fft.fftshift(source_image_f)
        target_image_f = np.fft.fft2(target_image[:, :, i])
        target_image_fshift = np.fft.fftshift(target_image_f)

        change_length = 1
        source_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
        int(h / 2) - change_length:int(h / 2) + change_length] = \
            target_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
            int(h / 2) - change_length:int(h / 2) + change_length]

        source_image_ifshift = np.fft.ifftshift(source_image_fshift)
        source_image_if = np.fft.ifft2(source_image_ifshift)
        source_image_if = np.abs(source_image_if)

        source_image_if[source_image_if > 255] = np.max(source_image[:, :, i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1, 0).swapaxes(1, 2)

    out = out.astype(np.uint8)
    return out


def fft_save(data_path):
    img_path_A = os.path.join(data_path, 'A')
    img_path_B = os.path.join(data_path, 'B')
    img_B_save = os.path.join(data_path, 'A_fft')
    names = os.listdir(img_path_A)
    if not os.path.exists(img_B_save):
        os.mkdir(img_B_save)
    for name in tqdm(names):
        img_A = cv2.imread(os.path.join(img_path_A, name))
        img_B = cv2.imread(os.path.join(img_path_B, name))
        img_B_fft = style_transfer(source_image=img_A, target_image=img_B)
        cv2.imwrite(os.path.join(img_B_save, name), img_B_fft)

if __name__ == '__main__':
    data_path = ''
    fft_save(data_path)