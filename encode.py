#!/usr/bin/python3

import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt

class Encode:
    '''
    将图像编码为jpeg格式
    '''
    # 亮度量化表
    Y_Table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                     [12, 12, 14, 19, 26, 58, 60, 55],
                                     [14, 13, 16, 24, 40, 57, 69, 56],
                                     [14, 17, 22, 29, 51, 87, 80, 62],
                                     [18, 22, 37, 56, 68, 109, 103, 77],
                                     [24, 35, 55, 64, 81, 104, 113, 92],
                                     [49, 64, 78, 87, 103, 121, 120, 101],
                                     [72, 92, 95, 98, 112, 100, 103, 99]])
    # 色差量化表
    CbCr_Table = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                             [18, 21, 26, 66, 99, 99, 99, 99],
                             [24, 26, 56, 99, 99, 99, 99, 99],
                             [47, 66, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99]])

    @staticmethod
    def rgb(path):
        im = Image.open(path)
        pixels = np.array(im)
        rarray = pixels[:,:,0]
        garray = pixels[:,:,1]
        barray = pixels[:,:,2]

        return rarray, garray, barray



    @staticmethod
    def rgb2ycbcr(rarray, garray, barray):
        """ 
        rarray为r二维数组
        garray为g二维数组
        barray为b二维数组
        数组为numpy.array类型

        """
        yarray = 0.299 * rarray + 0.587 * garray + 0.114 * barray
        cbarray = 128 - 0.168 * rarray - 0.331264 * garray + 0.5 * barray
        crarray = 128 + 0.5 * rarray - 0.418688 * garray - 0.081312 * barray
        return yarray, cbarray, crarray

    @staticmethod
    def ycbcr2rgb(yarray, cbarray, crarray):
        r = yarray + 1.402 * (crarray - 128)
        g = yarray - 0.344136 * (cbarray - 128) - 0.714136 * (crarray - 128)
        b = yarray + 1.772 * (cbarray - 128)
        return r, g, b

    @staticmethod
    def sample(yarray, cbarray, crarray):
        """采样规则：4:2:0"""
        return (Encode.sampleY(yarray), Encode.sampleCb(cbarray), Encode.sampleCr(crarray))

    @staticmethod
    def dct_quantize(array, table):
        print("dct_quantize")
        row = 8
        col = 8
        new_array = Encode.pad_array(array, row, col)
        new_array = Encode.apply_fn(new_array - 128, row, col, Encode.dct)
        quantize = lambda x: Encode.quantize(x, table)
        return Encode.apply_fn(new_array, row, col, quantize)

    @staticmethod
    def invert_quantize_dct(array, table):
        print("invert_dct_quantize")
        row = 8
        col = 8
        invert_quantize = lambda x: Encode.invert_quantize(x, table)
        new_array = Encode.apply_fn(array, row, col, invert_quantize)
        new_array = Encode.apply_fn(new_array, row, col, Encode.invert_dct)
        new_array = new_array + 128

        return new_array

    @staticmethod
    def apply_fn(array, row, col, fn):
        (origin_row, origin_col) = np.shape(array)
        dct_array = np.zeros((origin_row, origin_col), dtype=float)
        for i in range(0, origin_row, row):
            for j in range(0, origin_col, col):
                sub_array = array[i: i+row, j: j+col]
                dct_array[i:i+row, j: j+col] = fn(sub_array)

        return dct_array

    @staticmethod
    def pad_array(array, row, col):
        """将array分块，然后在每个分块执行函数fn"""
        (origin_row, origin_col) = np.shape(array)

        # 如果原数组行数或列数不能整除row或者col，则填充0进行扩容
        remainder = origin_row % row
        new_array = array
        if remainder > 0:
            new_array = np.concatenate((new_array, np.zeros((row - remainder, origin_col), dtype=float)), axis=0)
        remainder = origin_col % col
        if remainder > 0:
            new_array = np.concatenate((new_array, np.zeros((origin_row, col - remainder), dtype=float)), axis=1)

        return new_array

    @staticmethod
    def quantize(array, table):
        (row, col) = np.shape(array)
        new_array = array
        for i in range(0, row):
            for j in range(0, col):
                new_array[i, j] = round(array[i, j] // table[i, j])

        return new_array

    @staticmethod
    def invert_quantize(array, table):
        (row, col) = np.shape(array)
        new_array = array
        for i in range(0, row):
            for j in range(0, col):
                new_array[i, j] = round(array[i, j] * table[i, j])

    @staticmethod
    def dct(array):
        """离散余弦傅里叶变换"""
        (row, col) = np.shape(array)
        guv = np.zeros((row, col), dtype=float)
        for u in range(0, row):
            for v in range(0, col):
                sum = 0
                for x in range(0, row):
                    for y in range(0, col):
                        gxy = array[x, y]
                        cosx = math.cos(((2 * x + 1) * u * math.pi) / 16)
                        cosy = math.cos(((2 * y + 1) * v * math.pi) / 16)
                        sum = sum + gxy * cosx * cosy

                guv[u, v] = Encode.alpha(u) * Encode.alpha(v) * sum / 4

        return guv


    @staticmethod
    def invert_dct(array):
        """反离散余弦傅里叶变换"""
        (row, col) = np.shape(array)
        fxy = np.zeros((row, col), dtype=float)
        for x in range(0, row):
            for y in range(0, col):
                sum = 0
                for u in range(0, row):
                    for v in range(0, col):
                        fuv = array[u, v]
                        cosu = math.cos(((2 * x + 1) * u * math.pi) / 16)
                        cosv = math.cos(((2 * y + 1) * v * math.pi) / 16)
                        sum = sum + fuv * cosu * cosv * Encode.alpha(u) * Encode.alpha(v)

                fxy[x, y] = sum / 4

        return fxy

    @staticmethod
    def alpha(u):
        """ normalizing scale factor to make the transformation orthonormal"""
        if u == 0:
            return 1/math.sqrt(2)
        else:
            return 1

    @staticmethod
    def sampleY(array):
        return array

    @staticmethod
    def sampleCb(array):
        (row, col) = np.shape(array)
        cb = np.zeros((row, col), dtype=float)
        for i in range(0, row, 2):
            for j in range(0, col, 2):
                cb[i // 2, j // 2] = array[i, j]

        return cb


    @staticmethod
    def sampleCr(array):
        (row, col) = np.shape(array)
        cr = np.zeros((row, col), dtype=float)
        for i in range(1, row, 2):
            for j in range(1, col, 2):
                cr[(i-1) // 2, (j-1) // 2] = array[i, j]

        return cr





if __name__ == '__main__':
    (r,g,b) = Encode.rgb("./demo.png")
    (y, cb, cr) = Encode.sample(*Encode.rgb2ycbcr(r, g, b))
    y_dct_q = Encode.dct_quantize(y, Encode.Y_Table)
    cb_dct_q = Encode.dct_quantize(cb, Encode.CbCr_Table)
    cr_dct_q = Encode.dct_quantize(cr, Encode.CbCr_Table)

    y_dct_q_in = Encode.invert_quantize_dct(y_dct_q, Encode.Y_Table)
    cb_dct_q_in = Encode.invert_quantize_dct(cb_dct_q, Encode.CbCr_Table)
    cr_dct_q_in = Encode.invert_quantize_dct(cr_dct_q, Encode.CbCr_Table)

    (r, g, b) = Encode.ycbcr2rgb(y_dct_q_in, cb_dct_q_in, cr_dct_q_in)
    c = np.column_stack((r, g, b))
    plt.imshow(c)
    plt.show()



