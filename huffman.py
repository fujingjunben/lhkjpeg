#!/usr/bin/python3

###########################################
#  霍夫曼算法
# 日期：2019/06/13
# 作者：刘海宽
###########################################

class Huffman:
    _R = 256 # 字符范围

    def read(self):
        """读取需要编码的流"""
        with open('../txt', 'rb') as file:
            file.read(1)
        pass


    def calFreq(self):
        """计算字符出现的频率"""
        pass

    def buildTire(self):
        """生成编码树"""
        pass

    def buildCodeTable(self):
        """生成字符编码对应表"""
        pass

    def encode(self):
        """对流进行编码"""
        pass

    def write(self):
        """输出编码过的流"""
        pass



class Node:
    def __init__(self, ch='', freq=0, left=None, right=None):
        self.ch = ch # 待编码文本中的字符
        self.freq = freq # 字符出现的频率
        self.left = left # 左子树
        self.right = right # 右子树

