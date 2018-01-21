# coding: utf-8
import numpy as np
from activation_func import *

# 二次元データ　<=> 一次元データ変換用クラス
from common.util import im2col, col2im

# 畳み込み層
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        # フィルターと出力の形状のセットアップ
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        # フィルターのサイズに分割、1次元のベクトルに整形
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        # ベクトルの積和演算
        out = np.dot(col, col_W) + self.b

        # 出力の転置（並べ替え）
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        #　元の形状を記憶
        self.x = x
        self.col = col
        self.col_W = col_W

        return out


    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        # affine層と同様の逆伝播
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
