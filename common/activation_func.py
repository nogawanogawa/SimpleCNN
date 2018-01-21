# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

#シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#シグモイド関数のバックプロパゲーション関数
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

#ステップ関数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

#ReLU関数
def relu(x):
    return np.maximum(0, x)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換                                                                                            
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size



# サンプル描画用
#x = np.arange(-5.0, 5.0, 0.1)
#y1 = relu(x)
#y2 = step_function(x)
#y3 = sigmoid(x)

#plt.plot(x, y1)
#plt.ylim(-0.1, 1.1) #図で描画するy軸の範囲を指定
#plt.plot(x, y2)
#plt.plot(x, y3)
#plt.show()

