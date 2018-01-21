# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
from dataset.mnist import load_mnist

# mnistのデータを取得する
def get_data():

    # ２次元データのままデータを取得
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    
    return (x_train, t_train), (x_test, t_test)


# ネットワークの初期化
def init_network():
    # 今回は初期データを用意してある
    #with open("sample_weight.pkl", 'rb') as f:
    #    network = pickle.load(f)

    TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    return network
