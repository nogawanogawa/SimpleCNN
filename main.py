# coding: utf-8
import sys, os
sys.path.append('common/')

import numpy as np
import matplotlib.pylab as plt

from simple_convnet import SimpleConvNet
from trainer import Trainer
from initialize_func import get_data, init_network


# mnistデータの取得
(x_train, t_train), (x_test, t_test) = get_data()

max_epochs = 20

# networkの初期化(CNN)
network = SimpleConvNet(input_dim=(1,28,28), # 画像のサイズ（グレースケール、縦、横）
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100,# 隠れ層のノードの数
                        output_size=10,# 出力は0-10の予想値
                        weight_init_std=0.01) # ネットワークの重みの初期値

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
