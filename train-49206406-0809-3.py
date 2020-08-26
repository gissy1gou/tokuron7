import warnings
warnings.filterwarnings('ignore')

from glob import glob
dog_filepaths = glob('train4000/dog/*.jpg')
cat_filepaths = glob('train4000/cat/*.jpg')
from PIL import Image
import numpy as np

x, t = [], []
# 犬
for filepath in dog_filepaths:
    img = Image.open(filepath)
    img = img.resize((224, 224))#追加
    img = np.array(img)
    x.append(img)
    t.append(np.array(0))  # 犬は 0 とする
# 猫
for filepath in cat_filepaths:
    img = Image.open(filepath)
    img = img.resize((224, 224))#追加
    img = np.array(img)
    x.append(img)
    t.append(np.array(1))  # 猫は 0 とする

# 全体を numpy の形式に変換
x = np.array(x)  # f は float32
t = np.array(t)  # i は int32

# 特徴量の正規化 (0~1の範囲に)
x = x / 255

from sklearn.model_selection import train_test_split

# train : val = 0.7 : 0.3 の割合で分割する
train_x, val_x, train_t, val_t = train_test_split(x, t, train_size=0.7, random_state=0)


# 4. モデルを定義
import tensorflow as tf
import os
import random

def reset_seed(seed=0):

    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed) #　random関数のシードを固定
    np.random.seed(seed) #numpyのシードを固定
    tf.random.set_seed(seed) #tensorflowのシードを固定
# シードの固定を実行
reset_seed()
from tensorflow.keras import models, layers
# モデルの定義
model = models.Sequential([
    # Convolution
    layers.Conv2D(3, kernel_size=(3,3), padding='same', activation='relu', input_shape=(224, 224, 3)),
    # Pooling
    layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    # ベクトル化 (Flatten)
    layers.Flatten(),
    # 全結合層
    layers.Dense(100, activation='relu'),
    # 全結合層
    layers.Dense(2, activation='softmax') 
])

# モデルのコンパイル
model.compile(
    optimizer='SGD',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# モデルの学習
history = model.fit(
    train_x, train_t,
    batch_size=16,
    #epochs=10,
    epochs=20,
    validation_data=(val_x, val_t)
)

# HDF5 という形式で保存（TensorFlowではこちらを用いるようです）
model.save('dog_cat_cnn.h5')

# onnx形式で出力
import onnx
import sys
import pprint
pprint.pprint(sys.path)

import onnxruntime
onnxruntime.__version__

import keras2onnx
keras2onnx.__version__

# Keras -> ONNX の変換
onnx_model = keras2onnx.convert_keras(model, model.name)

# モデルを保存
keras2onnx.save_model(onnx_model, 'dog_cat_cnn.onnx')
















