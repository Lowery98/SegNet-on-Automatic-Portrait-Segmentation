from keras.callbacks import ModelCheckpoint
from keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import pandas as pd
from data_loader import *
from model import *

## 参数设置
model_name = 'fusionnet'   # fusionnet / segnet / fcn
backbone = 'resnet50'      # basic_encoder / vgg16 / resnet50
image_hw = (768, 576)   # 32的倍数
model_path = 'savefiles/{0}_{1}.hdf5'.format(model_name, backbone)
datadir = 'dataset'
batch_size = 1

## 加载数据
data_loader = data_loader(datadir, batch_size, image_hw)

## 建立模型
model = build_model(model_name, backbone, image_hw)
if os.path.exists(model_path):
    print('loading model:', model_path)
    model.load_weights(model_path)

## 模型训练
checkpoint = ModelCheckpoint(model_path, save_best_only=True, verbose=1)
model.compile(SGD(lr=1e-2), loss=binary_crossentropy, metrics=[iou_score])
history = model.fit_generator(data_loader.data_generator('train'),
                              steps_per_epoch=len(data_loader.train_files)//batch_size,
                              validation_data=data_loader.data_generator('test'),
                              validation_steps=len(data_loader.test_files)//batch_size,
                              epochs=30, verbose=2, callbacks=[checkpoint])

## 训练结果显示与保存
name = '{0}_{1}'.format(model_name, backbone)
data = pd.DataFrame(history.history)
data.to_csv('savefiles/{}_history.csv'.format(name), index=False)

plt.figure(1)
plt.plot(history.history['loss'], marker='o')
plt.plot(history.history['val_loss'], marker='o')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('savefiles/{}_loss.png'.format(name))
plt.show()

plt.figure(2)
plt.plot(history.history['iou_score'], marker='o')
plt.plot(history.history['val_iou_score'], marker='o')
plt.title('model iou_score')
plt.ylabel('iou_score')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('savefiles/{}_iou_score.png'.format(name))
plt.show()