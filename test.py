import cv2
from model import *

## 参数设置
image_filename = 'dataset/testing/00001.png'
model_name = 'fusionnet'   # fusionnet / segnet / fcn
backbone = 'vgg16'      # basic_encoder / vgg16 / resnet50

image_hw = (768, 576)      # 32的倍数
model_path = 'savefiles/{0}_{1}.hdf5'.format(model_name, backbone)

## 加载模型
model = build_model(model_name, backbone, image_hw)
model.load_weights(model_path)

## 测试
image = cv2.imread(image_filename) / 255.0
image = cv2.resize(image, image_hw[::-1])
pr = model.predict(image[None])
pr = image*(pr[0]>0.4)*pr[0]
pr = (255*pr).astype(np.uint8)
cv2.imshow('result', pr)
cv2.imwrite('savefiles/{0}_{1}_00001.png'.format(model_name, backbone), pr)
cv2.waitKey(-1)