from models.segnet import *
from models.fcn import *
import os

# 建立模型
def build_model(model_name, backbone, image_hw, n_classes=1):
    if model_name == 'fusionnet':
        (img_h, img_w) = image_hw
        input = Input(shape=(img_h, img_w, 3))
        model1 = segnet(n_classes, backbone, image_hw[0], image_hw[1])
        model2 = fcn(n_classes, backbone, image_hw[0], image_hw[1])
        if os.path.exists('savefiles/segnet_{}.hdf5'.format(backbone)):
            model1.load_weights('savefiles/segnet_{}.hdf5'.format(backbone))
        if os.path.exists('savefiles/fcn_{}.hdf5'.format(backbone)):
            model2.load_weights('savefiles/fcn_{}.hdf5'.format(backbone))
        out1 = model1(input)                                       #segnet输出
        out2 = model2(input)                                       #fcn输出
        output = Concatenate()([out1, out2])                       #合并两个输出
        output = Conv2D(n_classes, (1, 1), use_bias=False)(output) #对合并后的结果做卷积
        model = Model(inputs=input, outputs=output)
        weights = np.array([0.5, 0.5])                             #初始化卷积参数为[0.5,0.5],相当于segnet和fcn结果之和平均，之后训练微调
        weights = np.reshape(weights, (1,1,2,1))
        model.layers[4].set_weights(weights[None])
    elif model_name == 'segnet':
        model = segnet(n_classes, backbone, image_hw[0], image_hw[1])
    elif model_name == 'fcn':
        model = fcn(n_classes, backbone, image_hw[0], image_hw[1])
    return model

def iou_score(gt, pr):
    gt = K.cast(gt>=0.5, 'float32')
    intersection = K.sum(gt*pr)
    union = K.sum(gt+pr)-intersection
    iou = (intersection)/(union)
    iou = K.mean(iou)
    return iou