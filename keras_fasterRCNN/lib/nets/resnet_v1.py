"""
# 使用Keras搭建faster-rcnn特征提取基础网络Conv_5: resnet
"""
import os 
import keras 
import keras.models as KM 
import keras.layers as KL 
import keras.backend as KB 

def _bottleneckBlock(InputTensor, KernelSize, Filters, Strides, 
                    Stage, Block, Trainable=True, ShutCut=False):
    # 创建说明信息
    conv_name_base = 'res' + str(Stage) + Block + '_branch'
    bn_name_base = 'bn' + str(Stage) + Block + '_branch'

    # 获取卷积核的大小
    Ft1, Ft2, Ft3 = Filters 

    x = KL.Conv2D(Ft1, kernel_size=1, strides=Strides, padding='same', 
                    name=conv_name_base+'2a', trainable=Trainable)(InputTensor)
    x = KL.BatchNormalization(name=bn_name_base+'2a')(x)
    x = KL.Activation('relu')(x) 

    x = KL.Conv2D(Ft2, kernel_size=KernelSize, padding='same', name=conv_name_base+'2b', 
                    trainable=Trainable)(x) 
    x = KL.BatchNormalization(name=bn_name_base+'2b')(x)
    x = KL.Activation('relu')(x) 

    x = KL.Conv2D(Ft3, kernel_size=1, padding='same', name=conv_name_base+'2c', 
                    trainable=Trainable)(x)
    x = KL.BatchNormalization(name=bn_name_base+'2c')(x)
    
    if ShutCut:
        shutcut_x = KL.Conv2D(Ft3, kernel_size=1, strides=Strides, padding='same', 
                        name=conv_name_base+'1', trainable=Trainable)(InputTensor)
        shutcut_x = KL.BatchNormalization(name=bn_name_base+'1')(shutcut_x)
    else:
        shutcut_x = InputTensor
    
    x = KL.Add()([x, shutcut_x])
    x = KL.Activation('relu')(x) 

    return x 

def nnBase(image_input=KL.Input(shape=(256, 256, 3)), BlockSizes=[3, 4, 6, 3], Trainable=False):
    """
    :param image_input: KL.Input() 对象， a tensor variable
    :param BlockSizes:
    :param Trainable:
    :return:
    """
    # image_input = KL.Input(shape=InputShape)

    # 基础模块 stage=1
    x = KL.ZeroPadding2D(padding=(3, 3))(image_input)
    x = KL.Conv2D(64, kernel_size=7, strides=(2, 2), 
                padding='valid', name='conv1', trainable=Trainable)(x)
    x = KL.BatchNormalization(name='bn_conv1')(x)
    x = KL.Activation('relu')(x) 
    #print("1_OK")
    # stage=2
    x = _bottleneckBlock(x, 3, [64, 64, 256], Strides=(1, 1), 
                        Stage=2, Block=str(1), Trainable=Trainable, ShutCut=True)
    for i in range(1, BlockSizes[0]):
        x = _bottleneckBlock(x, 3, [64, 64, 256], Strides=(1, 1), 
                            Stage=2, Block=str(i+1), Trainable=Trainable, ShutCut=False)
    #print("2_OK")
    # stage=3 
    x = _bottleneckBlock(x, 3, [128, 128, 512], Strides=(2, 2), 
                        Stage=3, Block=str(1), Trainable=Trainable, ShutCut=True)
    for i in range(1, BlockSizes[1]):
        x = _bottleneckBlock(x, 3, [128, 128, 512], Strides=(1, 1), 
                            Stage=3, Block=str(i+1), Trainable=Trainable, ShutCut=False)
    #print("3_OK")
    # stage=4 
    x = _bottleneckBlock(x, 3, [256, 256, 1024], Strides=(2, 2), 
                        Stage=4, Block=str(1), Trainable=Trainable, ShutCut=True)
    for i in range(1, BlockSizes[2]):
        x = _bottleneckBlock(x, 3, [256, 256, 1024], Strides=(1, 1), 
                            Stage=4, Block=str(i+1), Trainable=Trainable, ShutCut=False)
    #print("4_OK")
    # stage=5 
    x = _bottleneckBlock(x, 3, [512, 512, 2048], Strides=(2, 2), 
                        Stage=5, Block=str(1), Trainable=Trainable, ShutCut=True)
    for i in range(1, BlockSizes[3]):
        x = _bottleneckBlock(x, 3, [512, 512, 2048], Strides=(1, 1), 
                            Stage=5, Block=str(i+1), Trainable=Trainable, ShutCut=False)
    #print("5_OK")
    return x, image_input


if __name__ == "__main__":
    InputShape = (256, 256, 3)
    image_input = KL.Input(shape=InputShape)
    feature_output, image_input = nnBase(image_input=image_input, Trainable=True)

    model = KM.Model(inputs=image_input, outputs=feature_output)

    model.summary() 

    # model graph 
    keras.utils.plot_model(model, to_file='resnet50_model.png')

    for layer in model.layers:
        print(layer.name, layer.trainable)

