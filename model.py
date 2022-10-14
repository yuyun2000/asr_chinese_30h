import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers

# （1）标准卷积模块
def conv_block(input_tensor, filters, alpha, kernel_size=(3, 3), strides=(1, 1)):
    # 超参数alpha控制卷积核个数
    filters = int(filters * alpha)

    # 卷积+批标准化+激活函数
    x = layers.Conv2D(filters, kernel_size,
                      strides=strides,  # 步长
                      padding='same',  # 0填充，卷积后特征图size不变
                      use_bias=False)(input_tensor)  # 有BN层就不需要计算偏置

    x = layers.BatchNormalization()(x)  # 批标准化

    x = layers.ReLU(6.0)(x)  # relu6激活函数

    return x  # 返回一次标准卷积后的结果

# （2）深度可分离卷积块
def depthwise_conv_block(input_tensor, point_filters, alpha, depth_multiplier, strides=(1, 1)):
    # 超参数alpha控制逐点卷积的卷积核个数
    point_filters = int(point_filters * alpha)

    # ① 深度卷积--输出特征图个数和输入特征图的通道数相同
    x = layers.DepthwiseConv2D(kernel_size=(3, 3),  # 卷积核size默认3*3
                               strides=strides,  # 步长
                               padding='same',  # strides=1时，卷积过程中特征图size不变
                               depth_multiplier=depth_multiplier,  # 超参数，控制卷积层中间输出特征图的长宽
                               use_bias=False)(input_tensor)  # 有BN层就不需要偏置

    x = layers.BatchNormalization()(x)  # 批标准化

    x = layers.ReLU(6.0)(x)  # relu6激活函数

    # ② 逐点卷积--1*1标准卷积
    x = layers.Conv2D(point_filters, kernel_size=(1, 1),  # 卷积核默认1*1
                      padding='same',  # 卷积过程中特征图size不变
                      strides=(1, 1),  # 步长为1，对特征图上每个像素点卷积
                      use_bias=False)(x)  # 有BN层，不需要偏置

    x = layers.BatchNormalization()(x)  # 批标准化
    x = layers.ReLU(6.0)(x)  # 激活函数
    return x  # 返回深度可分离卷积结果

def conv_block_withoutrelu(
        inputs,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1)
):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(
        inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return x



def model(input_shape, dropout_rate=0.5):
    # 创建输入层
    inputs = layers.Input(shape=input_shape)

    x = conv_block(inputs, 3, 1, strides=(2, 1))
    x = conv_block(x, 32, 1, strides=(2, 2))
    x = conv_block_withoutrelu(x,64,strides=(1,1))
    x = conv_block_withoutrelu(x,64,strides=(2,2))
    # x = conv_block(x, 128, 1, strides=(2, 1))
    x = conv_block(x, 128, 1, strides=(2, 2))
    x = conv_block(x, 128, 1, strides=(1, 1))

    x = layers.Reshape(target_shape=(-1, 1280))(x)


    x = layers.Dropout(rate=dropout_rate)(x)

    x = layers.Dense(395,use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Softmax(axis=-1)(x)


    # 构建模型
    model = Model(inputs, x)

    # 返回模型结构
    return model
from Flops import try_count_flops
if __name__ == '__main__':
    # 获得模型结构
    model = model(input_shape=[1400, 80, 1])  # 随即杀死神经元的概率
    flop = try_count_flops(model)
    print(flop/1000000)
    # 查看网络模型结构
    model.summary()
    # model.save("./mbtest.h5", save_format="h5")
    # print(model.layers[-3])