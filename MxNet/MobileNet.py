# -*- coding: UTF-8 -*- 
# Authorized by Vlon Jang
# Created on 2017-06-15
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# ©2015-2017 All Rights Reserved.
#
import mxnet as mx

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

def get_symbol(num_classes=1000):
    data = mx.symbol.Variable(name="data") # 224
    conv_1 = Conv(data, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1") # 112
    conv_2_dw = Conv(conv_1, num_group=32, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw") # 112
    conv_2 = Conv(conv_2_dw, num_filter=64, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_2") # 112
    conv_3_dw = Conv(conv_2, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_3_dw") # 56
    conv_3 = Conv(conv_3_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_3") # 56
    conv_4_dw = Conv(conv_3, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_4_dw") # 56
    conv_4 = Conv(conv_4_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_4") # 56
    conv_5_dw = Conv(conv_4, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_5_dw") # 28
    conv_5 = Conv(conv_5_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_5") # 28
    conv_6_dw = Conv(conv_5, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_6_dw") # 28
    conv_6 = Conv(conv_6_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6") # 28
    conv_7_dw = Conv(conv_6, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_7_dw") # 14
    conv_7 = Conv(conv_7_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_7") # 14

    conv_8_dw = Conv(conv_7, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_8_dw") # 14
    conv_8 = Conv(conv_8_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_8") # 14
    conv_9_dw = Conv(conv_8, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_9_dw") # 14
    conv_9 = Conv(conv_9_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_9") # 14
    conv_10_dw = Conv(conv_9, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_10_dw") # 14
    conv_10 = Conv(conv_10_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_10") # 14
    conv_11_dw = Conv(conv_10, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_11_dw") # 14
    conv_11 = Conv(conv_11_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_11") # 14
    conv_12_dw = Conv(conv_11, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_12_dw") # 14
    conv_12 = Conv(conv_12_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_12") # 14

    conv_13_dw = Conv(conv_12, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_13_dw") # 7
    conv_13 = Conv(conv_13_dw, num_filter=1024, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_13") # 7    
    conv_14_dw = Conv(conv_13, num_group=1024, num_filter=1024, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_14_dw") # 7
    conv_14 = Conv(conv_14_dw, num_filter=num_classes, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_14") # 7

    pool = mx.sym.Pooling(data=conv_14, global_pool=True, kernel=(7, 7), stride=(1, 1), pool_type="avg", name="global_pool")
    softmax = mx.symbol.SoftmaxOutput(data=pool, name='softmax')
    return softmax


if __name__ == '__main__':
    get_symbol(3)
