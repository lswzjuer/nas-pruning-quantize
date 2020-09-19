# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-16 17:11:29
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-18 02:23:53
from torch.nn import init
from .efficientnet import EfficientNetB0
from .googlenet import GoogLeNetV1
from .mobilenetv1 import MobileNetV1
from .mobilenetv2 import MobileNetV2
# modify resnet and preact_resnet to cifar10 from cifar100
from .preact_resnet import PreActResNet18,PreActResNet34,PreActResNet50,PreActResNet101,PreActResNet152
from .resnet_cifarv1 import ResNet18,ResNet34,ResNet50,ResNet101,ResNet152
# customed cifar10-resnet,has pretrained checkpoints
from .resnet_cifarv2 import resnet20,resnet32,resnet44,resnet56,resnet110,resnet1202
from .senet import SENet18
from .shufflenetv1 import ShuffleNetG2,ShuffleNetG3
from .shufflenetv2 import ShuffleNetV2_0_5,ShuffleNetV2_1,ShuffleNetV2_1_5,ShuffleNetV2_2
from .vgg import VGG11,VGG13,VGG16,VGG19

# imagenet resnet,has pretrained checkpoints
from .resnet import resnet18,resnet34,resnet50,resnet101,resnet152


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight, 1.0)
            init.constant_(m.bias, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)



def get_models(model_name,init_type="kaiming",**kwargs):
    if model_name=="EfficientNetB0":
        model=EfficientNetB0(**kwargs)

    elif model_name=="GoogLeNetV1":
        model=GoogLeNetV1(**kwargs)

    elif model_name=="MobileNetV1":
        model=MobileNetV1(**kwargs)
    elif model_name=="MobileNetV2":
        model=MobileNetV2(**kwargs)

    elif model_name=="PreActResNet18":
        model=PreActResNet18(**kwargs)
    elif model_name=="PreActResNet34":
        model=PreActResNet34(**kwargs)
    elif model_name=="PreActResNet50":
        model=PreActResNet50(**kwargs)
    elif model_name=="PreActResNet101":
        model=PreActResNet101(**kwargs)

    elif model_name=="ResNet18":
        model=ResNet18(**kwargs)
    elif model_name=="ResNet34":
        model=ResNet34(**kwargs)
    elif model_name=="ResNet50":
        model=ResNet50(**kwargs)
    elif model_name=="ResNet101":
        model=ResNet101(**kwargs)

    elif model_name=="resnet20":
        model=resnet20(**kwargs)
    elif model_name=="resnet32":
        model=resnet32(**kwargs)
    elif model_name=="resnet44":
        model=resnet44(**kwargs)
    elif model_name=="resnet56":
        model=resnet56(**kwargs)
    elif model_name=="resnet110":
        model=resnet110(**kwargs)

    elif model_name=="SENet18":
        model=SENet18(**kwargs)

    elif model_name=="ShuffleNetG2":
        model=ShuffleNetG2(**kwargs)
    elif model_name=="ShuffleNetG3":
        model=ShuffleNetG3(**kwargs)
    elif model_name=="ShuffleNetV2_0_5":
        model=ShuffleNetV2_0_5(**kwargs)
    elif model_name=="ShuffleNetV2_1":
        model=ShuffleNetV2_1(**kwargs)
    elif model_name=="ShuffleNetV2_1_5":
        model=ShuffleNetV2_1_5(**kwargs)
    elif model_name=="ShuffleNetV2_2":
        model=ShuffleNetV2_2(**kwargs)

    elif model_name=="VGG11":
        model=VGG11(**kwargs)
    elif model_name=="VGG13":
        model=VGG13(**kwargs)     
    elif model_name=="VGG16":
        model=VGG16(**kwargs)      
    elif model_name=="VGG19":
        model=VGG19(**kwargs)
    else:
        raise  NotImplementedError("The model {} is not supported !".format(model_name))

    init_weights(model,init_type)
    return model



