import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import copy
__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride):
    # number_of_weights = inp * inp * 3 * 3
    # weights = nn.Parameter(torch.rand((number_of_weights,1)) * 0.001, requires_grad=True)
    
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        # assert stride in [1, (2,1)]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == (1,1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, (3,1), stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, (1,1), (1,1), 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, (1,1), (1,1), 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, (3,1), stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, (1,1), (1,1), 0, bias=False),
                nn.BatchNorm2d(oup),
             )

    def forward(self, x):
        conv = self.conv(x)[:,:,:,1:2]
        # print(self.conv(x).shape)
        if self.identity:
            # print(x.shape)
            # print('conv shape when shortcut: ', self.conv(x).shape)
            return x + conv
        else:
            return conv

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2, width_mult=1.,last_channel = 1280):
        super(MobileNetV2, self).__init__()
        input_channel = 16

        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  input_channel, 1, (1,1)],
            # [2,  input_channel, 2, (1,1)],
            # [2,  input_channel, 2, (2,1)],
            # [6,  64, 4, 2],
            # [6,  96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]

        # self.cfgs = [
        #     # t, c, n, s
        #     [1,  16, 1, 1],
        #     [6,  24, 2, 1],
        #     [6,  32, 3, 2],
        #     [6,  64, 4, 2],
        #     [6,  96, 3, 1],
        #     [6, 160, 3, 2],
        #     [6, 320, 1, 1],
        # ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(1, input_channel, 1)]
        layers.append(nn.MaxPool2d((2,1)))
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(last_channel * width_mult, 4 if width_mult == 0.1 else 8) if width_mult >= 1.0 else 640
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(output_channel, num_classes)
        )
        # self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)