import torch
from torch import nn
import math
from swin_transformer import *
from collections import OrderedDict
import numpy as np

from torch.nn import TransformerEncoder, TransformerEncoderLayer



class Gate(nn.Module):
    def __init__(self, in_plane):
        super(Gate, self).__init__()
        self.gate = nn.Conv3d(in_plane, in_plane, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, rgb_fea):
        gate = torch.sigmoid(self.gate(rgb_fea))
        #print('请打印出这个时间、空间、和通道的注意力的形状gate.shape', gate.shape)
        gate_fea = rgb_fea * gate + rgb_fea

        return gate_fea


class FusionNetwork(nn.Module):
    def __init__(self):
        super(FusionNetwork, self).__init__()
        # 定义卷积层，保持输入和输出的通道数一致
        # 用来音视融合和触视融合之后将两个融合特征进行合并
        # 其输入是四个融合之后的视触特征和四个融合之后的视听特征
        self.conv1 = nn.Conv3d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(768, 768, kernel_size=3, stride=1, padding=1)

    def forward(self, h1, h2, h3, h4, a1, a2, a3, a4):
        # 通道维度相加
        f1 = h1 + a1
        f2 = h2 + a2
        f3 = h3 + a3
        f4 = h4 + a4

        # 通过卷积层
        y1 = self.conv1(f1)
        y2 = self.conv2(f2)
        y3 = self.conv3(f3)
        y4 = self.conv4(f4)

        return y1, y2, y3, y4

'''视频编码模型'''
class VideoSaliencyModel(nn.Module):
    def __init__(self, pretrain=None):
        super(VideoSaliencyModel, self).__init__()

        self.backbone = SwinTransformer3D(pretrained=pretrain)
        self.decoder = DecoderConvUp()
    def forward(self, x):
        x, [y1, y2, y3, y4] = self.backbone(x)
        #print('第一到第三阶段的shape', y1.shape, y2.shape, y3.shape, y4.shape)
        #print('第四阶段的输出, 即x的shape', x.shape)

        return self.decoder(x, y3, y2, y1)


'''触觉编码模型'''
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=3200, model_dim=256, feature_dim=128, n_heads=8, n_layers = 4, dropout=0.1):
        """
        构建一个基于 Transformer 的时序处理网络
        :param input_dim: 输入的每个时间点的数据点维度 (3200)
        :param model_dim: Transformer 中的隐藏维度 (通常比 input_dim 小)
        :param feature_dim: 输出的最终特征维度 (128)
        :param n_heads: 多头注意力的头数
        :param n_layers: Transformer 编码层的数量
        :param dropout: dropout 概率
        """
        super(TimeSeriesTransformer, self).__init__()

        # 输入降维层: 将 input_dim (3200) 映射到 model_dim
        self.fc_in = nn.Linear(input_dim, model_dim)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 全局池化层: 聚合时间维度
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 输出形状为 (batch, model_dim, 1)

        # 输出特征层
        self.fc_out = nn.Linear(model_dim, feature_dim)

    def forward(self, x):
        """
        :param x: 输入形状 (batch, seq_len, input_dim)
        :return: 输出形状 (batch, feature_dim)
        """
        # 输入降维 (batch, seq_len, input_dim) -> (batch, seq_len, model_dim)
        x = torch.relu(self.fc_in(x))

        # Transformer 要求输入形状为 (seq_len, batch, model_dim)
        x = x.permute(1, 0, 2)  # 转换为 (seq_len, batch, model_dim)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)  # 输出形状为 (seq_len, batch, model_dim)

        # 转回 (batch, seq_len, model_dim)
        x = x.permute(1, 0, 2)

        # 全局池化: (batch, seq_len, model_dim) -> (batch, model_dim, 1)
        x = x.permute(0, 2, 1)  # 转换为 (batch, model_dim, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, model_dim)

        # 映射到最终特征维度 (batch, model_dim) -> (batch, feature_dim)
        x = self.fc_out(x)
        return x


class haptic_encoder_fenlei(nn.Module):

    def __init__(self, input_size1):
        super(haptic_encoder_fenlei, self).__init__()

        self.densenet4 = nn.Linear(input_size1, 256)
        self.relu1 = nn.ReLU()
        self.densenet5 = nn.Linear(256, 18)

    def forward(self, x):
        x = self.densenet4(x)
        x = self.relu1(x)
        x1 = self.densenet5(x)  # 分类网络这里不需要使用softmax函数,

        return x1

    def penultimate_layer_output(self, x):
        x = self.densenet4(x)
        x = self.relu1(x)
        return x

class AffineTransform(nn.Module):
    def __init__(self, input_dim=128):
        """
        仿射变换模块
        :param input_dim: 输入特征的通道维度 (128)
        """
        super(AffineTransform, self).__init__()

        # 固定的目标形状（除 batch_size 外）
        self.target_shapes = [
            (96, 8, 56, 96),
            (192, 8, 28, 48),
            (384, 8, 14, 24),
            (768, 8, 7, 12)
        ]

        # 为每个目标形状创建一个对应的全连接层
        self.fc_layers = nn.ModuleList([
            nn.Linear(input_dim, shape[0]) for shape in self.target_shapes  # 映射到目标通道数 C
        ])
    def forward(self, x):
        """
        前向传播
        :param x: 输入特征，形状为 (batch, input_dim)
        :return: 变换后的特征列表，对应目标形状
        """
        batch_size = x.size(0)  # 获取 batch_size
        transformed_features = []

        for i, shape in enumerate(self.target_shapes):
            C, T, H, W = shape

            # 全连接层映射通道维度
            x_mapped = self.fc_layers[i](x)  # (batch, C)

            # 添加时间和空间维度
            x_expanded = x_mapped.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (batch, C, 1, 1, 1)

            # 重复时间和空间维度
            x_repeated = x_expanded.repeat(1, 1, T, H, W)  # (batch, C, T, H, W)
            transformed_features.append(x_repeated)

        return transformed_features[0], transformed_features[1], transformed_features[2], transformed_features[3]




'''联合视触觉的编码模型'''
class Video_HAPTIC_SaliencyModel(nn.Module):
    def __init__(self, pretrain=None):
        super(Video_HAPTIC_SaliencyModel, self).__init__()

        self.backbone = SwinTransformer3D(pretrained=pretrain)
        self.fusion_1 = cross_attetion_fusion(96)
        self.fusion_2 = cross_attetion_fusion(192)
        self.fusion_3 = cross_attetion_fusion(384)
        self.fusion_4 = cross_attetion_fusion(768)
        self.decoder = Decoder_pyramid()

    def forward(self, visual, haptic_1, haptic_2, haptic_3, haptic_4):
        x, [y1, y2, y3, y4] = self.backbone(visual)
        print('第一到第三阶段的shape', y1.shape, y2.shape, y3.shape, y4.shape) # [2,96,8,56,96] [2,192,8,28,48] [2,384,8,14,24] [2,768,8,7,12]--1.13日确认无误的

        #print('第一到第三阶段的shape', y1.shape, y2.shape, y3.shape, y4.shape)
        #print('第四阶段的输出, 即x的shape', x.shape)
        fusion_1 = self.fusion_1(haptic_1, y1)
        fusion_2 = self.fusion_2(haptic_2, y2)
        fusion_3 = self.fusion_3(haptic_3, y3)
        fusion_4 = self.fusion_4(haptic_4, x)
        # print('fusion_1.shape', fusion_1.shape)
        # print('fusion_2.shape', fusion_2.shape)
        # print('fusion_3.shape', fusion_3.shape)
        # print('fusion_4.shape', fusion_4.shape)
        out = self.decoder (fusion_4, fusion_3, fusion_2, fusion_1)
        return out
        #return  fusion_4, fusion_3, fusion_2, fusion_1

class Video_Audio_SaliencyModel(nn.Module):
    def __init__(self, pretrain=None):
        super(Video_Audio_SaliencyModel, self).__init__()

        self.backbone = SwinTransformer3D(pretrained=pretrain)
        self.fusion_1 = cross_attetion_fusion(96)
        self.fusion_2 = cross_attetion_fusion(192)
        self.fusion_3 = cross_attetion_fusion(384)
        self.fusion_4 = cross_attetion_fusion(768)

        self.decoder = Decoder_pyramid()

    def forward(self, visual, audio_1, audio_2, audio_3, audio_4):
        x, [y1, y2, y3, y4] = self.backbone(visual)
        print('第一到第三阶段的shape', y1.shape, y2.shape, y3.shape,
              y4.shape)  # [2,96,8,56,96] [2,192,8,28,48] [2,384,8,14,24] [2,768,8,7,12]--1.13日确认无误的

        # print('第一到第三阶段的shape', y1.shape, y2.shape, y3.shape, y4.shape)
        # print('第四阶段的输出, 即x的shape', x.shape)
        fusion_1 = self.fusion_1(audio_1, y1)
        fusion_2 = self.fusion_2(audio_2, y2)
        fusion_3 = self.fusion_3(audio_3, y3)
        fusion_4 = self.fusion_4(audio_4, x)
        # print('fusion_1.shape', fusion_1.shape)
        # print('fusion_2.shape', fusion_2.shape)
        # print('fusion_3.shape', fusion_3.shape)
        # print('fusion_4.shape', fusion_4.shape)
        out = self.decoder(fusion_4, fusion_3, fusion_2, fusion_1)

        return out



class Video_Audio_Haptic_SaliencyModel(nn.Module):
    def __init__(self, pretrain=None):
        super(Video_Audio_Haptic_SaliencyModel, self).__init__()
        '''三个模态的融合和解码生成网络'''
        self.backbone = SwinTransformer3D(pretrained=pretrain)
        '''视觉和听觉的融合'''
        self.fusionVA_1 = cross_attetion_fusion(96)
        self.fusionVA_2 = cross_attetion_fusion(192)
        self.fusionVA_3 = cross_attetion_fusion(384)
        self.fusionVA_4 = cross_attetion_fusion(768)
        '''视觉和触觉的融合'''
        self.fusionVH_1 = cross_attetion_fusion(96)
        self.fusionVH_2 = cross_attetion_fusion(192)
        self.fusionVH_3 = cross_attetion_fusion(384)
        self.fusionVH_4 = cross_attetion_fusion(768)

        '''将得到的两个特征融合'''
        self.cmf_fusion = FusionNetwork()

        '''金字塔的解码网络'''
        self.decoder = Decoder_pyramid()

    def forward(self, visual,  audio_1, audio_2, audio_3, audio_4, haptic_1, haptic_2, haptic_3, haptic_4):
        x, [y1, y2, y3, y4] = self.backbone(visual)
        print('第一到第三阶段的shape', y1.shape, y2.shape, y3.shape,
              y4.shape)  # [2,96,8,56,96] [2,192,8,28,48] [2,384,8,14,24] [2,768,8,7,12]--1.13日确认无误的

        # print('第一到第三阶段的shape', y1.shape, y2.shape, y3.shape, y4.shape)
        # print('第四阶段的输出, 即x的shape', x.shape)
        fusionVA_1 = self.fusionVA_1(audio_1, y1)
        fusionVA_2 = self.fusionVA_2(audio_2, y2)
        fusionVA_3 = self.fusionVA_3(audio_3, y3)
        fusionVA_4 = self.fusionVA_4(audio_4, x)


        fusionVH_1 = self.fusionVH_1(haptic_1, y1)
        fusionVH_2 = self.fusionVH_2(haptic_2, y2)
        fusionVH_3 = self.fusionVH_3(haptic_3, y3)
        fusionVH_4 = self.fusionVH_4(haptic_4, x)

        fusion_1, fusion_2, fusion_3, fusion_4 = self.cmf_fusion(fusionVH_1, fusionVH_2, fusionVH_3, fusionVH_4,fusionVA_1, fusionVA_2, fusionVA_3, fusionVA_4)
        out = self.decoder(fusion_4, fusion_3, fusion_2, fusion_1)

        return out










class PrintLayer(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x

''' 自己撰写的特征金字塔结构的上采样得到的结果  '''
class Decoder_pyramid(nn.Module):

    '''后面这部分可以修改，
    因为有的人的论文里面是直接使用卷积+上采样的方式paper33；
    有的是采用卷积+BN+relu+上采样的方式paper26；
    '''

    def __init__(self):
        super(Decoder_pyramid, self).__init__()
        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

        self.convtsp4_3 = nn.Sequential(
            nn.Conv3d(768, 384, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2)

        self.convtsp3_2 = nn.Sequential(
            nn.Conv3d(384, 192, kernel_size=(1, 3, 3), stride=(1,1,1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2)

        self.convtsp2_1 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2)

        # 这一层的输入是[10,96,8,56,96]
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            #PrintLayer(),  # 打印出来的shape是torch.Size([10, 48, 4, 56, 96])
            self.upsampling2,

            nn.Conv3d(
                48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            #PrintLayer(), # 打印出来的shape是torch.Size([10, 24, 2, 112, 192])
            self.upsampling2,

            #nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            #nn.ReLU(),

            nn.Conv3d(24, 12, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            #PrintLayer(), # [10,12,1,224,384]
            nn.Conv3d(12, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Sigmoid()
           # PrintLayer() #[10,1,1,224,384]
        )

    def forward(self, y4, y3, y2, y1):
        y4_ = self.convtsp4_3(y4)
        # print('y4.shape', y4.shape)
        # print('y3.shape', y3.shape)
        # print('y2.shape', y2.shape)
        # print('y1.shape', y1.shape)
        y3cat4 = torch.add(y4_, y3)
       #y3cat4 = torch.cat((y4_, y3), 1)
        y3_ = self.convtsp3_2(y3cat4)
        y2cat3 = torch.add(y3_, y2)
        #y2cat3 = torch.cat((y3_, y2), 1)
        y2_ = self.convtsp2_1(y2cat3)
        y1cat2 = torch.add(y2_, y1)
        #y1cat2 = torch.cat((y2_, y1), 1)
        #print('y1cat2.shape', y1cat2.shape)  # torch.Size([10,96,8,56,96])

        predict_out = self.convtsp1(y1cat2)

        #print('before last layer predict_out.shape', predict_out.shape)
        #predict_out = predict_out.view(predict_out.size(0), predict_out.size(3), predict_out.size(4))
        predict_out = predict_out.squeeze(1)
        print('after last layer predict_out.shape', predict_out.shape)
        return predict_out





class DecoderConvUp(nn.Module):
    def __init__(self):
        super(DecoderConvUp, self).__init__()

        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.upsampling4 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear')
        self.upsampling8 = nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear')

        self.conv1 = nn.Conv3d(96, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv2 = nn.Conv3d(192, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv3 = nn.Conv3d(384, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv4 = nn.Conv3d(768, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.convs1 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.convs2 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.convs3 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)

        self.convtsp1 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Sigmoid()
        )

        self.convtsp2 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Sigmoid()
        )

        self.convtsp3 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            self.upsampling2,
            nn.Sigmoid()
        )

        self.convtsp4 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            self.upsampling4,
            nn.Sigmoid()
        )

        self.convout = nn.Sequential(
            nn.Conv3d(4, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.Sigmoid()
        )

        self.gate1 = Gate(192)
        self.gate2 = Gate(192)
        self.gate3 = Gate(192)
        self.gate4 = Gate(192)

    def forward(self, y4, y3, y2, y1):

        '''执行时间减半，和特征通道的变换，特征的空间尺寸保持不变'''
        y1 = self.conv1(y1)
        y2 = self.conv2(y2)
        y3 = self.conv3(y3)
        y4 = self.conv4(y4)


        print('执行特征的减半和通道的变换,同时保持空间维度不变')
        print('y1.shape', y1.shape)  # torch.Size([10, 192, 8, 56, 56])
        print('y2.shape', y2.shape)  # torch.Size([10, 192, 8, 28, 28])
        print('y3.shape', y3.shape)  # torch.Size([10, 192, 8, 14, 14])
        print('y4.shape', y4.shape)  # torch.Size([10, 192, 8, 7, 7])


        t3 = self.upsampling2(y4) + y3
        y3 = self.convs3(t3)
        t2 = self.upsampling2(t3) + y2 + self.upsampling4(y4)
        y2 = self.convs2(t2)
        t1 = self.upsampling2(t2) + y1 + self.upsampling8(y4)
        y1 = self.convs1(t1)


        print('before 多维的注意力机制')
        print('y1.shape', y1.shape)  # torch.Size([10, 192, 8, 56, 56])
        print('y2.shape', y2.shape)  # torch.Size([10, 192, 8, 28, 28])
        print('y3.shape', y3.shape)  # torch.Size([10, 192, 8, 14, 14])
        print('y4.shape', y4.shape)  # torch.Size([10, 192, 8, 7, 7])

        print('after 多维的注意力机制')

        '''经过多维的通道注意力机制之后，模型的尺寸并不会受到影响，因为这个模块只是对通道进行了加权，而没有改变特征图的尺寸'''
        '''
        多维的空间--时间--通道注意力的shape为：跟他的输入的模型的权重是一致的
        [10, 192, 8, 56, 56]
        [10, 192, 8, 28, 28]
        [10, 192, 8, 14, 14]
        [10, 192, 8, 7, 7]
        '''

        y1 = self.gate1(y1)
        y2 = self.gate2(y2)
        y3 = self.gate3(y3)
        y4 = self.gate4(y4)

        print('y1.shape', y1.shape)  # torch.Size([10, 192, 8, 56, 56])
        print('y2.shape', y2.shape)  # torch.Size([10, 192, 8, 28, 28])
        print('y3.shape', y3.shape)  # torch.Size([10, 192, 8, 14, 14])
        print('y4.shape', y4.shape)  # torch.Size([10, 192, 8, 7, 7])


        '''执行特征的上采样'''
        z1 = self.convtsp1(y1)  # [10, 192, 8, 56, 56] -> [10, 1, 1, 224, 224]

        z2 = self.convtsp2(y2)  # [10, 192, 8, 28, 28] -> [10, 1, 1, 224, 224]

        z3 = self.convtsp3(y3)  # [10, 192, 8, 14, 14] -> [10, 1, 1, 224, 224]

        z4 = self.convtsp4(y4)  # [10, 192, 8, 7, 7] -> [10, 1, 1, 224, 224]

        print('z1.shape', z1.shape)  # torch.Size([10, 1, 1, 224, 224])
        print('z2.shape', z2.shape)  # torch.Size([10, 1, 1, 224, 224])
        print('z3.shape', z3.shape)  # torch.Size([10, 1, 1, 224, 224])
        print('z4.shape', z4.shape)  # torch.Size([10, 1, 1, 224, 224])

        z0 = self.convout(torch.cat((z1, z2, z3, z4), 1))
        print('z0.shape', z0.shape)  # torch.Size([10, 4, 1, 224, 224]
        z0 = z0.view(z0.size(0), z0.size(3), z0.size(4))
        print('z0.shape', z0.shape)  # torch.Size([10, 224, 224]
        return z0



'''融合网络的定义'''
class GlobalAvgPool3D(nn.Module):
    def __init__(self):
        super(GlobalAvgPool3D, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=(2, 3, 4), keepdim=True)

# Define the global max pooling layer
class GlobalMaxPool3D(nn.Module):
    def __init__(self):
        super(GlobalMaxPool3D, self).__init__()

    def forward(self, x):
        return torch.amax(x, dim=(2, 3, 4), keepdim=True)

# Define the channel-wise average pooling layer
class ChannelAvgPool3D(nn.Module):
    def __init__(self):
        super(ChannelAvgPool3D, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1, keepdim=True)

# Define the channel-wise max pooling layer
class ChannelMaxPool3D(nn.Module):
    def __init__(self):
        super(ChannelMaxPool3D, self).__init__()

    def forward(self, x):
        return torch.amax(x, dim=1, keepdim=True)

class cross_attetion_fusion(nn.Module):
    def __init__(self, in_channels):
        super(cross_attetion_fusion, self).__init__()

        '''基于通道的注意力'''
        #self.avg_pool_channel = GlobalAvgPool3D()
        #self.max_pool_channel = GlobalMaxPool3D()

        self.avg_pool_channel = nn.Sequential(
            GlobalAvgPool3D(),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(),
            nn.Conv3d(in_channels // 2, in_channels, kernel_size=1)
        )
        self.max_pool_channel = nn.Sequential(
            GlobalMaxPool3D(),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(),
            nn.Conv3d(in_channels // 2, in_channels, kernel_size=1)
        )

        # self.conv_bn_relu_2 = ConvBNReLU3D(4*in_channels, 2*in_channels) # 这个按照论文
        self.conv_bn_relu_2 = nn.Conv3d(2 * in_channels, in_channels, kernel_size=1)

        '''基于空间的注意力'''
        self.avg_pool_spatial = ChannelAvgPool3D()
        self.max_pool_saptial = ChannelMaxPool3D()
        self.sigmod = nn.Sigmoid()
        #self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False)
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False) # 尝试修改为3,1，没毛病也能运行

        self.conv_begin = nn.Conv3d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False)
        self.conv_end = nn.Conv3d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False)


    def forward(self, vis, haptic):

        '''基于通道的注意力'''
        # print('vis.shape', vis.shape)
        # print('haptic.shape', haptic.shape)
        fusion_1 = torch.cat((vis, haptic), 1)
        fusion_1 = self.conv_begin(fusion_1)

        max_fusion = self.max_pool_channel(fusion_1)
        avg_fusion = self.avg_pool_channel(fusion_1)

        fusion_2 = torch.cat((max_fusion, avg_fusion), 1)
        fusion_3_channel = self.conv_bn_relu_2(fusion_2)
        # print('基于通道的注意力fusion_3_channel.shape', fusion_3_channel.shape) # [2,96,1,1,1]; [2,192,1,1,1]; [2,384,1,1,1]; [2,768,1,1,1]
        # print('B,1,C,1,1')

        '''基于空间的注意力'''
        #fusion_1 = torch.cat((vis, haptic), 1)
        max_fusion_spatial = self.max_pool_saptial(fusion_1)
        avg_fusion_spatial = self.avg_pool_spatial(fusion_1)
        fusion_2_spatial = torch.cat((max_fusion_spatial, avg_fusion_spatial), 1)
        fusion_3_spatial = self.conv(fusion_2_spatial)
        # print('基于空间-时间的注意力fusion_3_spatial.shape', fusion_3_spatial.shape) # [2,1,8,56,96];   [2,1,8,28,48]; [2,1,8,14,24]; [2,1,8,7,12]
        # print('B,T,1,H,W')

        output_logits = self.sigmod(fusion_3_spatial*fusion_3_channel)
        #print('output_logits的形状', output_logits.shape)

        fusion = torch.cat([output_logits*vis, (1-output_logits)*haptic], 1)
        fusion_out = self.conv_end(fusion)

        return fusion_out
#
# class haptic_process:()
#     def __init__(self):


'''音频编码网络'''
class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1),
                               padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1),
                               padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1),
                               padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1),
                               padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)

        self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1),
                                    stride=(2, 1))
        self.conv8_scns = nn.Conv2d(1024, 401, kernel_size=(8, 1),
                                    stride=(2, 1))

    def forward(self, waveform):
        x = self.conv1(waveform)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)

        return x


import torch
from torch import nn




if __name__ == '__main__':

    '''测试音频编码网络'''
    a = torch.rand(2,1,64,1200)
    model = SoundNet()
    out = model(a)
    print(out.shape)  # torch.Size([2, 1024, 1, 75])
    exit()


    '''测试音视触的融合网络'''
    v = torch.rand(2, 3, 16, 224, 384)
    a1 = torch.rand(2, 96, 8, 56, 96)
    a2 = torch.rand(2, 192, 8, 28, 48)
    a3= torch.rand(2, 384, 8, 14, 24)
    a4 = torch.rand(2, 768, 8, 7, 12)

    h1 = torch.rand(2, 96, 8, 56, 96)
    h2 = torch.rand(2, 192, 8, 28, 48)
    h3 = torch.rand(2, 384, 8, 14, 24)
    h4 = torch.rand(2, 768, 8, 7, 12)

    model = Video_Audio_Haptic_SaliencyModel()
    out = model(v, a1, a2, a3, a4, h1, h2, h3, h4)
    print(out.shape)

    h1 = torch.rand(10, 96, 8, 56, 96)
    h2 = torch.rand(10, 192, 8, 28, 48)
    h3 = torch.rand(10, 384, 8, 14, 24)
    h4 = torch.rand(10, 768, 8, 7, 12)

    a1 = torch.rand(10, 96, 8, 56, 96)
    a2 = torch.rand(10, 192, 8, 28, 48)
    a3 = torch.rand(10, 384, 8, 14, 24)
    a4 = torch.rand(10, 768, 8, 7, 12)
    model = FusionNetwork()
    y1, y2, y3, y4 = model(h1, h2, h3, h4, a1, a2, a3, a4)

    print(y1.shape)  # torch.Size([10, 96, 8, 56, 96])
    print(y2.shape)  # torch.Size([10, 192, 8, 28, 48])
    print(y3.shape)  # torch.Size([10, 384, 8, 14, 24])
    print(y4.shape)  # torch.Size([10, 768, 8, 7, 12])

    exit()
    y1 = torch.rand(10, 96, 8, 56, 96)
    y2 = torch.rand(10, 192, 8, 28, 48)
    y3 = torch.rand(10, 384, 8, 14, 24)
    y4 = torch.rand(10, 768, 8, 7, 12)

    decoder = Decoder_pyramid()

    out = decoder(y4, y3, y2, y1)
    print(out.shape)
    exit()

    input_tensor = torch.randn(2, 3, 16, 64, 64)
    model = haptic_affine()
    output1, output2, output3, output4 = model(input_tensor)

    print(output1.shape)  # Expected: [1, 96, 8, 56, 96]
    print(output2.shape)  # Expected: [1, 96, 8, 24, 48]
    print(output3.shape)  # Expected: [1, 96, 8, 12, 24]
    print(output4.shape)  # Expected: [1, 96, 8, 6, 12]
    exit()


    y1 = torch.rand(10, 96, 8, 56, 56)
    y1_haptic = torch.rand(10, 96, 8, 56, 56)
    fusion_model = cross_attetion_fusion(96)
    out = fusion_model(y1, y1_haptic)
    print("out.shape", out.shape)

    exit()

    '''测试原来的decoder模型'''
    y1 = torch.rand(10, 96, 8, 56, 56)
    y2 = torch.rand(10, 192, 8, 28, 28)
    y3 = torch.rand(10, 384, 8, 14, 14)
    y4 = torch.rand(10, 768, 8, 7, 7)
    dec_model = DecoderConvUp()
    out = dec_model(y4, y3, y2, y1)
    print('out.shape', out.shape)
    exit()


    dec_model = Decoder_pyramid()
    out = dec_model(y4, y3, y2, y1)
    print('out.shape', out.shape)

    dummy_x = torch.rand(5, 3, 32, 160, 320)
    video_encoder = SwinTransformer3D(pretrained=None)
    x, [y1, y2, y3, y4] = video_encoder(dummy_x) # 4,3,2,1;[10, 768, 16, 7, 7],[10, 384, 16, 14, 14],[10, 192, 16, 28, 28],[10, 96, 16, 56, 56]


    '''触觉信号的编码特征'''
    random_tensor_y1 = torch.rand_like(y1)
    random_tensor_y2 = torch.rand_like(y2)
    random_tensor_y3 = torch.rand_like(y3)
    random_tensor_4 = torch.rand_like(x)
    model_gen_saliecny = Video_HAPTIC_SaliencyModel()
    print('random_tensor_y1.shape', random_tensor_y1.shape)
    print('random_tensor_y2.shape', random_tensor_y2.shape)
    print('random_tensor_y3.shape', random_tensor_y3.shape)
    print('random_tensor_4.shape', random_tensor_4.shape)
    out = model_gen_saliecny(dummy_x, random_tensor_y1, random_tensor_y2, random_tensor_y3, random_tensor_4)


    print('out.shape', out.shape)

    #dec_model = Decoder_pyramid()
    #out = dec_model(x, y3, y2, y1)





# exit()
#
# '''旧的main函数'''
# if __name__ == '__main__':
#     model = VideoSaliencyModel()
#     dummy_x = torch.rand(10, 3, 32, 224, 224)
#     logits = model(dummy_x)
#     print('ssss')
#     print(logits.shape)