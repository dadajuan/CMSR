import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
#from utils.process_video_audio import LoadVideoAudio_TRAIN
from utils.process_video_audio_new import LoadVideoAudio_TRAIN

from model_train import DAVE

from model_my import Video_HAPTIC_SaliencyModel, AffineTransform, TimeSeriesTransformer, Video_Audio_SaliencyModel, AffineTransform_Audio, Encoder_Auido_sound, Video_Audio_Haptic_SaliencyModel,Encoder_Auido_sound_2
import pdb
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cv2
from loss_function_my import kldiv, cc_score, nss_score


# Inside my model training code
# import wandb
# wandb.init(project="my-project")


# the folder find the videos consisting of video frames and the corredponding audio wav
#VIDEO_TRAIN_FOLDER = './data_ICME20/'
# 视频训练文件夹路径
#VIDEO_TRAIN_FOLDER = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\video\\train_14'

#VIDEO_TRAIN_FOLDER = r'F:\AVS360\new\AVS360\data\video\train_14'


'''这个决定了训练集的数据是什么'''
#VIDEO_TRAIN_FOLDER = r'/data1/liuhengfa/my_own_code_for_saliency/data/video0_renamed/train_14'
VIDEO_TRAIN_FOLDER = r'/data1/liuhengfa/my_own_code_for_saliency/data/video0_renamed/train_14'

# 这个是保存模型的路径，后面的main_wanndb也要跟着改变的,即最后加载模型的部分；
OUTPUT = r'/data1/liuhengfa/my_own_code_for_saliency/test_for_tuning_old_dataset_v_a_h/save_model_dataset/test_14'



# where tofind the model weights
# 模型路径
# MODEL_PATH = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\model.pth.tar'

# some config parameters

IMG_WIDTH = 256
IMG_HIGHT = 320
TRG_WIDTH = 32
TRG_HIGHT = 40

device = torch.device("cuda:3")

# 损失函数
loss_function = nn.KLDivLoss()
loss_function_bce = nn.BCELoss()


viusal_encoder_path = '/data1/liuhengfa/my_own_code_for_saliency/AVS360_audiovisual_saliency_360-master_new/xiaozhou_new/swin_small_patch244_window877_kinetics400_1k.pth'
audio_sound_path = '/data1/liuhengfa/my_own_code_for_saliency/test_for_tuning_old_dataset_v_a_h/AVS360_audiovisual_saliency_360-master_new/xiaozhou_new/swin_small_patch244_window877_kinetics400_1k.pth'



w1 = 1
w2 = 0.1
w3 = 0.1



def load_model_parameters(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
class TrainSaliency(object):

    def __init__(self):
        super(TrainSaliency, self).__init__()

        # 获取视频列表
        self.video_list = [os.path.join(VIDEO_TRAIN_FOLDER, p) for p in os.listdir(VIDEO_TRAIN_FOLDER)]
        self.video_list = self.video_list
        self.video_list.sort()
        # pdb.set_trace()

        # 加载模型
        #self.model = DAVE()
        # self.model_2 = Video_HAPTIC_SaliencyModel(pretrain=viusal_encoder_path)
        # self.model_2 = self.model_2.to(device)

        # self.model_2 = Video_Audio_SaliencyModel(pretrain=viusal_encoder_path)
        # self.model_2 = self.model_2.to(device)

        self.model_2 = Video_Audio_Haptic_SaliencyModel(pretrain=viusal_encoder_path)
        self.model_2 = self.model_2.to(device)

        '''触觉编码网络的两种选择，一种是transformer编码时序模型，一种是resnet仅编码最后一帧'''

        '''当这个修改的时候，
             （1） 下面加载模型的也要修改；233,234行，即触觉的编码，同时注意，其输入也不同；
             （2） predict的函数也是要修改的
             (3)  init_部分的仿射函数即Affinetransform也是需要修改的，一个是128，一个是256 即113行114行附近
             （4） 关于优化器也需要修改，即触觉编码器是否参与训练
             '''

        # 方案1
        self.haptic_encoder = TimeSeriesTransformer()
        self.haptic_encoder = self.haptic_encoder.to(device)

        self.audio_encoder = Encoder_Auido_sound_2(pretrain_path=audio_sound_path)
        self.audio_encoder = self.audio_encoder.to(device)


        # 方案2

        # haptic_encoder_resnet = haptic_encoder_fenlei(3200)
        # self.haptic_encoder_resnet = load_model_parameters(haptic_encoder_resnet, '/data1/liuhengfa/serach_haptic/encoder_haptic3_resnet.pth' )
        # self.haptic_encoder = self.haptic_encoder_resnet.to(device)

        self.model_haptic_affine = AffineTransform(input_dim=128)  # 使用resnet分类网络的时候用的256；使用transform的是128
        self.model_haptic_affine = self.model_haptic_affine.to(device)

        self.model_audio_affine = AffineTransform_Audio(input_dim=1024)  # 使用resnet分类网络的时候用的256；使用transform的是128
        self.model_audio_affine = self.model_audio_affine.to(device)

        #self.model = self.model.to(device)
        # self.model = self.model.cuda()
        # # 加载模型权重
        # self.model.load_state_dict(self._load_state_dict_(MODEL_PATH), strict=True)


        # 输出路径
        self.output = OUTPUT
        if not os.path.exists(self.output):
                os.mkdir(self.output)

        # 加载赤道偏置图像并进行预处理
        #equator_bias = cv2.resize(cv2.imread('ECB.png', 0), (10, 8))
        equator_bias = cv2.resize(cv2.imread(r'/data1/liuhengfa/my_own_code_for_saliency/ECB.png', 0), (10, 8))
        self.equator_bias = torch.tensor(equator_bias).to(device, dtype=torch.float)
        self.equator_bias = self.equator_bias.cuda()
        self.equator_bias = self.equator_bias / self.equator_bias.max()

        # 定义优化器

        '''下面两行确定触觉编码器、视觉编码器是否参与训练'''
        # self.optimizer = optim.Adam(
        #     list(self.model_2.parameters()) + list(self.model_haptic_affine.parameters())+ list(self.haptic_encoder.parameters()),
        #     lr=5*1e-5
        # )
        self.optimizer = optim.Adam(
            list(self.model_2.parameters()) + list(self.model_audio_affine.parameters()) + list(
                self.audio_encoder.parameters()) +list(self.model_haptic_affine.parameters())+ list(self.haptic_encoder.parameters()),
            lr=5 * 1e-5
        )
        v_num = len(self.video_list)


        #self.model.eval()
    
    @staticmethod
    def _load_state_dict_(filepath):
        if os.path.isfile(filepath):
            print("=> loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath, map_location=device)

            pattern = re.compile(r'module+\.*')
            state_dict = checkpoint['state_dict']

            #new_state_dict = {k : v for k, v in state_dict.items() if 'video_branch' in k}

            # 修改state_dict中的键
            for key in list(state_dict.keys()):
                if 'video_branch' in key:
                   state_dict[key[:12] + '_cubic' + key[12:]] = state_dict[key]#video_branch.conv1.weight -> Key: video_branch_cubic.conv1.weight

                if 'combinedEmbedding' in key:
                    state_dict[key[:17] + '_equi_cp' + key[17:]] = state_dict[key]#combinedEmbedding.weight -> combinedEmbedding_equi_cp.weight

            #pdb.set_trace()
            # 删除state_dict中的键
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    print('Y', key)
                    new_key = re.sub('module.', '', key)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]


        return state_dict



    def train(self, epoch):


        self.model_2.train()

        self.haptic_encoder.train()
        self.model_haptic_affine.train()

        self.audio_encoder.train()
        self.model_audio_affine.train()

        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_cc_loss = 0.0
        epoch_nss_loss = 0.0

        batch_count = 0.0
        start = time.time()
        for n, v in enumerate(self.video_list[:]):  # 逐个视频进行训练一，即先进行第一个；然后再进行第二个
            fps = 29.970030
            stimuli_path = v
            video_loader = LoadVideoAudio_TRAIN(stimuli_path, fps)
            #print("stimuli_path:", stimuli_path)
            vit = iter(video_loader)
            start = time.time()
            print('#############开始加载一个新的视频##############')
            for idx in range(len(video_loader)):  # 对第一个视频里面的第一帧；然后是第二帧，直至达到该视频中的总帧长
                #print('len(video_loader):', len(video_loader))
                video_data_equi, video_data_cube, audio_data, haptic_data, gt_salmap, fixation = next(vit)

                print('video_data_equi:', video_data_equi.shape)
                #print('video_data_cube:', video_data_cube.shape)
                print('audio_data:', audio_data.shape)
                print('haptic_data:', haptic_data.shape)

                # 将数据转换为tensor并放入GPU
                video_data_equi = video_data_equi.to(device=device, dtype=torch.float)
                #video_data_equi = video_data_equi.cuda()
                print('video_data_equi:', video_data_equi.shape)  # ([1, 3, 16, 320, 640]) 输入的是16个视频帧，输出的是一个32*40的显著图

                video_data_cube = video_data_cube.to(device=device, dtype=torch.float)
                #video_data_cube = video_data_cube.cuda()

                '''触觉信号直接生成通过仿射变化，来实现与视觉信号的尺寸匹配，首先加载触觉信号，
                然后通过特征的编码；
                最后执行仿射变换；
                '''
                audio_data = audio_data.to(torch.float32)  # 这里就是听觉信号
                audio_data = audio_data.to(device=device, dtype=torch.float)
                #audio_data = audio_data.cuda()
                #print('audio_data:', audio_data.shape)

                haptic_data = haptic_data.to(torch.float32)  # 这里是加载的txt的触觉信号了，只是名字没改格式是[b, T, 3200]
                haptic_data = haptic_data.to(device=device, dtype=torch.float)
                print('haptic_data:', haptic_data.shape)

                '''编码音频特征'''
                audio_feature = self.audio_encoder(audio_data)
                print('audio_feature的维度', audio_feature.shape)  # ([2, 1024] 这里是加载的txt的触觉信号了，只是名字没改格式是[b, T, 3200]，即1024*2
                audio_out1, audio_out2, audio_out3, audio_out4 = self.model_audio_affine(audio_feature)
                print('xxx-仿射变换后音频的维度-xxx', audio_out1.shape, audio_out2.shape, audio_out3.shape, audio_out4.shape)

                '''编码触觉特征'''
                # 方案1
                haptic_data = self.haptic_encoder(haptic_data)  # 使用trasnformer的方法

               #print('xxxxx---编码的触觉的特征的维度--xxxxx', haptic_data.shape)  # ([2, 256])
                haptic_out1, haptic_out2, haptic_out3, haptic_out4 = self.model_haptic_affine(haptic_data)
                print('xxx-仿射变换后触觉的维度-xxx', haptic_out1.shape, haptic_out2.shape, haptic_out3.shape, haptic_out4.shape)


                gt_salmap = gt_salmap.to(device=device, dtype=torch.float)
                #gt_salmap = gt_salmap.cuda()
                #print('gt_salmap:', gt_salmap.shape)  # ([1, 1,320, 640]) 输入的是16个视频帧，输出的是一个32*40的显著图

                fixation_map = fixation.to(device=device, dtype=torch.float)
                #fixation_map = fixation_map.cuda()
                #print('fixation_map:', fixation_map.shape)  # ([1, 1, 320, 640]) 输入的是16个视频帧，输出的是一个32*40的显著图
                #print('gt_salmap最大值和最小值:', torch.max(gt_salmap).item(), torch.min(gt_salmap).item())
                #print('fixation_map最大值和最小值:', torch.max(fixation_map).item(), torch.min(fixation_map).item())

                # 预测显著图
                #pred_salmap = self.model_2(video_data_equi, random_tensor_y1, random_tensor_y2, random_tensor_y3, random_tensor_4)
                #pred_salmap = self.model_2(video_data_equi, haptic_out1, haptic_out2, haptic_out3, haptic_out4)
                pred_salmap = self.model_2(video_data_equi, audio_out1, audio_out2, audio_out3, audio_out4, haptic_out1, haptic_out2, haptic_out3, haptic_out4)
                print('这个是联合视+音+触觉生成的显著性图')
                #exit()
                #pred_salmap = self.model(video_data_equi, video_data_cube, audio_data, self.equator_bias)

                #print('pred_salmap:', pred_salmap.shape)

                # print('gt_salmap:', gt_salmap.shape)
                # print('pred_salmap最大值和最小值:', torch.max(pred_salmap).item(), torch.min(pred_salmap).item())

                # 计算损失
                #loss = loss_function_bce(pred_salmap, gt_salmap)
                #print('loss:', loss)

                loss_kl = kldiv(pred_salmap, gt_salmap)  # 第一个参数是预测的显著图，第二个参数是真实的显著图

                print('loss_kl:', loss_kl)

                loss_cc = cc_score(pred_salmap, gt_salmap, 1)  # 计算显著图的CC分数
                print('loss_cc:',  loss_cc)

                loss_nss = nss_score(pred_salmap, fixation_map,1)  # 计算显著图的NSS分数
                print('loss_nss:', loss_nss)

                loss_overall = w1 * loss_kl + w2 * loss_cc + w3 * loss_nss
                print('loss_overall:', loss_overall)
                # 累加损失
                epoch_loss += loss_overall.cpu().data.numpy()
                self.optimizer.zero_grad()
                #loss.backward()
                loss_overall.backward()
                # 更新参数
                self.optimizer.step()
                '''把其他的也打印出来'''
                epoch_kl_loss += loss_kl.cpu().data.numpy()
                epoch_cc_loss += loss_cc.cpu().data.numpy()
                epoch_nss_loss += loss_nss.cpu().data.numpy()


            # 统计了一共有多少帧图片
            #print('len(video_loader):', len(video_loader))
            batch_count = batch_count + len(video_loader)
            print('batch_count:', batch_count)
            end = time.time()

            print(
                "=== Epoch {%s}  Loss: {%.8f}  Running time: {%4f}" % (
                str(epoch), (epoch_loss) / batch_count, end - start))

            if epoch % 50 == 0:
                torch.save(self.model_2, OUTPUT + '_my_model_ep' + str(epoch) + '.pkl')  # 这个是视频编码模型,融合模型，以及解码模型
                torch.save(self.audio_encoder, OUTPUT + '_my_audio_encoder_model_ep' + str(epoch) + '.pkl')
                torch.save(self.model_audio_affine, OUTPUT + '_my_audio_affine_model_ep' + str(epoch) + '.pkl')

                torch.save(self.haptic_encoder, OUTPUT + '_my_haptic_encoder_model_ep' + str(epoch) + '.pkl')
                torch.save(self.model_haptic_affine, OUTPUT + '_my_haptic_affine_model_ep' + str(epoch) + '.pkl')
                print('保存模型成功')
        #return (epoch_loss) / batch_count                 # 返回平均损失,即所有视频的总损失/所有视频中的总帧数

        return (epoch_loss) / batch_count,   (epoch_kl_loss)/ batch_count, (epoch_cc_loss)/batch_count, (epoch_nss_loss)/batch_count   # 返回平均损失,即所有视频的总损失/所有视频中的总帧数



if __name__ == '__main__':

    t = TrainSaliency()
    t.train()
  
