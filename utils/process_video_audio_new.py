
# generic must imports
import os
import torch
import numpy as np
import cv2
import re

import utils.audio_params as audio_params
import librosa as sf
from utils.audio_features import waveform_to_feature
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms.functional as F
from utils.equi_to_cube import Equi2Cube
import pdb
import random
from model_my import TimeSeriesTransformer
#from utils.haptic_encoder_lhf import get_haptic



'''尝试由320*640变成160*320'''
__all__ = ['LoadVideoAudio']

#defined params @TODO move them to a parameter config file
DEPTH = 16
GT_WIDTH = 320
GT_HIGHT = 640
haptic_feature_dim = 128

e2c = Equi2Cube(128, 256, 512)     # Equi2Cube(out_w, in_h, in_w) 

MEAN = [ 110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0 ]#MEAN = [0.43387713, 0.40455043, 0.37721659]
STD = [ 38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0 ]#STD = [0.15100728, 0.14855877, 0.15697562]

def adjust_len(a, b):
    # adjusts the len of two sorted lists
    al = len(a)
    bl = len(b)
    if al > bl:
        start = (al - bl) // 2
        end = bl + start
        a = a[start:end]
    if bl > al:
        a, b = adjust_len(b, a)
    return a, b


def create_data_packet(in_data, frame_number):

    # 处理输入的音频或视频数据，生成一个固定帧数的数据包
    # 输入的是特征和帧数，输出的是数据包和有效帧数

    n_frame = in_data.shape[0]#n_frame的值是输入数据in_data的第一个维度的大小：帧的数量

    # print("n_frame:", n_frame)
    '''看看这个创建函数的是干嘛的'''
    #print("输入的创建数据包的函数的形状是什么in_data shape:", in_data.shape)  # [23, 64, 64],这里的23是指视频的帧数

    # print(f"create_data_packet 函数中原始 frame_number: {frame_number}")
    frame_number = min(frame_number, n_frame)
    # print(f"create_data_packet 函数中修改后 frame_number: {frame_number}")
    starting_frame = frame_number - DEPTH + 1
    starting_frame = max(0, starting_frame) #确保我们没有任何负帧
    data_pack = in_data[starting_frame:frame_number+1, :, :]
    n_pack = data_pack.shape[0]


    if n_pack < DEPTH:
        nsh = DEPTH - n_pack
        data_pack = np.concatenate((np.tile(data_pack[0,:,:], (nsh, 1, 1)), data_pack), axis=0)

    assert data_pack.shape[0] == DEPTH
    # 这行代码 data_pack = np.tile(data_pack, (3, 1, 1, 1)) 使用 np.tile 函数沿第一个维度（通道）复制 data_pack 数组。
    # 这实际上将单通道数据数组转换为三通道数据数组，通过沿第一个维度重复原始数据三次来实现。
    #print('在没有经过复制之前的data_pack:', data_pack.shape)  # [16,64,64]
    data_pack = np.tile(data_pack, (3, 1, 1, 1))
    #print("输出的创建数据包的函数的形状是什么data_pack shape:", data_pack.shape)  # [3, 16, 64, 64]

    return data_pack, frame_number


def load_wavfile(total_frame, wav_file):
    """load a wave file and retirieve the buffer ending to a given frame
       加载 WAV 文件并提取到给定帧的音频数据。
       Args:
         wav_file: String path to a file, or a file-like object. The file
         is assumed to contain WAV audio data with signed 16-bit PCM samples.
         文件或类文件对象的字符串路径。该文件假设包含16位PCM样本的WAV音频数据。
       参数：
         total_frame：所需的音频帧的总数。
         wav_file：WAV 音频数据文件的路径或文件对象。
         frame_number: Is the frame to be extracted as the final frame in the buffer
         frame_number：是要提取的作为缓冲区中的最终帧
       Returns:
         See waveform_to_feature.
       """
    # 使用指定的采样率和数据类型加载wav文件
    # wav_data, sr = sf.load(wav_file, sr=audio_params.SAMPLE_RATE, dtype='float32')
    # assert sf.get_duration(wav_data, sr) > 1
    #print('-----打印音频文件的地址----wav_file:', wav_file)
    wav_data, sr = sf.load(path=wav_file, sr=audio_params.SAMPLE_RATE, dtype='float32')
    assert sf.get_duration(y=wav_data, sr=sr) > 1

    #将波形数据转换成对数梅尔谱图特征
    features = waveform_to_feature(wav_data, sr)

    #print("-----打印原本波形图生成的特征的形状----features shape:", features.shape)  # [1182, 64, 64]
    features = np.resize(features, (int(total_frame), features.shape[1], features.shape[2]))
    #print("features shape:", features.shape)  # [23, 64, 64]
    # # 将音频数据设置为全零
    # features = np.zeros_like(features)

    return features


def get_wavFeature(features, frame_number):
    
    audio_data, valid_frame_number = create_data_packet(features, frame_number)
    return torch.from_numpy(audio_data).float(), valid_frame_number


def load_maps(file_path):
    '''
        Load the gt maps
    :param file_path: 地图文件的路径
    :return: 作为浮点数的numpy数组
    '''


    with open(file_path, 'rb') as f:
        with Image.open(f) as img:
            #将图像转换为灰度图（'L'模式）并调整大小
            img = img.convert('L').resize((GT_HIGHT, GT_WIDTH), resample=Image.BICUBIC)
            data = F.to_tensor(img)#将图像转化为PyTorch张量
    return data


def load_video_frames(end_frame, frame_number, valid_frame_number):
    frame_path, frame_name = os.path.split(end_frame)  # 获取帧的路径和名称
    assert int(frame_name[4:9]) == frame_number  # 确保帧名称中的数字部分与给定的帧号一致

    frame_number = min(frame_number, valid_frame_number)  # 确保帧号不超过有效帧号
    start_frame_number = frame_number - DEPTH + 1  # 计算起始帧号
    start_frame_number = max(1, start_frame_number)  # 确保起始帧号不小于1
    frame_list = [f for f in range(start_frame_number, frame_number + 1)]  # 生成从起始帧号到当前帧号的帧列表
    #print(f"frame_list:{frame_list}")

    if len(frame_list) < DEPTH:  # 如果帧列表长度小于深度，进行填充
        nsh = DEPTH - len(frame_list)  # 计算需要填充的帧数
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)  # 填充帧列表
        #print(f"如果不够,需要补充的话,frame_list:{frame_list}")
    frames_cube = []  # 初始化立方体帧列表
    frames_equi = []  # 初始化等距帧列表

    for i in range(len(frame_list)):  # 遍历帧列表，处理每一帧
        imgpath = os.path.join(frame_path, 'img_{0:05d}.{1:s}'.format(frame_list[i], frame_name[-3:]))  # 构建图像路径
        #print(f"等距投影的图像的imgpath:{imgpath}")
        #print('zzzzzz')
        with open(imgpath, 'rb') as f:  # 打开图像文件
            with Image.open(f) as img:  # 使用PIL库打开图像
                img = cv2.resize(cv2.imread(imgpath), (512, 256))  # 读取图像并调整大小
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # 将图像从BGR转换为RGB并归一化

                img_c = e2c.to_cube(img)  # 将等距图像转换为立方体图像
                img_cube = []  # 初始化立方体图像列表
                for face in range(6):  # 遍历立方体的6个面
                    img_f = F.to_tensor(img_c[face])  # 将图像转换为张量
                    img_f = F.normalize(img_f, MEAN, STD)  # 对图像进行归一化
                    img_cube.append(img_f)  # 将处理后的图像添加到立方体图像列表中
                img_cube_data = torch.stack(img_cube)  # 将立方体图像列表堆叠成张量
                frames_cube.append(img_cube_data)  # 将立方体图像张量添加到立方体帧列表中


                #img = cv2.resize(img, (320, 160))  # 调整等距图像大小
                img = cv2.resize(cv2.imread(imgpath), (384, 224))  # 调整等距图像大小
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                img_equi = F.to_tensor(img)  # 将等距图像转换为张量
                img_equi = F.normalize(img_equi, MEAN, STD)  # 对等距图像进行归一化
                frames_equi.append(img_equi)  # 将处理后的等距图像添加到等距帧列表中

    data_cube = torch.stack(frames_cube, dim=0)  # 将立方体帧列表堆叠成张量
    data_equi = torch.stack(frames_equi, dim=0)  # 将等距帧列表堆叠成张量

    return data_equi.permute([1, 0, 2, 3]), data_cube.permute([1, 2, 0, 3, 4])  # 返回等距图像和立方体图像的张量


def load_AEM_frames(end_frame, frame_number, valid_frame_number):
    # load video frames, process them and return a suitable tensor
    frame_path, frame_name = os.path.split(end_frame)
    #pdb.set_trace()
    assert int(frame_name[0:-4]) == frame_number
    frame_number = min(frame_number, valid_frame_number)
    start_frame_number = frame_number - DEPTH+1
    start_frame_number = max(0, start_frame_number)
    frame_list = [f for f in range(start_frame_number, frame_number+1)]
    if len(frame_list) < DEPTH:
        nsh = DEPTH - len(frame_list)
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)
    frames = np.zeros((8, 10))
    count = 0.0

    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, '{0:04d}.{1:s}'.format(frame_list[i], frame_name[-3:]))    
        try:
            img = cv2.resize(cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE), (10, 8))
            img = img/255.0
            frames = frames + img
            count = count + 1
        except:
            continue
                    
    frames = frames/count
    if frames.sum()>0:
        frames = frames/frames.max()
    frames = F.to_tensor(frames)
    #pdb.set_trace()
    #data = torch.stack(frames, dim=0)
    return frames


def load_txt_file(file_path):
    """
    读取txt文件，将每行数据存入NumPy数组
    """
    with open(file_path, 'r') as f:
        data = f.readlines()
    data = [float(line.strip()) for line in data]
    data = np.array(data)
    num_frames = len(data) // 3200
    frames = data.reshape(num_frames, 3200)  # 每帧3200点

    return frames

def get_haptic(in_data, frame_number):

    '''该函数负责把输入的完整触觉数据，从中截去有效帧数的触觉数据'''
    n_frame = in_data.shape[0]#n_frame的值是输入数据in_data的第一个维度的大小：帧的数量
    # print("n_frame:", n_frame)
    '''看看这个创建函数的是干嘛的'''
    #print("输入的创建数据包的函数的触觉形状是什么in_data shape:", in_data.shape)  # [23, 64, 64],这里的23是指视频的帧数

    # print(f"create_data_packet 函数中原始 frame_number: {frame_number}")
    frame_number = min(frame_number, n_frame)
    # print(f"create_data_packet 函数中修改后 frame_number: {frame_number}")
    starting_frame = frame_number - DEPTH + 1
    starting_frame = max(0, starting_frame) #确保我们没有任何负帧
    data_pack = in_data[starting_frame:frame_number+1, :]
    n_pack = data_pack.shape[0]

    if n_pack < DEPTH:
        nsh = DEPTH - n_pack
        data_pack = np.concatenate((np.tile(data_pack[0,:], (nsh, 1)), data_pack), axis=0)

    assert data_pack.shape[0] == DEPTH
    # 这行代码 data_pack = np.tile(data_pack, (3, 1, 1, 1)) 使用 np.tile 函数沿第一个维度（通道）复制 data_pack 数组。
    # 这实际上将单通道数据数组转换为三通道数据数组，通过沿第一个维度重复原始数据三次来实现。
    #print('在没有经过复制之前的data_pack:', data_pack.shape)  # [16,64,64]
    #data_pack = np.tile(data_pack, (3, 1, 1, 1))
    #print("输出的创建数据包的函数的触觉的形状是什么data_pack shape:", data_pack.shape, type(data_pack))  # [3, 16, 64, 64]

    return data_pack, frame_number



def load_gt_frames(end_frame, frame_number, valid_frame_number):
    frame_path, frame_name = os.path.split(end_frame)  # 获取帧的路径和名称
    assert int(frame_name[7:12]) == frame_number  # 确保帧名称中的数字部分与给定的帧号一致

    # print('frame_number:', frame_number)
    # print("frame_name", frame_name)

    frame_number = min(frame_number, valid_frame_number)  # 确保帧号不超过有效帧号
    start_frame_number = frame_number - DEPTH + 1  # 计算起始帧号
    start_frame_number = max(1, start_frame_number)  # 确保起始帧号不小于0
    frame_list = [f for f in range(start_frame_number, frame_number + 1)]  # 生成从起始帧号到当前帧号的帧列表

    if len(frame_list) < DEPTH:  # 如果帧列表长度小于深度，进行填充
        nsh = DEPTH - len(frame_list)  # 计算需要填充的帧数
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)  # 填充帧列表

    frames = np.zeros((224, 384))  # 初始化帧数组
    count = 0.0  # 初始化计数器

    for i in range(len(frame_list)):  # 遍历帧列表，处理每一帧
        imgpath = os.path.join(frame_path, 'eyeMap_{0:05d}.{1:s}'.format(frame_list[i], frame_name[-3:]))  # 构建图像路径

        if i == len(frame_list) - 1:
            img_new = cv2.resize(cv2.imread(imgpath), (384, 224))  # 读取图像并调整大小
            img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB) / 255.0  # 将图像从BGR转换为RGB
            print('我实际上是最后一帧的saliency的imgpath:', imgpath)
            frame2 = F.to_tensor(img_new)  # 将图像转换为张量
            frame2 = frame2.mean(dim=0, keepdim=True)
        #print('真实的saliency的imgpath:', imgpath)
        try:
            #img = cv2.resize(cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE), (384, 224))  # 读取图像并调整大小
            img = cv2.resize(cv2.imread(imgpath), (384, 224))  # 读取图像并调整大小
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0  # 将图像从BGR转换为RGB
            #print('img_shape', img.shape)
            frames = frames + img  # 累加图像
            #print('frames_shape', frames.shape)
            count = count + 1  # 增加计数器
        except:
            continue  # 如果读取图像失败，跳过该帧
    # 打印frame的尺寸
    #print('frame的尺寸', frames.shape)
    frames = frames / count  # 计算平均图像
    frames = frames / frames.sum()  # 归一化平均图像
    frames = F.to_tensor(frames)  # 将图像转换为张量

    #print('原始方案的frames_shape:', frames.shape)
    #print('saliency的对比')这个其实是没问题的
    #print('fixation_map:', frames.shape, frames.max(), frames.min())
    frames = frame2

    #print('新设计的只加载最后一帧的frame2:', frames.shape)

    return frames  # 返回处理后的张量


def load_fix_frames(end_frame, frame_number, valid_frame_number):

    frame_path, frame_name = os.path.split(end_frame)  # 获取帧的路径和名称
    #print('frame_name', frame_name)
    #print('frame_number', frame_number)
    assert int(frame_name[7:12]) == frame_number  # 确保帧名称中的数字部分与给定的帧号一致
    #assert (int(frame_name.split('.')[0])+1) == frame_number  # 确保帧名称中的数字部分与给定的帧号一致

    frame_number = min(frame_number, valid_frame_number)  # 确保帧号不超过有效帧号
    start_frame_number = frame_number - DEPTH + 1  # 计算起始帧号
    start_frame_number = max(1, start_frame_number)  # 确保起始帧号不小于0
    frame_list = [f for f in range(start_frame_number, frame_number + 1)]  # 生成从起始帧号到当前帧号的帧列表

    if len(frame_list) < DEPTH:  # 如果帧列表长度小于深度，进行填充
        nsh = DEPTH - len(frame_list)  # 计算需要填充的帧数
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)  # 填充帧列表

    frames = np.zeros((224, 384))  # 初始化帧数组
    count = 0.0  # 初始化计数器

    for i in range(len(frame_list)):  # 遍历帧列表，处理每一帧
        #imgpath = os.path.join(frame_path, int(frame_name.split('.')[0])+1.format(frame_list[i], frame_name[-3:])) # 构建图像路径

        imgpath = os.path.join(frame_path, 'fixMap_{0:05d}.{1:s}'.format(frame_list[i], frame_name[-3:]))  # 构建图像路径，这行代码的意思是啥
        #print('fixation_img_path：', imgpath)
        if i == len(frame_list) - 1:
            img_new = cv2.resize(cv2.imread(imgpath), (384, 224))  # 读取图像并调整大小
            img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB) / 255.0  # 将图像从BGR转换为RGB
            print('我实际上是最后一帧的fixation的imgpath:', imgpath)
            frame2 = F.to_tensor(img_new)  # 将图像转换为张量
            frame2 = frame2.mean(dim=0, keepdim=True)

        try:
            # img = cv2.resize(cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE), (384, 224))  # 读取图像并调整大小
            # img = img / 255.0  # 归一化图像
            img = cv2.resize(cv2.imread(imgpath), (384, 224))  # 读取图像并调整大小
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # 将图像从BGR转换为RGB

            #print('查看一下img读取出来有没有问题')
            #print('fixation_map:', img.shape, img.max(), img.min())
            frames = frames + img  # 累加图像
            count = count + 1  # 增加计数器
        except:
            continue  # 如果读取图像失败，跳过该帧

    frames = frames / count  # 计算平均图像
    frames = frames / frames.sum()  # 归一化平均图像
    frames = F.to_tensor(frames)  # 将图像转换为张量
    #print('从源头处检查fixation_map的数据')
    #print('fixation_map:', frames.shape, frames.max(), frames.min())

    '''直接加载最后一帧的fixation_map'''
    frames = frame2

    return frames  # 返回处理后的张量


class LoadVideoAudio(object):
    # 测试用的
    """
        load the audio video
    """

    def __init__(self, stimuli_in, vfps):
        """
        :param stimuli_in:
        :param gt_in:
        """

        # self.root_folder = stimuli_in + '/frames/'
        self.root_folder = stimuli_in
        self.sample = []
        self.batch_size = 1
        fr = vfps
        # print('路径：', self.root_folder)
        video_frames = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]  # 加载视频帧：从指定文件夹（root_folder中获取三个特定类型文件）

        video_frames.sort()
        total_frame = str(len(video_frames))

        audio_file = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                      if f.endswith('.wav')]
        self.audio_data = load_wavfile(total_frame, audio_file[0])

        haptic_file = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                       if f.endswith('.txt')]
        self.haptic_data = load_txt_file(haptic_file[0])

        # pdb.set_trace()
        cnt = 0
        for video_frame in video_frames:
            # frame_number = os.path.basename(video_frame)[0:-4]
            frame_number = os.path.basename(video_frame)[4:9]
            sample = {'total_frame': total_frame, 'fps': fr,
                      'img': video_frame, 'frame_number': frame_number}
            self.sample.append(sample)
            cnt = cnt + 1
        # print(f"sample:{sample}")

        # print(f"frame_number:{frame_number}")

    def __len__(self):
        # return len(self.sample)
        return int(len(self.sample) / self.batch_size)

    def __getitem__(self, item):
        # print(f"item:{item}")
        sample = self.sample[item: item + self.batch_size]
        # print(f"sample:{sample}")
        # sample_AEM = self.sample_AEM[item : item + self.batch_size]

        video_data_equi_batch = []
        video_data_cube_batch = []
        audio_data_batch = []
        haptic_data_batch = []
        # AEM_data_batch = []
        # gt_data_batch = []
        '''处理一个批次的数据，其中 video_data_equi_batch 包含了等距视频数据，
         video_data_cube_batch 包含了立方体视频数据，audio_data_batch 包含了音频数据，
         AEM_data_batch 包含了 AEM 数据'''
        for i in range(self.batch_size):
            audio_params.EXAMPLE_HOP_SECONDS = 1 / int(sample[i]['fps'])
            audio_data, valid_frame_number = get_wavFeature(self.audio_data, int(sample[i]['frame_number']))
            audio_data_batch.append(audio_data)

            # print(f"sample[i]:{sample[i]}")

            '自己撰写的触觉模型'
            # haptic_data = self.haptic_data
            haptic_data, valid_frame_number = get_haptic(self.haptic_data, int(sample[i]['frame_number']))
            haptic_data = torch.from_numpy(haptic_data)
            # print('haptic_data:', haptic_data.shape, type(haptic_data))
            haptic_data_batch.append(haptic_data)


            video_data_equi, video_data_cube = load_video_frames(sample[i]['img'], int(sample[i]['frame_number']),
                                                                 valid_frame_number)
            # print(f"valid_frame_number:{valid_frame_number}")
            # video_data_cube = load_video_frames(sample[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
            video_data_equi_batch.append(video_data_equi)
            video_data_cube_batch.append(video_data_cube)
            '''(logo)print("sample_AEM[i]['frame']: ", sample_AEM[i]['frame'])
            AEM_data = load_AEM_frames(sample_AEM[i]['frame'], int(sample_AEM[i]['frame_number']), valid_frame_number)
            AEM_data_batch.append(AEM_data)
            '''
            # AEM_data(AEM数据对应的视频帧的文件路径)
            # AEM_data = load_AEM_frames(sample_AEM[i]['frame'], int(sample_AEM[i]['frame_number']), valid_frame_number)
            # AEM_data_batch.append(AEM_data)

            # gt_data = load_gt_frames(sample[i]['gtsal_frame'], int(sample[i]['frame_number']), valid_frame_number)
            # gt_data_batch.append(gt_data)

        video_data_equi_batch = torch.stack(video_data_equi_batch, dim=0)  # [10, 3, 16, 256, 512]
        video_data_cube_batch = torch.stack(video_data_cube_batch, dim=0)  # [10, 6, 3, 16, 128, 128]
        audio_data_batch = torch.stack(audio_data_batch, dim=0)  # [10, 3, 16, 64, 64]
        haptic_data_batch = torch.stack(haptic_data_batch, dim=0)  # [10, 16, 128]

        '''因为不想改变后面，所以直接把触觉的值付给音频的名称'''
        audio_data_batch = haptic_data_batch

        # AEM_data_batch = torch.stack(AEM_data_batch, dim=0)                   # [10, 1, 8, 16]
        # gt_data_batch = torch.stack(gt_data_batch, dim=0)                     # [10, 1, 8, 16]

        return video_data_equi_batch, video_data_cube_batch, audio_data_batch  # , AEM_data_batch



class LoadVideoAudio_TRAIN(object):
#训练用的

    """
        load the audio video
    """

    def __init__(self, stimuli_in, vfps):
        """
        :param stimuli_in:
        :param gt_in:
        """

        #self.root_folder = stimuli_in + '/frames/'
        self.batch_size = 2
        self.root_folder = stimuli_in
        last_part = os.path.split(stimuli_in)[-1]
        second_last_part = os.path.split(os.path.split(stimuli_in)[0])[-1]

        #print(f"second_last_part:{second_last_part}")
        #print(f"last_part:{last_part}")
        # self.gt_folder = os.path.join('C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\saliency(change_name)', second_last_part, last_part)
        # self.fix_folder = os.path.join('C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\fixation(change_name)', second_last_part, last_part)

        # self.gt_folder = os.path.join('F:\\AVS360\\new\\AVS360\\data\\saliency(change_name)',
        #                               second_last_part, last_part)
        # self.fix_folder = os.path.join('F:\\AVS360\\new\\AVS360\\data\\fixation(change_name)',
        #                                second_last_part, last_part)

        self.gt_folder = os.path.join('/data1/liuhengfa/my_own_code_for_saliency/data/saliency_renamed',
                                      second_last_part, last_part)
        self.fix_folder = os.path.join('/data1/liuhengfa/my_own_code_for_saliency/data/fixation_renamed',
                                       second_last_part, last_part)



        # self.gt_folder = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\saliency(change_name)\\hys\\01'
        # self.fix_folder = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\fixation(change_name)\\hys\\01'

        self.sample = []

        fr = vfps
        
        video_frames = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        video_frames.sort()
        total_frame = str(len(video_frames))

        audio_file = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                      if f.endswith('.wav')]

        haptic_file = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                       if f.endswith('.txt')]

        self.audio_data = load_wavfile(total_frame, audio_file[0])

        self.haptic_data = load_txt_file(haptic_file[0])
        
        gtsal_frames = [os.path.join(self.gt_folder, f) for f in os.listdir(self.gt_folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        gtsal_frames.sort()
        #print(f"gtsal_frames:{gtsal_frames}")

        fixmap_frames = [os.path.join(self.fix_folder, f) for f in os.listdir(self.fix_folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]

        #fixmap_frames.sort()
        fixmap_frames = sorted(fixmap_frames, key=lambda x: int(re.search(r'(\d+)\.jpg$', x).group(1)))
        #print(f"fixmap_frames:{fixmap_frames}")

        cnt = 0

        '''
        video_frames = video_frames[:self.max_frame_num]
        random.shuffle(video_frames)
        video_frames = video_frames[:50]
        '''
        for v, video_frame in enumerate(video_frames):
            frame_number = os.path.basename(video_frame)[4:9]
            #print(f"frame_number:{frame_number}")
            # print(f"v:{v}")
            # print(f"gtsal_frame[v]:{gtsal_frames[v]}")
            gtsal_frame = gtsal_frames[v]

            # print(f"fixmap_frames[v]:{fixmap_frames[v]}")
            fixmap_frame = fixmap_frames[v]
            #print(f"fixmap_frame:{fixmap_frame}")
            sample = {'total_frame': total_frame, 'fps': fr,
                      'img': video_frame, 'eyeMap': gtsal_frame, 'fixMap': fixmap_frame, 'frame_number': frame_number}
            self.sample.append(sample)
            cnt = cnt + 1
        # print(f"初始化时的 frame_number: {frame_number}")


        #random.shuffle(self.sample)
        self.sample = self.sample[::5]

        '''
        # AEM
        vid_name = stimuli_in.split('mono')[-1][:-3]
        self.root_folder_AEM = '/media/fchang/My Passport/ICME2020/sound_map/' + vid_name[2:] + '/frame/'
        self.sample_AEM = []
        AEM_frames = [os.path.join(self.root_folder_AEM, f) for f in os.listdir(self.root_folder_AEM)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        AEM_frames.sort()
        cnt = 0
        total_frame = str(len(AEM_frames))
        for AEM_frame in AEM_frames:
            frame_number = os.path.basename(AEM_frame)[0:-4]
            sample_AEM = {'total_frame': total_frame, 'fps': fr,
                      'frame': AEM_frame, 'frame_number': frame_number}
            self.sample_AEM.append(sample_AEM)
            cnt = cnt + 1
        '''    

    def __len__(self):
        return int(len(self.sample)/self.batch_size)

    def __getitem__(self, item):

        # print('xxxx----xxx----运行的utils文件夹里的这个加载数据的函数xxxxxxx-xxxxxx')
        # print('xxxx----xxx----运行的utils文件夹里的这个加载数据的函数xxxxxxx-xxxxxx')
        # print('xxxx----xxx----运行的utils文件夹里这个加载数据的函数xxxxxxx-xxxxxx')

        sample = self.sample[item : item + self.batch_size]
        #sample_AEM = self.sample_AEM[item : item + self.batch_size]
        #print('int(len(self.sample)):', int(len(self.sample)))
        #print('一共取了多少次？', int(len(self.sample) / self.batch_size))

        video_data_equi_batch = []
        video_data_cube_batch = []
        audio_data_batch = []
        haptic_data_batch = []
        #AEM_data_batch = []
        gt_data_batch = []
        fix_data_batch = []

        '''处理一个批次的数据，其中 video_data_equi_batch 包含了等距视频数据，
         video_data_cube_batch 包含了立方体视频数据，audio_data_batch 包含了音频数据，
         AEM_data_batch 包含了 AEM 数据'''
        for i in range(self.batch_size):
            audio_params.EXAMPLE_HOP_SECONDS = 1/int(sample[i]['fps'])

            #print('打印一下这个音频信号的特征输入get_wave之前是什么:', self.audio_data.shape)  # [23, 64, 64]
            audio_data, valid_frame_number = get_wavFeature(self.audio_data, int(sample[i]['frame_number']))
            #print('打印一下这个音频信号的特征经过get_wave之后的输出是什么', audio_data.shape, type(audio_data))  # [3, 16, 64, 64]
            audio_data_batch.append(audio_data)

            #print('打印一下这个触觉信号的特征输入get_hapticFeature之前是什么:', self.haptic_data.shape)

            '自己撰写的触觉模型'
            #haptic_data = self.haptic_data
            haptic_data, valid_frame_number = get_haptic(self.haptic_data, 64)
            haptic_data = torch.from_numpy(haptic_data)
            #print('haptic_data:', haptic_data.shape, type(haptic_data))
            haptic_data_batch.append(haptic_data)



            video_data_equi, video_data_cube = load_video_frames(sample[i]['img'], int(sample[i]['frame_number']), valid_frame_number)
            #video_data_cube = load_video_frames(sample[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
            video_data_equi_batch.append(video_data_equi)
            video_data_cube_batch.append(video_data_cube)

            #AEM_data = load_AEM_frames(sample_AEM[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
            #AEM_data_batch.append(AEM_data)
            
            gt_data = load_gt_frames(sample[i]['eyeMap'], int(sample[i]['frame_number']), valid_frame_number)
            gt_data_batch.append(gt_data)


            fix_data = load_fix_frames(sample[i]['fixMap'], int(sample[i]['frame_number']), valid_frame_number)
            fix_data_batch.append(fix_data)

        video_data_equi_batch = torch.stack(video_data_equi_batch, dim=0)     # [1, 3, 16, 320, 640]
        video_data_cube_batch = torch.stack(video_data_cube_batch, dim=0)     # [1, 6, 3, 16, 128, 128]

        #print('堆叠前--audio_data_batch:',  type(audio_data_batch), len(audio_data_batch))
        audio_data_batch = torch.stack(audio_data_batch, dim=0)               # [1, 3, 16, 64, 64],这个是个张量
        #print('堆叠后audio_data_batch:',  type(audio_data_batch), audio_data_batch.shape)

        #print('堆叠前--haptic_data_batch', type(haptic_data_batch), len(haptic_data_batch))
        #haptic_data_batch = torch.from_numpy(haptic_data_batch)
        haptic_data_batch = torch.stack(haptic_data_batch, dim=0)             # [1, 1, 16, 64, 64]
        #print('堆叠后---haptic_data_batch:', haptic_data_batch.shape, type(haptic_data_batch))


        '''这一步的赋值只是为了让输出的时候，音频的输出实际是触觉的数值'''
        audio_data_batch = haptic_data_batch
        #AEM_data_batch = torch.stack(AEM_data_batch, dim=0)

        gt_data_batch = torch.stack(gt_data_batch, dim=0)                     # [1, 1, 320, 640]
        fix_data_batch = torch.stack(fix_data_batch, dim=0)                   # [1, 1, 320, 640]

        '''音频实际的输出是触觉，只不是过我没改名字而已'''
        return video_data_equi_batch, video_data_cube_batch, audio_data_batch, gt_data_batch, fix_data_batch
        # return video_data_equi_batch, video_data_cube_batch, gt_data_batch





