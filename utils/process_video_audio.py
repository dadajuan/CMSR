
# generic must imports
import os
import torch
import numpy as np
import cv2

import utils.audio_params as audio_params
import librosa as sf
from utils.audio_features import waveform_to_feature
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms.functional as F
from utils.equi_to_cube import Equi2Cube
import pdb
import random

__all__ = ['LoadVideoAudio']

#defined params @TODO move them to a parameter config file
DEPTH = 16
GT_WIDTH = 32
GT_HIGHT = 40

e2c = Equi2Cube(128, 256, 512)     # Equi2Cube(out_w, in_h, in_w) 

MEAN = [ 110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0 ]
STD = [ 38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0 ]

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
    n_frame = in_data.shape[0]  # n_frame的值是输入数据in_data的第一个维度的大小：帧的数量
    # print("n_frame:", n_frame)
    # print("in_data shape:", in_data.shape)
    # print(f"create_data_packet 函数中原始 frame_number: {frame_number}")
    frame_number = min(frame_number, n_frame)
    # print(f"create_data_packet 函数中修改后 frame_number: {frame_number}")
    starting_frame = frame_number - DEPTH + 1
    starting_frame = max(0, starting_frame)  # 确保我们没有任何负帧
    data_pack = in_data[starting_frame:frame_number+1, :, :]
    n_pack = data_pack.shape[0]

    if n_pack < DEPTH:
        nsh = DEPTH - n_pack
        data_pack = np.concatenate((np.tile(data_pack[0, :, :], (nsh, 1, 1)), data_pack), axis=0)
    print("填充之后音频的数据大小---data_pack shape:", data_pack.shape)
    assert data_pack.shape[0] == DEPTH

    data_pack = np.tile(data_pack, (3, 1, 1, 1)) # 这行代码 data_pack = np.tile(data_pack, (3, 1, 1, 1)) 使用 np.tile 函数沿第一个维度（通道）复制 data_pack 数组。这实际上将单通道数据数组转换为三通道数据数组，通过沿第一个维度重复原始数据三次来实现。

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
    wav_data, sr = sf.load(path=wav_file, sr=audio_params.SAMPLE_RATE, dtype='float32')
    assert sf.get_duration(y=wav_data, sr=sr) > 1
    #将波形数据转换成对数梅尔谱图特征
    features = waveform_to_feature(wav_data, sr)
    features = np.resize(features, (int(total_frame), features.shape[1], features.shape[2]))

    # 将音频数据设置为全零
    #features = np.zeros_like(features)

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
    # load video frames, process them and return a suitable tensor
    frame_path, frame_name = os.path.split(end_frame)
    print(f"frame_name:{frame_name}")
    print(frame_name[4:9])
    print(frame_number)
    assert int(frame_name[4:9]) == frame_number
    frame_number = min(frame_number, valid_frame_number)  # 当前帧和有效帧数中的最小值,确保你取得值不会超过有效帧数
    print(f"frame_number:{frame_number}")

    start_frame_number = frame_number - DEPTH+1   # DEPTH=16,这个是指每次预测使用多少帧的数据
    print(f" start_frame_number: { start_frame_number}")  # 好多时候是负数，然后他就被设置为1，即把第一帧重复多次；
    start_frame_number = max(1, start_frame_number)
    print(f" start_frame_number: {start_frame_number}")
    frame_list = [f for f in range(start_frame_number, frame_number+1)]  # 包括首部但不包括尾部，所以要加1
    print(f"frame_list:{frame_list}")


    if len(frame_list) < DEPTH:
        nsh = DEPTH - len(frame_list)  # 要补充的帧数
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)  # 将第一帧复制nsh次，然后和原来的帧列表拼接
        print(f"如果不够,需要补充的话,frame_list:{frame_list}")

    frames_cube = []
    frames_equi = []

    #如果帧列表长度小于深度，进行填充
    for i in range(len(frame_list)):
        # print("len(frame_list):",len(frame_list))
        # print(frame_path, 'frame_{0:05d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
        imgpath = os.path.join(frame_path, 'img_{0:05d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
        #print("Constructed imgpath:", imgpath)

        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                #img = img.convert('RGB')
               # print('原始读取的话，其图像尺寸是多大:', cv2.imread(imgpath).shape)  # (1920,3840,3)
                img = cv2.resize(cv2.imread(imgpath), (512, 256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0   # 将图像转换为RGB格式并归一化
                #pix=np.array(img.getdata()).reshape(256,320,3)/255.0 


                # 输入cube的图片,即进行立方体投影之前的图片是512*256，然后经过投影之后得到的是128*128的图像，而等距投影自己得到的矩阵确是320*256
                img_c = e2c.to_cube(img)
                img_cube = []
                for face in range(6):
                    img_f = F.to_tensor(img_c[face])
                    img_f = F.normalize(img_f, MEAN, STD)
                    img_cube.append(img_f)
                img_cube_data = torch.stack(img_cube)               
                frames_cube.append(img_cube_data)

                img = cv2.resize(img, (320, 256))
                #img = cv2.resize(cv2.imread(imgpath), (640, 320)) # 这个其实是我自己想要的，但不太确定这个等距投影和尺寸有没有关系。
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # 将图像转换为RGB格式并归一化
                img_equi = F.to_tensor(img)
                img_equi = F.normalize(img_equi, MEAN, STD)
                frames_equi.append(img_equi)
                
    data_cube = torch.stack(frames_cube, dim=0)
    data_equi = torch.stack(frames_equi, dim=0)

    '''
    frames = []
    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, '{0:07d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                #pdb.set_trace()
                #img = img.convert('RGB')
                img = cv2.resize(cv2.imread(imgpath), (512,256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
                img = F.to_tensor(img)
                img = F.normalize(img, MEAN, STD)
                frames.append(img)
    data_equi = torch.stack(frames, dim=0)
    '''
    return data_equi.permute([1, 0, 2, 3]), data_cube.permute([1, 2, 0, 3, 4])
    #return data_cube.permute([1, 2, 0, 3, 4])


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


def load_gt_frames(end_frame, frame_number, valid_frame_number):
    # load video frames, process them and return a suitable tensor
    frame_path, frame_name = os.path.split(end_frame)
    print(f"frame_name:{frame_name}")
    print(frame_name[7:12])
    print(f"frame_path:{frame_path}")
    assert int(frame_name[7:12]) == frame_number
    frame_number = min(frame_number, valid_frame_number)
    start_frame_number = frame_number - DEPTH+1
    # print(f"start_frame_number:{start_frame_number}")6
    start_frame_number = max(0, start_frame_number)
    # print(f"start_frame_number:{start_frame_number}")6
    frame_list = [f for f in range(start_frame_number, frame_number+1)]
    if len(frame_list) < DEPTH:
        nsh = DEPTH - len(frame_list)
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)
    frames = np.zeros((32, 40))
    count = 0.0

    print(f"len(frame_list):{len(frame_list)}")#16
    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, 'eyeMap_{0:05d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
        try:
            img = cv2.resize(cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE), (40, 32))
            #img = cv2.GaussianBlur(img, (7,7),cv2.BORDER_DEFAULT)
            img = img/255.0
            frames = frames + img
            count = count + 1
        except:
            continue
    print(f"count:{count}")
    frames = frames/count
    frames = frames/frames.sum()
    frames = F.to_tensor(frames)
    #pdb.set_trace()
    #data = torch.stack(frames, dim=0)
    return frames


#test
# def load_gt_frames(end_frame, frame_number, valid_frame_number):
#     frame_path, frame_name = os.path.split(end_frame)
#     assert int(frame_name[7:12]) == frame_number
#     frame_number = min(frame_number, valid_frame_number)
#     start_frame_number = frame_number - DEPTH + 1
#     start_frame_number = max(0, start_frame_number)
#     frame_list = [f for f in range(start_frame_number, frame_number + 1)]
#     if len(frame_list) < DEPTH:
#         nsh = DEPTH - len(frame_list)
#         frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)
#     frames = np.zeros((32, 40))
#     count = 0.0
#
#     for i in range(len(frame_list)):
#         imgpath = os.path.join(frame_path, 'eyeMap_{0:05d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
#         print(f"Trying to process frame: {imgpath}")
#         try:
#             img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 print(f"Failed to read image: {imgpath}")
#                 continue
#             img = cv2.resize(img, (40, 32))  # (64, 32)
#             img = img / 255.0
#             frames = frames + img
#             count = count + 1
#         except Exception as e:
#             print(f"Exception occurred while processing frame {frame_list[i]}: {e}")
#             continue
#
#     if count > 0:
#         frames = frames / count
#         frames = frames / frames.sum()
#     else:
#         raise ValueError("No valid frames found to process.")
#
#     frames = F.to_tensor(frames)
#     return frames


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

        #self.root_folder = stimuli_in + '/frames/'
        self.root_folder = stimuli_in
        self.sample = []
        self.batch_size = 1
        fr = vfps
        print('路径：', self.root_folder)
        video_frames = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]  # 加载视频帧：从指定文件夹（root_folder中获取三个特定类型文件）

        video_frames.sort()
        total_frame = str(len(video_frames))
        print(f"total_frame:{total_frame}")  # 32

        audio_file = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                      if f.endswith('.wav')]
        #print(f"audio_file:{audio_file}")
        self.audio_data = load_wavfile(total_frame, audio_file[0])

        
        #pdb.set_trace()  
        cnt = 0
        for video_frame in video_frames:
            # frame_number = os.path.basename(video_frame)[0:-4]
            #print(f"video_frame:{video_frame}")

            frame_number = os.path.basename(video_frame)[4:9]
            #print(f"frame_number:{frame_number}")
            sample = {'total_frame': total_frame, 'fps': fr,
                      'img': video_frame, 'frame_number': frame_number}
            self.sample.append(sample)
            cnt = cnt + 1
       # print(f"sample:{sample}")

       # print(f"frame_number:{frame_number}")

    def __len__(self):
        #return len(self.sample)
        return int(len(self.sample)/self.batch_size)


    def __getitem__(self, item):
        #print(f"item:{item}")
        sample = self.sample[item : item + self.batch_size]
        #print(f"sample:{sample}")
        # sample_AEM = self.sample_AEM[item : item + self.batch_size]
        
        video_data_equi_batch = []
        video_data_cube_batch = []
        audio_data_batch = []
        # AEM_data_batch = []
        #gt_data_batch = []
        '''处理一个批次的数据，其中 video_data_equi_batch 包含了等距视频数据，
         video_data_cube_batch 包含了立方体视频数据，audio_data_batch 包含了音频数据，
         AEM_data_batch 包含了 AEM 数据'''
        for i in range(self.batch_size):
            audio_params.EXAMPLE_HOP_SECONDS = 1/int(sample[i]['fps'])
           # print('i的值', i)
            audio_data, valid_frame_number = get_wavFeature(self.audio_data, int(sample[i]['frame_number']))
            audio_data_batch.append(audio_data)

           # print(f"sample[i]:{sample[i]}")
            video_data_equi, video_data_cube = load_video_frames(sample[i]['img'], int(sample[i]['frame_number']), valid_frame_number)
            print(f"valid_frame_number:{valid_frame_number}")
            #video_data_cube = load_video_frames(sample[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
            video_data_equi_batch.append(video_data_equi)
            video_data_cube_batch.append(video_data_cube)

            '''(logo)print("sample_AEM[i]['frame']: ", sample_AEM[i]['frame'])
            AEM_data = load_AEM_frames(sample_AEM[i]['frame'], int(sample_AEM[i]['frame_number']), valid_frame_number)
            AEM_data_batch.append(AEM_data)
            '''
            #AEM_data(AEM数据对应的视频帧的文件路径)
            # AEM_data = load_AEM_frames(sample_AEM[i]['frame'], int(sample_AEM[i]['frame_number']), valid_frame_number)
            # AEM_data_batch.append(AEM_data)
            
            #gt_data = load_gt_frames(sample[i]['gtsal_frame'], int(sample[i]['frame_number']), valid_frame_number)
            #gt_data_batch.append(gt_data)

        video_data_equi_batch = torch.stack(video_data_equi_batch, dim=0)     # [10, 3, 16, 256, 512]---
        video_data_cube_batch = torch.stack(video_data_cube_batch, dim=0)     # [10, 6, 3, 16, 128, 128]---
        audio_data_batch = torch.stack(audio_data_batch, dim=0)               # [10, 3, 16, 64, 64]---
        # AEM_data_batch = torch.stack(AEM_data_batch, dim=0)                   # [10, 1, 8, 16]
        #gt_data_batch = torch.stack(gt_data_batch, dim=0)                     # [10, 1, 8, 16]

        # 输出的尺寸分别为 [1,3,16,256,320]; [1,6,3,16,128,128]; [1,3,16,64,64]
        return video_data_equi_batch, video_data_cube_batch, audio_data_batch#, AEM_data_batch





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
        self.batch_size = 1
        self.root_folder = stimuli_in
        last_part = os.path.split(stimuli_in)[-1]
        second_last_part = os.path.split(os.path.split(stimuli_in)[0])[-1]
        #self.gt_folder = os.path.join('C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\saliency(change_name)', second_last_part, last_part)
        self.gt_folder = os.path.join('F:\\AVS360\\new\\AVS360\\data\\saliency(change_name)', second_last_part, last_part)  # 加载显著性图
        # self.gt_folder = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\saliency(change_name)\\hys\\01'
        self.sample = []

        fr = vfps
        
        video_frames = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        video_frames.sort()
        total_frame = str(len(video_frames))

        audio_file = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                      if f.endswith('.wav')]
        self.audio_data = load_wavfile(total_frame, audio_file[0])
        
        gtsal_frames = [os.path.join(self.gt_folder, f) for f in os.listdir(self.gt_folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        gtsal_frames.sort()

        cnt = 0

        '''
        video_frames = video_frames[:self.max_frame_num]
        random.shuffle(video_frames)
        video_frames = video_frames[:50]
        '''
        for v, video_frame in enumerate(video_frames):
            frame_number = os.path.basename(video_frame)[4:9]
            gtsal_frame = gtsal_frames[v]
            # print(f"v:{v}")
            # print(f"gtsal_frame:{gtsal_frame}")
            sample = {'total_frame': total_frame, 'fps': fr,
                      'img': video_frame, 'eyeMap': gtsal_frame, 'frame_number': frame_number}
            self.sample.append(sample)

            cnt = cnt + 1
        #print(f"初始化时的 frame_number: {frame_number}")

        print('读取第一个视频文件夹sample的长度:', len(self.sample))
        #random.shuffle(self.sample)
        #self.sample = self.sample[::5]
        print('处理之后第一个视频文件夹sample的长度:', len(self.sample))
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
    
        sample = self.sample[item : item + self.batch_size]
        #sample_AEM = self.sample_AEM[item : item + self.batch_size]
        
        video_data_equi_batch = []
        video_data_cube_batch = []
        audio_data_batch = []
        #AEM_data_batch = []
        gt_data_batch = []
        '''处理一个批次的数据，其中 video_data_equi_batch 包含了等距视频数据，
         video_data_cube_batch 包含了立方体视频数据，audio_data_batch 包含了音频数据，
         AEM_data_batch 包含了 AEM 数据'''
        for i in range(self.batch_size):
            audio_params.EXAMPLE_HOP_SECONDS = 1/int(sample[i]['fps'])

            audio_data, valid_frame_number = get_wavFeature(self.audio_data, int(sample[i]['frame_number']))
            audio_data_batch.append(audio_data)

            #print(f"sample[i]:{sample[i]}")

            video_data_equi, video_data_cube = load_video_frames(sample[i]['img'], int(sample[i]['frame_number']), valid_frame_number)
            #video_data_cube = load_video_frames(sample[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
            video_data_equi_batch.append(video_data_equi)
            video_data_cube_batch.append(video_data_cube)

            #AEM_data = load_AEM_frames(sample_AEM[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
            #AEM_data_batch.append(AEM_data)
            
            gt_data = load_gt_frames(sample[i]['eyeMap'], int(sample[i]['frame_number']), valid_frame_number)
            gt_data_batch.append(gt_data)

        
        video_data_equi_batch = torch.stack(video_data_equi_batch, dim=0)     # [10, 3, 16, 256, 512]
        video_data_cube_batch = torch.stack(video_data_cube_batch, dim=0)     # [10, 6, 3, 16, 128, 128]
        audio_data_batch = torch.stack(audio_data_batch, dim=0)               # [10, 3, 16, 64, 64]
        #AEM_data_batch = torch.stack(AEM_data_batch, dim=0)                   # [10, 1, 8, 16]
        gt_data_batch = torch.stack(gt_data_batch, dim=0)                     # [10, 1, 8, 16]

        #return video_data_equi_batch, video_data_cube_batch, audio_data_batch, AEM_data_batch, gt_data_batch
        return video_data_equi_batch, video_data_cube_batch, audio_data_batch, gt_data_batch
        # return video_data_equi_batch, video_data_cube_batch, gt_data_batch


# if __name__ == "__main__":
#    a = LoadVideoAudio_TRAIN("C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\video\\hys\\01",  29.970030)# 视频文件路径；地面真值（gt）的路径
#    video_data_equi, video_data_cube, audio_data,gt_map = a.__getitem__(a.__len__()-1)
#    print(a.__len__())#输出样本数量
#    print(video_data_equi.shape)
#    print(video_data_cube.shape)
#    print(audio_data.shape)
#    print(gt_map.shape)
'''
torch.Size([1, 3, 16, 256, 320]):  
1: 批次大小（batch size），表示有一个样本。
3: 通道数（channels），表示视频的RGB三个通道。
16: 帧数（frames），表示视频包含16帧。
256: 高度（height），表示每帧的高度为256像素。
320: 宽度（width），表示每帧的宽度为320像素。

torch.Size([1, 6, 3, 16, 128, 128]):  
1: 批次大小（batch size），表示有一个样本。
6: 立方体面数（cube faces），表示立方体视频的6个面。
3: 通道数（channels），表示每个面的RGB三个通道。
16: 帧数（frames），表示每个面包含16帧。
128: 高度（height），表示每帧的高度为128像素。
128: 宽度（width），表示每帧的宽度为128像素。

torch.Size([1, 3, 16, 64, 64]):  
1: 批次大小（batch size），表示有一个样本。
3: 通道数（channels），表示音频数据的三个通道。
16: 帧数（frames），表示音频数据包含16帧。
64: 高度（height），表示每帧的高度为64。
64: 宽度（width），表示每帧的宽度为64。

torch.Size([1, 1, 32, 40]):
1: 批次大小（batch size），表示有一个样本。
1: 通道数（channels），表示地面真值数据的单通道（灰度图）。
32: 高度（height），表示地面真值数据的高度为32像素。
40: 宽度（width），表示地面真值数据的宽度为40像素
'''
   # print(AEM_data.shape)
if __name__ == "__main__":
    a = LoadVideoAudio("C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\video\\hys\\01",
                             29.970030)  # 视频文件路径；地面真值（gt）的路径
    video_data_equi, video_data_cube, audio_data = a.__getitem__(a.__len__()-1)
    print(a.__len__())  # 输出样本数量
    print(video_data_equi.shape)
    print(video_data_cube.shape)
    print(audio_data.shape)



