import re
import os

import torch
import numpy as np

from PIL import Image
from utils.process_video_audio_new import LoadVideoAudio

from model_predi import DAVE
from model_my import Video_HAPTIC_SaliencyModel, AffineTransform, TimeSeriesTransformer, haptic_encoder_fenlei
import pdb
import matplotlib.pyplot as plt
import cv2


VIDEO_TEST_FOLDER = r'/data1/liuhengfa/my_own_code_for_saliency/data/video0_renamed/test_4'


'''这个是生成的显著性图的地址'''
OUTPUT = '/data1/liuhengfa/my_own_code_for_saliency/test_for_tuning_old_dataset/output_dataset_test14/test_4_test'
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)
# where tofind the model weights

# MODEL_PATH = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\outputDAVE_ep29.pkl'
# some config parameters

IMG_WIDTH = 256
IMG_HIGHT = 320
TRG_WIDTH = 32
TRG_HIGHT = 40


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def load_model_parameters(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
class PredictSaliency(object):

    def __init__(self, model_path, model_path_haptic_encoder,  model_path_haptic_affine):
        super(PredictSaliency, self).__init__()

        self.video_list = [os.path.join(VIDEO_TEST_FOLDER, p) for p in os.listdir(VIDEO_TEST_FOLDER)]
        print(self.video_list)

        # self.model = DAVE()
        # self.model= torch.load(model_path)
        # self.output = OUTPUT
        # if not os.path.exists(self.output):
        #         os.mkdir(self.output)
        # self.model = self.model.cuda()
        # self.model.eval()


        self.model = Video_HAPTIC_SaliencyModel()
        self.model = torch.load(model_path)
        self.output = OUTPUT
        if not os.path.exists(self.output):
                 os.mkdir(self.output)
        self.model = self.model.to(device)
        self.model.eval()

        '''两种触觉编码方案'''
        # 方案1
        self.haptic_encoder = TimeSeriesTransformer()
        self.haptic_encoder = self.haptic_encoder.to(device)
        self.haptic_encoder.eval()



        '''加载触觉的仿射变换模型'''

        self.model_haptic_affine = AffineTransform()  # 默认就是128，所以如果是方案1的话不需要填，方案2的话是需要写256；
        self.model_haptic_affine = torch.load(model_path_haptic_affine)
        self.model_haptic_affine = self.model_haptic_affine.to(device)
        self.model_haptic_affine.eval()



    @staticmethod
    def _load_state_dict_(filepath):
        if os.path.isfile(filepath):
            pdb.set_trace()
            print("=> loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath, map_location=device)
            
            pattern = re.compile(r'module+\.*')
            state_dict = checkpoint['state_dict']
            
            for key in list(state_dict.keys()):
                if 'video_branch' in key: 
                   state_dict[key[:12] + '_cubic' + key[12:]] = state_dict[key]
            
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = re.sub('module.', '', key)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        return state_dict

    def predict(self, stimuli_path, fps, out_path):

        #equator_bias = cv2.resize(cv2.imread('C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360_37\\ECB.png', 0), (10,8))

        equator_bias = cv2.resize(cv2.imread(r'/data1/liuhengfa/my_own_code_for_saliency/ECB.png', 0),
                                  (10, 8))
        equator_bias = torch.tensor(equator_bias).to(device, dtype=torch.float)
        equator_bias = equator_bias.cuda()
        equator_bias = equator_bias/equator_bias.max()

        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if not os.path.exists(out_path+ '/overlay'):  
            os.mkdir(out_path + '/overlay')
        video_loader = LoadVideoAudio(stimuli_path, fps)   # 在新的测试上这行就不对了


        vit = iter(video_loader)
        # print('len(video_loader)', len(video_loader))
        # print('vit', vit)
        for idx in range(len(video_loader)):


            video_data_equi, video_data_cube, audio_data= next(vit)
            # video_data_equi, video_data_cube, audio_data = next(vit)           # video_data_equi = [3, 16, 256, 512], video_data_cube = [6, 3, 16, 128, 128], audio_data = [3, 16, 64, 64], AEM_data = [1, 8, 16]
            print(idx, len(video_loader))
           
            #video_data_equi = video_data_equi.to(device=device, dtype=torch.float)
            #video_data_equi = video_data_equi.cuda()
            video_data_equi = video_data_equi.float().to(device=device)
            video_data_cube = video_data_cube.float().to(device=device)
            #video_data_cube = video_data_cube.to(device=device, dtype=torch.float)
            #video_data_cube = video_data_cube.cuda()

            # AEM_data = AEM_data.to(device=device, dtype=torch.float)
            # AEM_data = AEM_data.cuda()


            # audio_data = audio_data.to(torch.float32)  # 这里其实已经是加载的txt的触觉信号了，只是名字没改格式是[b, T, 3200]
            # audio_data = audio_data.to(device=device, dtype=torch.float)
            # audio_data = audio_data.cuda()

            '''触觉编码的两种方案'''

            '''方案1--当触觉是使用transformer进行编辑的时候'''

            audio_data = audio_data.float().to(device=device)
            haptic_data = self.haptic_encoder(audio_data)

            '''方案2--当触觉是使用resnet进行编辑的时候'''
            # audio_data = audio_data.float().cuda()
            # audio_data_last_time = audio_data[:, -1, :]
            # haptic_data = self.haptic_encoder.penultimate_layer_output(audio_data_last_time)

            print('xxxxx---编码的触觉的特征的维度--xxxxx', haptic_data.shape)




            haptic_out1, haptic_out2, haptic_out3, haptic_out4 = self.model_haptic_affine(haptic_data)
            print('haptic_out1, haptic_out2, haptic_out3, haptic_out4', haptic_out1.shape, haptic_out2.shape, haptic_out3.shape, haptic_out4.shape)


            # 预测显著图
            prediction = self.model(video_data_equi,  haptic_out1, haptic_out2, haptic_out3, haptic_out4)



            saliency = prediction.cpu().data.numpy()
            saliency = np.squeeze(saliency)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            saliency = Image.fromarray((saliency*255).astype(np.uint8))
            print('before resize----saliency,shape:', saliency.size)

            saliency = saliency.resize((384, 224), Image.LANCZOS)
            saliency.save('{}/{}.jpg'.format(out_path, idx+1), 'JPEG')
            print('成功保存显著图')

            # 释放缓存
            del prediction
            torch.cuda.empty_cache()



    def predict_sequences(self):
        for v in self.video_list[:]:
            print(v)
            sample_rate = 29.970030  # 默认采样率，根据需要更改
            bname = os.path.basename(v).split('.')[0]

            output_path = os.path.join(self.output, bname)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            # print('output_path', output_path)
            self.predict(v, sample_rate, output_path)


if __name__ == '__main__':
    print('''*****------开始预测------*****''')
    p = PredictSaliency()
    p.predict_sequences()
