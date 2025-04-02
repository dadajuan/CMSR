
import os
import torch
import cv2
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from train import TrainSaliency
from predict import PredictSaliency
from evaluation_saliencymap import evaluate_saliency_maps_2, get_subfolders
import wandb
import random


from datetime import datetime


device = torch.device("cuda:2")
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")



wandb.login(key='xxx')

if __name__ == "__main__":
    print('ssss')
    wandb.init(
        # Set the project where this run will be logged
        project="datasetold_AVS360",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"1-0.1-0.1结果{current_time}",
        # Track hyperparameters and run metadata
        config={

            "lr": 2*1e-5,
            "epochs": 152,
            '训练接和测试机': '均是train14',
            '是否加载预训练模型': 1,
            '是否是自己的数据集': '新的自采数据集',
            '损失的比例': '1, 0.1, 0.1',
            '触觉是否继续参与训练': '参与',
            '触觉编码网络是renset还是transformer': 'transformer'
        })
    config = wandb.config

    nb_epoch = 152

    t = TrainSaliency()

    '''为了保证测试的正确执行，严格按照地址写明真实的数据集的地址，以及fixation的地址，以及预测的地址'''

    '''这两个在main函数里找，这个是测试集的地址
    '''
    gt_saliency_path = '/data1/liuhengfa/my_own_code_for_saliency/data/saliency_renamed/test_4'
    gt_fix_path = '/data1/liuhengfa/my_own_code_for_saliency/data/fixation_renamed/test_4'
    '''这个在predict函数里找'''
    pd_path = '/data1/liuhengfa/my_own_code_for_saliency/test_for_tuning_old_dataset/output_dataset_test14/test_4_test'





    for epoch in tqdm(range(nb_epoch)):  # 逐个epoch进行训练,即遍历完所有的视频,以及每个视频的所有帧
        '''这一行如果注释掉就是不再进行训练，只是测试'''
        loss, kl_loss, cc_loss, nss_loss = t.train(epoch)

        if epoch in [150]:
            '''这三行的地址要根据trian函数去改'''
            MODEL_PATH = f'/data1/liuhengfa/my_own_code_for_saliency/test_for_tuning_old_dataset/save_model_dataset/test_4_my_model_ep{epoch}.pkl'  # 这个要根据train函数的更改；
            model_haptic_encoder_path = f'/data1/liuhengfa/my_own_code_for_saliency/test_for_tuning_old_dataset/save_model_dataset/test_4_my_haptic_encoder_model_ep{epoch}.pkl'
            model_haptic_affine_path = f'/data1/liuhengfa/my_own_code_for_saliency/test_for_tuning_old_dataset/save_model_dataset/test_4_my_haptic_affine_model_ep{epoch}.pkl'

            p = PredictSaliency(MODEL_PATH, model_haptic_encoder_path, model_haptic_affine_path)
        # predict all sequences
            p.predict_sequences()
            print('完成一次预测，即生成对应的显著性图')

            '''请注意力，这里还要修改一个函数，即评价函数里面的 cal_all_map函数的地址'''
            avgAUCj, avgAUCb, avgNSS, avgCC, avgKLD, avgSIM, avgSAUC = evaluate_saliency_maps_2(epoch, gt_saliency_path, gt_fix_path, pd_path)
            print('xxxxxxxxx------------------执行一次评估，并保存打分的结果----------------xxxxxxx')

        wandb.log({
        "epoch": epoch,
        "loss_all": loss,
        'kl_loss': kl_loss,
        'cc_loss': cc_loss,
        'nss_loss': nss_loss
            # 下面这几行只有在你需要没步都测试，每步都打印的的时候才需要
    #     # "avgAUCj": avgAUCj,
    #     # "avgAUCb": avgAUCb,
    #     # "avgNSS": avgNSS,
    #     # "avgCC": avgCC,
    #     # "avgKLD": avgKLD,
    #     # 'avgSIM': avgSIM,
    #     # 'avgSAUC': avgSAUC
         })
    wandb.finish()
