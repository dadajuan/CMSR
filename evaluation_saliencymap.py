import numpy as np
import glob
import os
import cv2
from skimage.transform import resize
from numpy import random
import pandas as pd
import math
import re

EPSILON = np.finfo('float').eps

def normalize(x, method='standard', axis=None):
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res

def KLD(p, q):
    p = normalize(p, method='sum')
    q = normalize(q, method='sum')
    return np.sum(np.where(p != 0, p * np.log((p+EPSILON) / (q+EPSILON)), 0))

def CC(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3, mode='constant') # bi-cubic/nearest is what Matlab imresize() does by default
    map1 = normalize(map1, method='standard')
    map2 = normalize(map2, method='standard')
    return np.corrcoef(map1.ravel(), map2.ravel())[0,1]

def NSS(saliency_map, fixation_map):
    s_map = np.array(saliency_map, copy=False)
    f_map = np.array(fixation_map, copy=False) > 0.5

    if not np.any(f_map):
        print('no fixation to predict')
        return np.nan

    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape)

    s_map = normalize(s_map, method='standard')
    return np.mean(s_map[f_map])

def genERP(i,j,N):
    val = math.pi/N
    w = math.cos( (j - (N/2) + 0.5) * val )
    return w

def compute_map_ws(h, w):
    equ = np.zeros((h, w))
    for j in range(0,equ.shape[0]):
        for i in range(0,equ.shape[1]):
            equ[j,i] = genERP(i,j,equ.shape[0])
    equ = equ/equ.max()
    return equ

def AUC_Judd(saliency_map, fixation_map, jitter=False):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7
    saliency_map = normalize(saliency_map, method='range')
    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]
    n_fix = len(S_fix)
    n_pixels = len(S)
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)
        tp[k+1] = (k + 1) / float(n_fix)
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix)
    return np.trapz(tp, fp)

def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5

    # true_count = np.sum(fixation_map)
    # print(f"0.5_fixation_map: {true_count}")


    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    saliency_map = normalize(saliency_map, method='range')
    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]
    n_fix = len(S_fix)
    n_pixels = len(S)
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r]
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc)

# new ////////////////////////////////////////////////////////////////////////

def SIM(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3,
                      mode='constant')  # bi-cubic/nearest is what Matlab imresize() does by default
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)




#cal_all_map 函数的作用是计算所有测试集视频的注视图（fixation map），并将它们累加到一个总的注视图 fixmap_all 中。
def cal_all_map(test_set, all_map_path):
    # 定义图像的高度和宽度
    H = 224
    W = 384

    # 初始化一个全零的fixmap_all数组，大小为H x W
    fixmap_all = np.zeros([H, W], dtype=np.float32)

    # 遍历test_set中的每个视频
    for vid in test_set:
        # 构建fixation图像的路径
        # fix_root = os.path.join("/data1/liuhengfa/my_own_code_for_saliency/data/fixation_renamed/jxb", vid)
        fix_root = os.path.join(all_map_path, vid)

        # 获取目录中的所有jpg文件并排序
        files = sorted([f for f in os.listdir(fix_root) if f.endswith('.jpg')])



        # 遍历fix_root目录中的每个文件（每帧图像）
        for file_name in files:
            # 构建完整的图像路径
            file_path = os.path.join(fix_root, file_name)

            # 读取fixmap图像并转换为灰度图
            fixmap_gt = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # 调整尺寸
            fixmap_gt = cv2.resize(fixmap_gt, (W, H))

            # 将fixmap_gt中大于0的值累加到fixmap_all中
            fixmap_all += fixmap_gt > 0

    # 返回累加后的fixmap_all
    return fixmap_all

def calc_other_map(fixmap_all, fixmap2):
    # 将fixmap_all和fixmap2转换为二值图（大于0的值变为1，其他值变为0）
    fixmap_all = (fixmap_all > 0).astype(int)
    fixmap2 = (fixmap2 > 0).astype(int)

    # 计算other_map，表示fixmap_all中有而fixmap2中没有的注视点
    other_map = fixmap_all - fixmap2

    # 返回other_map
    return other_map

def AUC_shuffled(saliency_map, fixation_map, other_map , n_split=100, step_size=.1):
    # 将输入的显著图和注视图转换为numpy数组，并确保注视图是二值图
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0

#验证>0和>0.5的区别，结果一样
    # true_count = np.sum(fixation_map)
    # print(f"0_fixation_map: {true_count}")

    # 如果没有注视点，返回NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan

    # 如果显著图和注视图的形状不同，调整显著图的大小以匹配注视图
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')

    # 将显著图归一化到[0,1]范围
    saliency_map = normalize(saliency_map, method='range')

    # 将图像展平以便于处理
    S = saliency_map.ravel()
    F = fixation_map.ravel()
    Oth = other_map.ravel()

    # 获取注视点位置的显著图值
    Sth = S[F]
    Nfixations = len(Sth)

    # 获取other_map中非零值的索引
    ind = np.where(Oth > 0)[0]
    Nfixations_oth = np.min([Nfixations, len(ind)])

    # 初始化数组以存储随机注视点值
    randfix = np.zeros((Nfixations_oth, n_split)) * np.nan

    # 从other_map中生成随机注视点
    for i in range(n_split):
        randind = ind[np.random.permutation(len(ind))]
        randfix[:, i] = S[randind[:Nfixations_oth]]


    # 初始化数组以存储AUC值
    auc = np.zeros((n_split)) * np.nan

    # 计算每个随机分割的AUC
    for s in range(n_split):
        allthreshes = np.r_[0:np.max(np.r_[Sth, randfix[:, s]]):step_size][::-1]
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[0] = 0
        tp[-1] = 1
        fp[0] = 0
        fp[-1] = 1
        for k, thresh in enumerate(allthreshes):
            tp[k + 1] = np.sum(Sth >= thresh) / float(Nfixations)
            fp[k + 1] = np.sum(randfix[:, s] >= thresh) / float(Nfixations_oth)
        auc[s] = np.trapz(tp, fp)

    # 返回所有分割的平均AUC值
    return np.mean(auc)



# ////////////////////////////////////////////////////////////////////////

def get_subfolders(folder_path):
    subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    return subfolders

def evaluate_saliency_maps_2(epoch, gt_saliency_path, gt_fix_path, pd_path):

    # folder = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data'
    # pd_saliency_folder = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\output'

    #folder = 'F:\\AVS360\\new\\AVS360\\data'
    #pd_saliency_folder = 'F:\\AVS360\new\\AVS360\\output'

    #folder = '/data1/liuhengfa/my_own_code_for_saliency/data'  # 真实的数据集的地址
    #pd_saliency_folder = '/data1/liuhengfa/my_own_code_for_saliency/output/test_4_test'  # 预测的显著性图的地址

    '''这个是数据集名称,其实是没啥影响的，用什么都可以'''
    sound_type = ['jxb']

    list_vid = get_subfolders(gt_saliency_path)
    print('list_vid:', list_vid)

    list_id = list_vid

    '''这个是csv表格地址+名称'''
    csv_path = pd_path + '_score.csv'

    # stretch_map = compute_map_ws(480, 640)
    stretch_map = compute_map_ws(224, 384)  # 修改显著性图的尺寸

    AUCj_score = np.zeros((3, len(list_vid)))
    AUCb_score = np.zeros((3, len(list_vid)))
    NSS_score = np.zeros((3, len(list_vid)))
    CC_score = np.zeros((3, len(list_vid)))
    KLD_score = np.zeros((3, len(list_vid)))
    SIM_score = np.zeros((3, len(list_vid)))
    sAUC_score = np.zeros((3, len(list_vid)))

    fixmap_all = cal_all_map(list_vid, gt_fix_path)

    score_total = np.zeros((7, 3, len(list_vid), 800))

    start = True
    for v, vid_name in enumerate(list_vid[:]):
        for s, st in enumerate(sound_type[:]):

            gt_fixmap_list = [k for k in glob.glob(os.path.join( gt_fix_path, vid_name, '*.jpg'))]  # 读取所有的fixation图像

            # print('gt_fixmap_list:', gt_fixmap_list)
            gt_fixmap_list.sort()
            gt_fixmap_len = len(gt_fixmap_list)


            gt_salmap_list = [k for k in glob.glob(os.path.join(gt_saliency_path  , vid_name, '*.jpg'))]  # 读取所有的saliency图像
            # print('gt_salmap_list:', gt_salmap_list)
            gt_salmap_list.sort()
            gt_salmap_len = len(gt_salmap_list)

            AUCj_score_v = np.zeros(gt_fixmap_len)
            AUCb_score_v = np.zeros(gt_fixmap_len)
            NSS_score_v = np.zeros(gt_fixmap_len)
            CC_score_v = np.zeros(gt_fixmap_len)
            KLD_score_v = np.zeros(gt_fixmap_len)
            SIM_score_v = np.zeros(gt_fixmap_len)
            sAUC_score_v = np.zeros(gt_fixmap_len)


            pd_salmap_path = [k for k in glob.glob(os.path.join(pd_path,  vid_name, '*.jpg'))]  # 小周主要就修改了这个

            print('pd_salmap_path:', pd_salmap_path)
            pd_salmap_path.sort()

            for lnum, fpath in enumerate(gt_fixmap_list):
                gt_fixmap = cv2.imread(fpath, 0)
                print('fpath:', fpath)
                # 提取 fid 字符串中的数字部分
                # fid = ''.join(filter(str.isdigit, fpath.split('.jpg')[0].split('\\')[-1]))    # 这里是windows的路径
                fid = ''.join(filter(str.isdigit, fpath.split('.jpg')[0].split('/')[-1]))   # 这里是linux的路径
                print('fid:', fid)


                # gt_salmap = cv2.resize(cv2.imread(gt_salmap_list[lnum], 0), (640,480))
                gt_salmap = cv2.resize(cv2.imread(gt_salmap_list[lnum], 0), (384, 224))  # 这里把他们修改为了384*224
                gt_salmap = gt_salmap * stretch_map

                pd_salmap_path = [k for k in glob.glob(os.path.join(pd_path, vid_name, str(int(fid) ) + '.jpg'))][0]  # 小周改了这里，我本来是加了1的；

                # print('pd_salmap_path:', pd_salmap_path)
                print('xxxx----xxxxx-----指标的计算和评估-----xxxxxx-----xxxxxx')
                print('pd_salmap_path:', pd_salmap_path)
                print('gt_fixmap:', fpath)
                print('gt_salmap:', gt_salmap_list[lnum])

                pd_salmap = cv2.imread(pd_salmap_path, cv2.IMREAD_GRAYSCALE)
                # pd_salmap = cv2.resize(pd_salmap, (640,480))

                pd_salmap = cv2.resize(pd_salmap, (384, 224))
                pd_salmap = pd_salmap * stretch_map

                gt_fixmap = cv2.resize(gt_fixmap, (384, 224))
                other_map = calc_other_map(fixmap_all, gt_fixmap)  # all the other maps of the dataset

                if pd_salmap.sum() == 0:
                    AUCj_score_v[lnum] = np.nan
                    AUCb_score_v[lnum] = np.nan
                    NSS_score_v[lnum] = np.nan
                    CC_score_v[lnum] = np.nan
                    KLD_score_v[lnum] = np.nan
                    SIM_score_v[lnum] = np.nan
                    sAUC_score_v[lnum] = np.nan

                else:
                    AUCj_score_v[lnum] = AUC_Judd(pd_salmap, gt_fixmap)
                    AUCb_score_v[lnum] = AUC_Borji(pd_salmap, gt_fixmap)
                    NSS_score_v[lnum] = NSS(pd_salmap, gt_fixmap)
                    CC_score_v[lnum] = CC(pd_salmap, gt_salmap)
                    KLD_score_v[lnum] = KLD(pd_salmap, gt_salmap)
                    SIM_score_v[lnum] = SIM(pd_salmap, gt_salmap)
                    sAUC_score_v[lnum] = AUC_shuffled(pd_salmap, gt_fixmap, other_map)

            AUCj_score[s, v] = np.nanmean(AUCj_score_v)
            AUCb_score[s, v] = np.nanmean(AUCb_score_v)
            NSS_score[s, v] = np.nanmean(NSS_score_v)
            CC_score[s, v] = np.nanmean(CC_score_v)
            KLD_score[s, v] = np.nanmean(KLD_score_v)
            SIM_score[s, v] = np.nanmean(SIM_score_v)
            sAUC_score[s, v] = np.nanmean(sAUC_score_v)

            score_total[0, s, v, 0:len(AUCj_score_v)] = AUCj_score_v
            score_total[1, s, v, 0:len(NSS_score_v)] = NSS_score_v
            score_total[2, s, v, 0:len(CC_score_v)] = CC_score_v
            score_total[3, s, v, 0:len(KLD_score_v)] = KLD_score_v
            score_total[4, s, v, 0:len(AUCb_score_v)] = AUCb_score_v
            score_total[5, s, v, 0:len(SIM_score_v)] = SIM_score_v
            score_total[6, s, v, 0:len(sAUC_score_v)] = sAUC_score_v


            pdData = {
                "epoch": epoch,
                "ID": str(list_id[v]),
                "file": vid_name,
                "sound": st,
                "avgAUCj": np.nanmean(AUCj_score_v),
                "avgAUCb": np.nanmean(AUCb_score_v),
                "avgNSS": np.nanmean(NSS_score_v),
                "avgCC" : np.nanmean(CC_score_v),
                "avgKLD": np.nanmean(KLD_score_v),
                "avgSIM": np.nanmean(SIM_score_v),
                "avgSAUC": np.nanmean(sAUC_score_v),

                "maxAUCj": np.nanmax(AUCj_score_v),
                "minAUCj": np.nanmin(AUCj_score_v),
                "stdAUCj": np.nanstd(AUCj_score_v),

                "maxAUCb": np.nanmax(AUCb_score_v),
                "minAUCb": np.nanmin(AUCb_score_v),
                "stdAUCb": np.nanstd(AUCb_score_v),

                "maxNSS": np.nanmax(NSS_score_v),
                "minNSS": np.nanmin(NSS_score_v),
                "stdNSS": np.nanstd(NSS_score_v),

                "maxCC": np.nanmax(CC_score_v),
                "minCC": np.nanmin(CC_score_v),
                "stdCC": np.nanstd(CC_score_v),

                "maxKLD": np.nanmax(KLD_score_v),
                "minKLD": np.nanmin(KLD_score_v),
                "stdKLD": np.nanstd(KLD_score_v),

                "maxSIM": np.nanmax(SIM_score_v),
                "minSIM": np.nanmin(SIM_score_v),
                "stdSIM": np.nanstd(SIM_score_v),

                "maxSAUC": np.nanmax(sAUC_score_v),
                "minSAUC": np.nanmin(sAUC_score_v),
                "stdSAUC": np.nanstd(sAUC_score_v)

            }
            pdData = pd.DataFrame([pdData], index=None)
            if start:
                pdData.to_csv(csv_path, header=True, mode='a')
                start = False
            else:
                pdData.to_csv(csv_path, header=False, mode='a')

    # AUCj_score[AUCj_score==0.0] = np.nan
    # AUCb_score[AUCb_score==0.0] = np.nan
    # NSS_score[NSS_score==0.0] = np.nan
    # CC_score[CC_score==0.0] = np.nan
    # KLD_score[KLD_score==0.0] = np.nan
    #
    # print('AUCj_score:', AUCj_score)
    # allvid_avgAUCj = np.nanmean(AUCj_score, axis=1)
    # allvid_avgAUCb = np.nanmean(AUCb_score, axis=1)
    # allvid_avgNSS = np.nanmean(NSS_score, axis=1)
    # allvid_avgCC = np.nanmean(CC_score, axis=1)
    # allvid_avgKLD = np.nanmean(KLD_score, axis=1)

    # return allvid_avgAUCj, allvid_avgAUCb, allvid_avgNSS, allvid_avgCC, allvid_avgKLD
    return np.nanmean(AUCj_score_v), np.nanmean(AUCb_score_v), np.nanmean(NSS_score_v), np.nanmean(CC_score_v), np.nanmean(KLD_score_v), np.nanmean(SIM_score_v), np.nanmean(sAUC_score_v)


def evaluate_saliency_maps(epoch):

    # folder = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data'
    # pd_saliency_folder = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\output'

    #folder = 'F:\\AVS360\\new\\AVS360\\data'
    #pd_saliency_folder = 'F:\\AVS360\new\\AVS360\\output'

    folder = '/data1/liuhengfa/my_own_code_for_saliency/data'  # 真实的数据集的地址
    pd_saliency_folder = '/data1/liuhengfa/my_own_code_for_saliency/output/test_4_test'


    sound_type = ['test_4']

    list_vid = ['001', '03', '007', '07']
    list_id = ['001', '03', '007', '07']
    fps_list = [30, 30, 30, 30]

    # list_vid = ['oegasz59U7I', 'Oue_XEKHq3g', 'OZOaN_5ymrc', 'RbgxpagCY_c_2']
    # list_id = ['oegasz59U7I', 'Oue_XEKHq3g', 'OZOaN_5ymrc', 'RbgxpagCY_c_2']
    # fps_list = [59.94, 30, 29.97, 30]


    gt_saliency_folder = folder + '//saliency_renamed//'
    gt_fixation_folder = folder + '//fixation_renamed//'
    csv_path = pd_saliency_folder + 'AVS360_score_test.csv'

    # stretch_map = compute_map_ws(480, 640)
    stretch_map = compute_map_ws(224, 384)  # 修改显著性图的尺寸

    AUCj_score = np.zeros((3, len(list_vid)))
    AUCb_score = np.zeros((3, len(list_vid)))
    NSS_score = np.zeros((3, len(list_vid)))
    CC_score = np.zeros((3, len(list_vid)))
    KLD_score = np.zeros((3, len(list_vid)))

    score_total = np.zeros((5, 3, len(list_vid), 800))

    start = True
    for v, vid_name in enumerate(list_vid[:]):
        print('v', v, 'vid_name:', vid_name)  # vid name就是子文件的名字；
        for s, st in enumerate(sound_type[:]):
            # print('vid_name:', vid_name, 'st:', st)
            print('st:', st,'s', s, 'vid_name:') # 0, test4
            gt_fixmap_list = [k for k in glob.glob(os.path.join(gt_fixation_folder, st, vid_name, '*.jpg'))] # 读取所有的fixation图像
            print('gt_fixmap_list:', gt_fixmap_list)
            gt_fixmap_list.sort()
            gt_fixmap_len = len(gt_fixmap_list)

            gt_salmap_list = [k for k in glob.glob(os.path.join(gt_saliency_folder, st, vid_name, '*.jpg'))] # 读取所有的saliency图像
            print('gt_salmap_list:', gt_salmap_list)
            gt_salmap_list.sort()
            gt_salmap_len = len(gt_salmap_list)

            AUCj_score_v = np.zeros(gt_fixmap_len)
            AUCb_score_v = np.zeros(gt_fixmap_len)
            NSS_score_v = np.zeros(gt_fixmap_len)
            CC_score_v = np.zeros(gt_fixmap_len)
            KLD_score_v = np.zeros(gt_fixmap_len)

            #pd_salmap_path = [k for k in glob.glob(os.path.join(pd_saliency_folder, st+'_test', vid_name, '*.jpg'))]
            pd_salmap_path = [k for k in glob.glob(os.path.join(pd_saliency_folder,  vid_name, '*.jpg'))]
            pd_salmap_path = sorted(pd_salmap_path, key=lambda x: int(re.search(r'(\d+)\.jpg$', x).group(1)))
            # print('pd_salmap_path:', pd_salmap_path)
            #pd_salmap_path.sort()

            for lnum, fpath in enumerate(gt_fixmap_list):
                gt_fixmap = cv2.imread(fpath, 0)
                match = re.search(r'(\d{2})\.jpg$', fpath)
                if match:
                    number = match.group(1)
                    # 去掉前导零
                    number = number.lstrip('0')
                    print(number)
                fid = str(int(number) - 1)
                #print('fid的值-V1', fid)
                #fid = fpath.split('.jpg')[0].split('\\')[-1]

                # gt_salmap = cv2.resize(cv2.imread(gt_salmap_list[lnum], 0), (640,480))
                #gt_salmap = cv2.resize(cv2.imread(gt_salmap_list[lnum], 0), (320, 160)) # 这里把他们修改为了200*100
                gt_salmap = cv2.resize(cv2.imread(gt_salmap_list[lnum], 0), (384, 224))  # 这里把他们修改为了200*100
                gt_salmap = gt_salmap * stretch_map
                #pd_salmap_path = [k for k in glob.glob(os.path.join(pd_saliency_folder, st+'_test' , vid_name, str(int(fid)+1)+'.jpg'))][0]
                pd_salmap_path = [k for k in glob.glob(os.path.join(pd_saliency_folder, vid_name, str(int(fid) + 1) + '.jpg'))][0]

                # print('pd_salmap_path:', pd_salmap_path)
                print('xxxx----xxxxx-----指标的计算和评估-----xxxxxx-----xxxxxx')
                print('pd_salmap_path:', pd_salmap_path)
                print('gt_fixmap:', fpath)
                print('gt_salmap:', gt_salmap_list[lnum])

                pd_salmap = cv2.imread(pd_salmap_path, cv2.IMREAD_GRAYSCALE)
                # pd_salmap = cv2.resize(pd_salmap, (640,480))
                #pd_salmap = cv2.resize(pd_salmap, (320, 160))
                pd_salmap = cv2.resize(pd_salmap, (384, 224))
                pd_salmap = pd_salmap * stretch_map

                if pd_salmap.sum() == 0:
                    AUCj_score_v[lnum] = np.nan
                    AUCb_score_v[lnum] = np.nan
                    NSS_score_v[lnum] = np.nan
                    CC_score_v[lnum] = np.nan
                    KLD_score_v[lnum] = np.nan
                else:
                    AUCj_score_v[lnum] = AUC_Judd(pd_salmap, gt_fixmap)
                    AUCb_score_v[lnum] = AUC_Borji(pd_salmap, gt_fixmap)
                    NSS_score_v[lnum] = NSS(pd_salmap, gt_fixmap)
                    CC_score_v[lnum] = CC(pd_salmap, gt_salmap)
                    KLD_score_v[lnum] = KLD(pd_salmap, gt_salmap)

            AUCj_score[s, v] = np.nanmean(AUCj_score_v)
            AUCb_score[s, v] = np.nanmean(AUCb_score_v)
            NSS_score[s, v] = np.nanmean(NSS_score_v)
            CC_score[s, v] = np.nanmean(CC_score_v)
            KLD_score[s, v] = np.nanmean(KLD_score_v)

            score_total[0, s, v, 0:len(AUCj_score_v)] = AUCj_score_v
            score_total[1, s, v, 0:len(NSS_score_v)] = NSS_score_v
            score_total[2, s, v, 0:len(CC_score_v)] = CC_score_v
            score_total[3, s, v, 0:len(KLD_score_v)] = KLD_score_v
            score_total[4, s, v, 0:len(KLD_score_v)] = AUCb_score_v

            pdData = {
                "epoch": epoch,
                "ID": str(list_id[v]),
                "file": vid_name,
                "sound": st,
                "avgAUCj": np.nanmean(AUCj_score_v),
                "avgAUCb": np.nanmean(AUCb_score_v),
                "avgNSS": np.nanmean(NSS_score_v),
                "avgCC" : np.nanmean(CC_score_v),
                "avgKLD": np.nanmean(KLD_score_v),
                "maxAUCj": np.nanmax(AUCj_score_v),
                "minAUCj": np.nanmin(AUCj_score_v),
                "maxAUCb": np.nanmax(AUCb_score_v),
                "minAUCb": np.nanmin(AUCb_score_v),
                "maxNSS": np.nanmax(NSS_score_v),
                "minNSS": np.nanmin(NSS_score_v),
                "maxCC": np.nanmax(CC_score_v),
                "minCC": np.nanmin(CC_score_v),
                "maxKLD": np.nanmax(KLD_score_v),
                "minKLD": np.nanmin(KLD_score_v),
                "stdAUCj": np.nanstd(AUCj_score_v),
                "stdAUCb": np.nanstd(AUCb_score_v),
                "stdNSS": np.nanstd(NSS_score_v),
                "stdCC": np.nanstd(CC_score_v),
                "stdKLD": np.nanstd(KLD_score_v),
            }
            pdData = pd.DataFrame([pdData], index=None)
            if start:
                pdData.to_csv(csv_path, header=True, mode='a')
                start = False
            else:
                pdData.to_csv(csv_path, header=False, mode='a')

    # AUCj_score[AUCj_score==0.0] = np.nan
    # AUCb_score[AUCb_score==0.0] = np.nan
    # NSS_score[NSS_score==0.0] = np.nan
    # CC_score[CC_score==0.0] = np.nan
    # KLD_score[KLD_score==0.0] = np.nan
    #
    # print('AUCj_score:', AUCj_score)
    # allvid_avgAUCj = np.nanmean(AUCj_score, axis=1)
    # allvid_avgAUCb = np.nanmean(AUCb_score, axis=1)
    # allvid_avgNSS = np.nanmean(NSS_score, axis=1)
    # allvid_avgCC = np.nanmean(CC_score, axis=1)
    # allvid_avgKLD = np.nanmean(KLD_score, axis=1)

    # return allvid_avgAUCj, allvid_avgAUCb, allvid_avgNSS, allvid_avgCC, allvid_avgKLD
    return np.nanmean(AUCj_score_v), np.nanmean(AUCb_score_v), np.nanmean(NSS_score_v), np.nanmean(CC_score_v), np.nanmean(KLD_score_v)


if __name__ == "__main__":
    gt_saliency_path = '/data1/liuhengfa/my_own_code_for_saliency/data/saliency_renamed/train_14'
    gt_fix_path = '/data1/liuhengfa/my_own_code_for_saliency/data/fixation_renamed/train_14'

    '''这个在predict函数里找'''
    pd_path = '/data1/liuhengfa/my_own_code_for_saliency/output_14_test/test_4_test'

    evaluate_saliency_maps_2(1, gt_saliency_path, gt_fix_path, pd_path)