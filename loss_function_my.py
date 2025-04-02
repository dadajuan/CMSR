import numpy as np
import torch
import torch.nn.functional as F

'''
这个里面的损失函数都已经经过我的多个代码的验证，是正确的

与论文20的结果一致；

值得注意的是，我们的CC和nss算出来都是负数，所以在计算的时候那个weight取正数就行了；

'''



def kldiv(s_map, gt):

    #
    # 去掉通道维度
    s_map = s_map.squeeze(1)
    gt = gt.squeeze(1)
    assert s_map.size() == gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)
    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
    assert expand_gt.size() == gt.size()

    eps = 2.2204e-16
    s_map = s_map / (expand_s_map + eps)
    gt = gt / (expand_gt + eps)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    result = gt * torch.log(eps + gt / (s_map + eps))

    return torch.mean(torch.sum(result, 1))




def logit(x):
    return np.log(x/(1-x+1e-08)+1e-08)


def sigmoid_np(x):
    return 1/(1+np.exp(-x))



def cc_score(x, y, weights, batch_average=False, reduce=True):

    # x是预测的显著性图；y是真实的fixation map
    x=x.squeeze(1)
    x = F.sigmoid(x)
    y=y.squeeze(1)
    mean_x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    mean_y = torch.mean(torch.mean(y, 1, keepdim=True), 2, keepdim=True)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    # 计算逐元素乘积
    r_num = torch.sum(torch.sum(torch.mul(xm,ym), 1, keepdim=True), 2, keepdim=True)
    # 计算xm的平方和
    r_den_x = torch.sum(torch.sum(torch.mul(xm, xm), 1, keepdim=True), 2, keepdim=True)
    # 计算ym的平方和
    r_den_y = torch.sum(torch.sum(torch.mul(ym, ym), 1, keepdim=True), 2, keepdim=True) +  np.finfo(np.float32).eps.item()
    # 计算相关系数
    r_val = torch.div(r_num, torch.sqrt(torch.mul(r_den_x,r_den_y)))
    r_val = torch.mul(r_val.squeeze(), weights)
    if batch_average:
        r_val = -torch.sum(r_val) / torch.sum(weights)
    else:
        if reduce:
            r_val = -torch.sum(r_val)
        else:
            r_val = -r_val
    return r_val


def nss_score(x, y, weights, batch_average=False, reduce=True):
    # X是预测的显著性图；Y是真实的fixation map
    x = x.squeeze(1)
    x = F.sigmoid(x)

    # 大于0的值为1，小于0的值为0
    y = y.squeeze(1)
    y = torch.gt(y, 0.0).float()
    #print('验证fix计算的对不对')
    #print('y:', y, y.shape, y.max(), y.min())
    # 计算均值，标准差，归一化
    mean_x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    std_x = torch.sqrt(torch.mean(torch.mean(torch.pow(torch.sub(x, mean_x), 2), 1, keepdim=True), 2, keepdim=True))
    x_norm = torch.div(torch.sub(x, mean_x), std_x)

    # 计算x_norm和y的逐元素乘积,并求和
    r_num = torch.sum(torch.sum(torch.mul(x_norm, y), 1, keepdim=True), 2, keepdim=True)

    #  计算 y 在第 1 和第 2 维度上的和。
    r_den = torch.sum(torch.sum(y, 1, keepdim=True), 2, keepdim=True)

    r_val = torch.div(r_num, r_den + np.finfo(np.float32).eps.item())

    r_val = torch.mul(r_val.squeeze(), weights)
    #print('r_val_1:', r_val)
    if batch_average:
        r_val = -torch.sum(r_val) / torch.sum(weights)

    else:
        if reduce:
            r_val = -torch.sum(r_val)
            #print('r_val_2:', r_val)
        else:
            r_val = -r_val
    return r_val