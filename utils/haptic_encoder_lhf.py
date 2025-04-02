
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer





class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, feature_dim, n_heads, n_layers):
        """
        :param input_dim: 每帧输入的特征维度（每秒3200点降维到256）
        :param feature_dim: 输出特征的维度 x
        :param n_heads: 自注意力头数
        :param n_layers: Transformer编码层数量
        """
        super(TimeSeriesTransformer, self).__init__()
        self.fc_in = nn.Linear(3200, input_dim)  # 输入降维：3200点 -> input_dim
        encoder_layer = TransformerEncoderLayer(d_model=input_dim, nhead=n_heads)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(input_dim, feature_dim)  # 输出调整：input_dim -> feature_dim

    def forward(self, x):
        """
        :param x: 输入形状 (batch_size, num_frames, 3200)
        :return: 输出形状 (batch_size, num_frames, feature_dim)
        """
        x = torch.relu(self.fc_in(x))  # (batch_size, num_frames, input_dim)
        x = self.transformer(x)  # (batch_size, num_frames, input_dim)
        x = self.fc_out(x)  # (batch_size, num_frames, feature_dim)
        return x



def create_data_packet(in_data, frame_number):

    # 处理输入的音频或视频数据，生成一个固定帧数的数据包
    # 输入的是特征和帧数，输出的是数据包和有效帧数
    n_frame = in_data.shape[0]#n_frame的值是输入数据in_data的第一个维度的大小：帧的数量
    # print("n_frame:", n_frame)
    '''看看这个创建函数的是干嘛的'''
    print("输入的创建数据包的函数的形状是什么in_data shape:", in_data.shape)  # [23, 64, 64],这里的23是指视频的帧数

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
    print('在没有经过复制之前的data_pack:', data_pack.shape)  # [16,64,64]
    data_pack = np.tile(data_pack, (3, 1, 1, 1))
    print("输出的创建数据包的函数的形状是什么data_pack shape:", data_pack.shape)  # [3, 16, 64, 64]

    return data_pack, frame_number

def get_haptic(in_data, frame_number):

    # 处理输入的音频或视频数据，生成一个固定帧数的数据包
    # 输入的是特征和帧数，输出的是数据包和有效帧数
    n_frame = in_data.shape[0]#n_frame的值是输入数据in_data的第一个维度的大小：帧的数量
    # print("n_frame:", n_frame)
    '''看看这个创建函数的是干嘛的'''
    print("输入的创建数据包的函数的触觉形状是什么in_data shape:", in_data.shape)  # [23, 64, 64],这里的23是指视频的帧数

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
    print("输出的创建数据包的函数的触觉的形状是什么data_pack shape:", data_pack.shape)  # [3, 16, 64, 64]

    return data_pack, frame_number







def get_hapticFeature(data, feature_dim, input_dim=256, n_heads=8, n_layers=4):
    # 加载数据
    #data = load_txt_file(file_path)

    # 验证数据长度是否是3200的倍数
    if len(data) % 3200 != 0:
        raise ValueError("输入数据的长度必须是每秒点数3200的整数倍")

    # 分帧操作
    num_frames = len(data) // 3200
    frames = data.reshape(num_frames, 3200)  # 每帧3200点

    # 转换为Tensor
    frames_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)  # 添加batch维度 (1, num_frames, 3200)

    # 初始化网络
    model = TimeSeriesTransformer(input_dim=input_dim, feature_dim=feature_dim, n_heads=n_heads, n_layers=n_layers)

    # 提取特征
    with torch.no_grad():  # 禁用梯度计算
        features = model(frames_tensor)

    return features.squeeze(0).numpy()  # 去除batch维度并返回NumPy格式
