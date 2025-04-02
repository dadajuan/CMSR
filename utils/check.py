import torch

# 加载检查点
checkpoint_path = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\AVS360_ep25.pkl'
checkpoint = torch.load(checkpoint_path, map_location='cpu')  # map_location='cpu' 确保你在任何设备上都可以查看

# 检查 checkpoint 的类型
if isinstance(checkpoint, dict):
    # 如果是字典，打印出字典的 keys
    print("Checkpoint 是字典，包含以下 keys：", checkpoint.keys())
    if 'state_dict' in checkpoint:
        print("它包含一个 'state_dict'，这是一个状态字典。")
    else:
        print("这是其他类型的字典，但不包含 'state_dict'。")
else:
    # 如果 checkpoint 不是字典，它可能是整个模型
    print(f"Checkpoint 是一个 {type(checkpoint)} 类型，可能是整个模型。")
