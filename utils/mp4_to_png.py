'''
import cv2
import os
import imageio

def extract_frames(video_path, output_folder, saliency_folder):
    # 读取视频文件
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    # 获取显著图文件列表
    saliency_files = sorted([f for f in os.listdir(saliency_folder) if f.endswith('.jpg')])
    num_saliency_maps = len(saliency_files)

    # 计算每张显著图对应的视频时间
    time_per_saliency = video_duration / num_saliency_maps

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saliency_index = 0
    current_time = 0


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 计算当前帧对应的时间
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # 如果当前时间超过了下一个显著图的时间，则更新显著图
        if current_time >= (saliency_index + 1) * time_per_saliency and saliency_index < num_saliency_maps - 1:
            saliency_index += 1

        # 保存当前帧为图片
        frame_filename = os.path.join(output_folder, f'img_{saliency_index + 1:05d}.png')#img_00001开始
        # frame_filename = os.path.join(output_folder, f'img_{saliency_index:05d}.png')#img_00000开始
        cv2.imwrite(frame_filename, frame)

    cap.release()

# 示例用法
video_path = r'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\video\\hys\\01\\01.mp4'
output_folder = r'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\video\\hys\\01'
saliency_folder = r'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\saliency(change_name)\\hys\\01'

extract_frames(video_path, output_folder, saliency_folder)
'''
import cv2
import os

def extract_frames(video_path, output_folder, saliency_folder):
    # 读取视频文件
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    # 获取显著图文件列表
    saliency_files = sorted([f for f in os.listdir(saliency_folder) if f.endswith('.jpg')])
    num_saliency_maps = len(saliency_files)

    # 计算每张显著图对应的视频时间
    time_per_saliency = video_duration / num_saliency_maps

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saliency_index = 0
    current_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 计算当前帧对应的时间
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # 如果当前时间超过了下一个显著图的时间，则更新显著图
        if current_time >= (saliency_index + 1) * time_per_saliency and saliency_index < num_saliency_maps - 1:
            saliency_index += 1

        # 保存当前帧为图片
        frame_filename = os.path.join(output_folder, f'img_{saliency_index + 1:05d}.png')
        cv2.imwrite(frame_filename, frame)

    cap.release()

# 遍历hys文件夹中的所有子文件夹
base_video_folder = r'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\video\\jxb'
base_saliency_folder = r'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\saliency(change_name)\\jxb'

for subfolder in os.listdir(base_video_folder):
    video_subfolder_path = os.path.join(base_video_folder, subfolder)
    if os.path.isdir(video_subfolder_path):
        video_file_path = os.path.join(video_subfolder_path, f'{subfolder}.mp4')
        saliency_subfolder_path = os.path.join(base_saliency_folder, subfolder)
        if os.path.exists(video_file_path) and os.path.exists(saliency_subfolder_path):
            extract_frames(video_file_path, video_subfolder_path, saliency_subfolder_path)