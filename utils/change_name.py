import os
import shutil

def rename_images(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    else:
        print(f"Output folder already exists: {output_folder}")

    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    for idx, image_file in enumerate(image_files):
        # 构建旧文件路径
        old_file_path = os.path.join(input_folder, image_file)

        # 构建新文件名和新文件路径
        new_file_name = f"fixMap_{idx+1:05d}.jpg"
        #new_file_name = f"{idx}.jpg"#eva
        new_file_path = os.path.join(output_folder, new_file_name)

        # 复制并重命名文件
        shutil.copy2(old_file_path, new_file_path)
        print(f"Copied and renamed {old_file_path} to {new_file_path}")

# 示例用法
# base_input_folder = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\fixation\\jxb'
# base_output_folder = 'C:\\Users\\jj\\Desktop\\research\\AVS360\\AVS360\\data\\fixation(change_name)\\jxb'

base_input_folder =  'F:\\AVS360\\new\AVS360\\data\\fixation(change_name)\\train_14'
base_output_folder = 'F:\\AVS360\\new\AVS360\\data\\fixation(change_name)\\train_14_new'


for subfolder in os.listdir(base_input_folder):
    input_subfolder_path = os.path.join(base_input_folder, subfolder)
    if os.path.isdir(input_subfolder_path):
        output_subfolder_path = os.path.join(base_output_folder, subfolder)
        rename_images(input_subfolder_path, output_subfolder_path)