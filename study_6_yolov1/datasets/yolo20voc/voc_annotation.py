import os

import cv2


def list_files_in_folder(folder_path):
    # 检查路径是否存在并且是一个文件夹
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("指定路径不是一个有效的文件夹路径。")
        return
    
    # 遍历文件夹中的文件
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    
    return file_names

def yolo_to_voc(yolo_box, img_width, img_height):
    # Extracting YOLO box coordinates
    x_center, y_center, box_width, box_height = yolo_box

    # Converting YOLO coordinates to VOC coordinates
    x_min = max(0, (x_center - box_width / 2) * img_width)
    y_min = max(0, (y_center - box_height / 2) * img_height)
    x_max = min(img_width, (x_center + box_width / 2) * img_width)
    y_max = min(img_height, (y_center + box_height / 2) * img_height)

    return int(x_min), int(y_min), int(x_max), int(y_max)



if __name__ ==  "__main__":
    
    root_path = "/home/trq/data/Algorithm_model/DeepLearning-CV/all_mask/"
    # 指定文件夹路径
    list_file = open('all_img.txt', 'w', encoding='utf-8')
    folder_path = "JPEGImages"
    label_path = "Annotations"
    # 获取文件列表
    img_files = list_files_in_folder(root_path + folder_path)
    len_img = len(img_files)
    all_img_txt = 'all_img.txt'
    # 打印文件名字
    if img_files :
        for step, file in enumerate(img_files):
        # for file in img_files :
            image_abs = root_path + folder_path + '/' + file
            img = cv2.imread(image_abs)
            height, width, channels = img.shape
            print(step)
            img_name = file.split(".")[0]
            label_file = root_path + label_path + "/" + img_name + ".txt"
            
            # print(image_abs)
            with open(label_file, 'r') as file:
                # 读取文件内容
                content = file.readlines()
                for line in content:
                    cls_id = line.split(" ")[0]
                    x_min = line.split(" ")[1]
                    y_min = line.split(" ")[2]
                    x_max = line.split(" ")[3]
                    y_max = line.split(" ")[4]
                    b = (float(x_min), float(y_min),float(x_max), float(y_max))
                    voc_coordinates = yolo_to_voc(b, width, height)
                    # print(b)
            list_file.write(image_abs + " " + ",".join([str(a) for a in voc_coordinates]) + ',' + str(cls_id) + '\n')
    else:
        print("文件夹中没有文件")
    
    with open(all_img_txt , 'r') as file:
        # 读取文件内容
        content = file.readlines()
        for line in content:
            print(line)
        



