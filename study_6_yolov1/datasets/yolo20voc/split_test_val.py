def split_txt_file(input_file_path, output_file1_path, output_file2_path, ratio):
    try:
        # 打开原始文件
        with open(input_file_path, 'r') as input_file:
            # 读取原始文件内容
            lines = input_file.readlines()
            
            # 计算切分位置
            split_index = int(len(lines) * ratio)
            
            # 将内容分配到两个文件中
            content1 = ''.join(lines[:split_index])
            content2 = ''.join(lines[split_index:])
            
            # 写入内容到第一个文件
            with open(output_file1_path, 'w') as output_file1:
                output_file1.write(content1)
            
            # 写入内容到第二个文件
            with open(output_file2_path, 'w') as output_file2:
                output_file2.write(content2)
            
            print("切分成功！")
    except Exception as e:
        print("切分失败：", e)


# 指定原始文件路径
input_file_path = "./all_img.txt"
# 指定要写入的两个文件路径
output_file1_path = "./train.txt"
output_file2_path = "./test.txt"
# 指定切分比例
ratio = 0.9  # 9:1比例

# 调用函数进行切分
split_txt_file(input_file_path, output_file1_path, output_file2_path, ratio)
