import os  # 导入操作系统模块，用于处理文件路径和文件存在性检查
import json  # 导入JSON模块，用于读取JSON文件
import torch  # 导入PyTorch库，用于深度学习模型
from PIL import Image  # 导入PIL库中的Image模块，用于读取和显示图像
from torchvision import transforms  # 导入torchvision库中的transforms模块，用于图像预处理
import matplotlib.pyplot as plt  # 导入matplotlib库中的pyplot模块，用于绘制图像
from model import AlexNet  # 导入自定义的AlexNet模型
from torchsummary import summary


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU（如果可用）或CPU
    
    data_transform = transforms.Compose(
        [transforms.Resize(224),  # 图像大小调整为224x224
         transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为tensor，并缩放到[0.0, 1.0]
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # 对tensor图像进行标准化
    
    img_path = "./12.jpg" # 设置图像路径
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)  # 检查文件是否存在
    img = Image.open(img_path)  # 使用PIL的Image模块打开图像
    
    plt.imshow(img)  # 使用matplotlib显示图像
    # 将图像转换为[N, C, H, W]格式，其中N为批次大小，C为通道数，H为高度，W为宽度
    img = data_transform(img)  # 对图像进行预处理
    # 扩展批次维度，因为模型期望输入是一个批次的数据
    img = torch.unsqueeze(img, dim=0)  # 在第一个维度（批次维度）上增加一个维度
    
    # 读取类别索引字典
    json_path = './class_indices.json'  # 设置类别索引字典的路径
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)  # 检查文件是否存在
    
    with open(json_path, "r") as f:  # 打开文件并读取内容
        class_indict = json.load(f)  # 使用json模块的load方法将文件内容转换为Python字典
    
    # 创建模型
    model = AlexNet(num_classes=2).to(device)  # 创建一个AlexNet模型，并将模型移动到指定的设备上（GPU或CPU）

    summary(model, (3, 224, 224))
    # 加载模型权重
    weights_path = "./AlexNet.pth"  # 设置模型权重文件的路径
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)  # 检查文件是否存在
    model.load_state_dict(torch.load(weights_path))  # 加载模型权重
    
    model.eval()  # 将模型设置为评估模式，关闭dropout和batch normalization的训练模式
    with torch.no_grad():  # 不计算梯度，节省内存和计算资源
        # 预测类别
        output = torch.squeeze(model(img.to(device))).cpu()  # 将图像数据移动到设备上，模型进行预测，然后移动到CPU上
        predict = torch.softmax(output, dim=0)  # 对输出进行softmax操作，得到每个类别的概率
        predict_cla = torch.argmax(predict).numpy()  # 获取概率最大的类别的索引
    
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],  # 格式化输出预测结果
                                                 predict[predict_cla].numpy())  # 获取预测类别的概率
    plt.title(print_res)  # 设置图像标题为预测结果
    for i in range(len(predict)):  # 遍历所有类别的概率
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],  # 打印每个类别的名称和概率
                                                  predict[i].numpy()))
    plt.show()  # 显示图像


if __name__ == '__main__':
    main()  # 运行主函数