import os
import torch
from torchvision import datasets, transforms


def calculate_mean_std(dataset):
    """
    计算数据集的均值和方差
    :param dataset: 数据集
    :return: mean, std
    """
    # init mean and std
    mean = 0.
    std = 0.
    total_samples = len(dataset)
    print("tran dataset len:", total_samples)
    
    # Traversing the dataset calculates the mean and std
    for data, _ in dataset:
        """
        data张量是一个代表图像的张量，通常具有三个维度：(通道, 高度, 宽度)。
        dim=(1, 2)参数指定了要在高度和宽度维度上进行求均值的操作。
        计算每个通道上的像素值的平均值，得到的结果是一个包含每个通道上的平均值的张量。
        """
        mean += torch.mean(data, dim=(1, 2))
        std += torch.std(data, dim=(1, 2))
    
    # Calculate the population mean and std
    mean /= total_samples
    std /= total_samples
    
    return mean, std


if __name__ == "__main__":
    # 定义数据转换
    transform = transforms.Compose([transforms.ToTensor()])
    # get data root path
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    # flower data set path
    image_path = os.path.join(data_root, "data_set", "flower_data")
    # 加载数据集
    dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=transform)
    # calculate mean and std
    mean, std = calculate_mean_std(dataset)
    print("Mean:", mean)
    print("Std:", std)
