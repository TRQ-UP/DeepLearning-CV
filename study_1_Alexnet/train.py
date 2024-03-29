import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from model import AlexNet
import json
from tqdm import tqdm
import sys


def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 准备数据集
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(), # 转换为张量
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(), # 转换为张量
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}
    
    # get data root path
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    # flower data set path
    image_path = os.path.join(data_root, "kagglecatsanddogs_5340", "PetImages")
    
    train_dataset = ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    val_dataset = ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    
    
    # 定义损失函数和优化器
    model = AlexNet(num_classes=2) # 假设有5个类别
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.0005)
    train_steps = len(train_loader)
    val_num = len(val_dataset)
    
    # 训练模型
    num_epochs = 10
    best_acc = 0.0
    save_path = './AlexNet.pth'
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, num_epochs, loss)
    
        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    print("strat training...")
    main()
    print('Finished Training')

