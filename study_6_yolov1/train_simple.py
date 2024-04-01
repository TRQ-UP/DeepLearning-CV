import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from models.vgg_yolo import vgg16_bn
from models.yoloLoss import yoloLoss
from utils.dataset import yoloDataset


def create_datset(file_root, batch_size):

    # ---------------------数据读取---------------------
    train_dataset = yoloDataset(root=file_root, list_file='train.txt', train=True,
                                transform=[transforms.ToTensor()])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = yoloDataset(root=file_root, list_file='val.txt', train=False,
                               transform=[transforms.ToTensor()])

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print('the train dataset has %d images' % (len(train_dataset)))
    print('the test dataset has %d images' % (len(test_dataset)))
    print('the batch_size is %d' % batch_size)

    return train_dataset, train_loader, test_dataset, test_loader



if __name__ == "__main__":
    # 1. set args
    warnings.filterwarnings('ignore')
    use_gpu = True
    file_root= 'datasets/maskdataset/'
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 6

    # 2. 创建数据
    train_dataset, train_loader, test_dataset, test_loader = create_datset(file_root,batch_size)


    # 3. 网络选择
    net = vgg16_bn()
    #  从指定模型加载训练:将VGG16模型的特征提取部分的权重加载到当前模型中，以利用VGG16在大规模图像数据上预训练得到的特征提取能力
    vgg = models.vgg16_bn(pretrained=True)
    new_state_dict = vgg.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and k.startswith('features'):
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)

    if use_gpu:
        device = "cuda"
        print('device gpu:', device)
        net.to(device)

    # 定义损失函数
    criterion = yoloLoss(7, 2, 5, 0.5)

    # 定义优化器
    # 实现对模型不同部分使用不同学习率的设置
    params = []
    params_dict = dict(net.named_parameters())
    for key, value in params_dict.items():
        if key.startswith('features'):
            params += [{'params': [value], 'lr': learning_rate * 1}]
        else:
            params += [{'params': [value], 'lr': learning_rate}]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # Adam对GPU的显存占用更大
    # optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

    # ---------------------训练---------------------
    logfile = open('checkpoints/log.txt', 'w')
    num_iter = 0
    best_test_loss = np.inf

    for epoch in range(num_epochs):
        # train
        net.train()
        if epoch == 30:
            learning_rate = 0.0001
        if epoch == 40:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        total_loss = 0.

        for i, (images, target) in enumerate(train_loader):
            if use_gpu:
                images, target = images.cuda(), target.cuda()

            pred = net(images)
            # print(pred.shape, target.shape)
            loss = criterion(pred, target)
            total_loss += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))
                num_iter += 1

        # validation
        validation_loss = 0.0
        net.eval()
        for i, (images, target) in enumerate(test_loader):
            if use_gpu:
                images, target = images.cuda(), target.cuda()

            pred = net(images)
            loss = criterion(pred, target)
            validation_loss += loss.item()
        validation_loss /= len(test_loader)

        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(), 'checkpoints/best.pth')
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
        logfile.flush()



