import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from config import config
from model import ResNet18

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

def get_dataloaders():
    # CIFAR-10数据预处理 compose()串联pipeline
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # 归一化[0,1]
        transforms.ToTensor(),
        # 标准化 加速收敛
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader, CIFAR10_CLASSES

def train():
    # 获取数据
    train_loader, test_loader, class_names = get_dataloaders()
    print(f"数据集：CIFAR-10")
    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"测试样本: {len(test_loader.dataset)}")
    print(f"类别: {class_names}")

    # 初始化模型
    model = ResNet18(config.num_classes)
    model.to(config.device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                          momentum=config.momentum, weight_decay=config.weight_decay)

    # 每隔step_size的epochs，learning rate * gamma
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    best_acc = 0
    train_losses = []
    test_accuracies = []

    print("\n training starts")
    for epoch in range(config.num_epochs):
        start_time = time.time()

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            #1 inputs & targets移动到指定设备
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            #2 每个batch之前进行梯度清零
            optimizer.zero_grad()
            #3 前向传播
            outputs = model(inputs)
            #4 损失函数
            loss = criterion(outputs, targets)
            #5 反向传播
            loss.backward()
            #6 更新参数
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        # 测试阶段
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        # 计算指标
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        epoch_time = time.time() - start_time

        # 记录历史
        train_losses.append(train_loss / len(train_loader))
        test_accuracies.append(test_acc)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f'Epoch {epoch + 1:3d}/{config.num_epochs} | '
              f'Time: {epoch_time:.2f}s | '
              f'LR: {current_lr:.4f} | '
              f'Train Loss: {train_loss / len(train_loader):.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Acc: {test_acc:.2f}%')

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config.__dict__
            }, config.model_path)
            print(f'✅ 保存最佳模型，准确率: {test_acc:.2f}%')

    print(f'\n🎉 训练完成！最佳测试准确率: {best_acc:.2f}%')
    print(f'💾 模型已保存至: {config.model_path}')


if __name__ == '__main__':
    train()