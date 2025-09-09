import torch
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from config import config
from model import ResNet18

# CIFAR-10类别名称
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')


def load_model():
    """加载训练好的模型"""
    model = ResNet18(config.num_classes)
    checkpoint = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    return model


def predict_image(image_path, model):
    """预测单张图像"""
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(config.device)

    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return image, predicted.item(), confidence.item(), probabilities[0].cpu().numpy()


def predict_test_set(model, test_loader):
    """在整个测试集上评估模型"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def visualize_prediction(image, prediction, confidence, probabilities, class_names=CIFAR10_CLASSES):
    """可视化预测结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 显示图像
    ax1.imshow(image)
    ax1.set_title(f'预测: {class_names[prediction]}\n置信度: {confidence:.2%}')
    ax1.axis('off')

    # 显示概率分布
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probabilities, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.invert_yaxis()
    ax2.set_xlabel('概率')
    ax2.set_title('类别概率分布')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10图像分类预测')
    parser.add_argument('--image', type=str, help='图像文件路径')
    parser.add_argument('--test', action='store_true', help='在测试集上评估模型')
    args = parser.parse_args()

    # 加载模型
    print("[表情] 加载模型...")
    model = load_model()
    print("[表情] 模型加载完成")

    if args.test:
        # 在测试集上评估
        _, test_loader, _ = get_dataloaders()
        accuracy = predict_test_set(model, test_loader)
        print(f"[表情] 测试集准确率: {accuracy:.2f}%")

    elif args.image:
        # 预测单张图像
        image, prediction, confidence, probabilities = predict_image(args.image, model)

        print(f"\n[表情] 图像: {args.image}")
        print(f"[表情] 预测类别: {CIFAR10_CLASSES[prediction]}")
        print(f"[表情] 置信度: {confidence:.2%}")

        # 显示前3个最可能的类别
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        print("\n[表情] 最可能的前3个类别:")
        for i, idx in enumerate(top3_indices):
            print(f"{i + 1}. {CIFAR10_CLASSES[idx]}: {probabilities[idx]:.2%}")

        # 可视化结果
        visualize_prediction(image, prediction, confidence, probabilities)

    else:
        print("请指定要预测的图像文件或使用 --test 在测试集上评估")


def get_dataloaders():
    """获取数据加载器"""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return None, test_loader, CIFAR10_CLASSES


if __name__ == '__main__':
    main()
