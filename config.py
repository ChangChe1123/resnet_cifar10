import torch

class Config:
    dataset_name = "CIFAR10"

    model_name = "resnet18"
    num_classes = 10
    pretrained = False

    batch_size = 128
    learning_rate = 0.1
    num_epochs = 50
    momentum = 0.9
    weight_decay = 5e-4

    lr_scheduler = "step"
    step_size = 30
    gamma = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2

    model_path = "cifar10_resnet18.pth"
    log_file = "training_log.txt"

config = Config()