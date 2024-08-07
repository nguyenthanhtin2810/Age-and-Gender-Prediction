import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class AgeGenderCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 48x48
        self.conv1 = self.make_block(1, 32)
        # 24x24
        self.conv2 = self.make_block(32, 64)
        # 12x12
        self.conv3 = self.make_block(64, 128)
        # 6x6
        self.conv4 = self.make_block(128, 256)
        # 3x3
        self.reg_fc = nn.Sequential(
            nn.Linear(3 * 3 * 256, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1)
        )
        self.cls_fc = nn.Sequential(
            nn.Linear(3 * 3 * 256, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        reg_output = self.reg_fc(x)
        cls_output = self.cls_fc(x)
        return reg_output, cls_output

class AgeGenderResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        del self.backbone.fc

        for name, param in self.backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

        self.reg_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1)
        )
        self.cls_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2)
        )


    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        reg_output = self.reg_fc(x)
        cls_output = self.cls_fc(x)
        return reg_output, cls_output

if __name__ == '__main__':
    # model = AgeGenderCNN()
    # input_data = torch.rand(8, 1, 48, 48))
    model = AgeGenderResNet50()
    input_data = torch.rand(8, 3, 224, 224)
    if torch.cuda.is_available():
        model.cuda()
        input_data = input_data.cuda()
    while True:
        reg, cls = model(input_data)
        print(reg.shape, cls.shape)
        break
