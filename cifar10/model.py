import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.functional import F


class BaseModel(pl.LightningModule):
    def __init__(self, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform = transform
        self.criterion = nn.NLLLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-03, weight_decay=0.01)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        if self.transform:
            x = self.transform(x)

        z = self.forward(x)
        loss = self.criterion(z, y)

        pred = torch.argmax(z, dim=1)
        acc = accuracy(pred, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.forward(x)
        loss = self.criterion(z, y)

        pred = torch.argmax(z, dim=1)
        acc = accuracy(pred, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)


class MLPCifar10(BaseModel):
    def __init__(self, hidden_size, hidden_cnt, dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.hidden_cnt = hidden_cnt
        self.dropout = dropout

        self.example_input_array = torch.rand(1, 3, 32, 32)

        self.fc_1 = nn.Sequential(
            nn.Linear(32 * 32 * 3, hidden_size),
            nn.Hardswish(),
            nn.BatchNorm1d(hidden_size)
        )

        self.fc_h = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                 nn.Hardswish(),
                                                 nn.BatchNorm1d(hidden_size),
                                                 nn.Dropout(dropout))
                                   for x in range(hidden_cnt)])

        self.fc_o = nn.Sequential(
            nn.Linear(hidden_size, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc_1(out)

        for layer in self.fc_h:
            out = layer(out)

        out = self.fc_o(out)
        return out


class CNNCifar10(BaseModel):
    def __init__(self):
        super().__init__()

        self.example_input_array = torch.rand(1, 3, 32, 32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, 5, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 240),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(240, 120),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(120, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc1(torch.flatten(out, 1))
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class DCNNCifar10(BaseModel):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.rand(1, 3, 32, 32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.pooling1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pooling2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pooling3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pooling1(out)
        out = self.conv5(out)
        out = self.pooling2(out)
        out = self.conv6(out)
        out = self.pooling3(out)
        out = self.conv7(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ResCifar10(BaseModel):
    class Block(nn.Module):
        expansion = 1

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.shortcut = nn.Sequential()
            self.relu = nn.ReLU()

            if stride != 1 or in_channels != ResCifar10.Block.expansion * out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * ResCifar10.Block.expansion, kernel_size=1, stride=stride,
                              bias=False),
                    nn.BatchNorm2d(out_channels * ResCifar10.Block.expansion)
                )

        def forward(self, x):
            out = self.res(x) + self.shortcut(x)
            out = self.relu(out)
            return out

    def __init__(self, num_blocks, num_classes=10, in_channels=16):
        super().__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        def make_layer(out_channels, num_block, stride):
            strides = [stride] + [1] * (num_block - 1)
            layers = []
            for stride in strides:
                layers.append(ResCifar10.Block(self.in_channels, out_channels, stride))
                self.in_channels = out_channels * ResCifar10.Block.expansion
            return nn.Sequential(*layers)

        # self.layers = [make_layer(64 * (2 ** i), block, 1 if i == 0 else 2) for i, block in enumerate(num_blocks)]
        self.l1 = make_layer(in_channels, num_blocks[0], stride=1)
        self.l2 = make_layer(in_channels * 2, num_blocks[1], stride=2)
        self.l3 = make_layer(in_channels * 4, num_blocks[2], stride=2)
        self.l4 = make_layer(in_channels * 8, num_blocks[3], stride=2)

        self.projection = nn.Sequential(
            nn.Linear(in_channels * 8, in_channels * 8),
            nn.ReLU(),
            nn.Linear(in_channels * 8, num_classes),
            nn.LogSoftmax()
        )

    def forward(self, x):
        out = self.conv1(x)

        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)

        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        out = self.projection(out)
        return out


class ResV2Cifar10(ResCifar10):
    def __init__(self, num_blocks, num_classes=10, in_channels=16):
        super().__init__(num_blocks, num_classes, in_channels)

        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        out = self.conv1(x)

        out = self.l1(out)
        out = self.l2(out)
        out = self.dropout(out)
        out = self.l3(out)
        out = self.l4(out)

        out = F.avg_pool2d(out, 4)
        out = self.dropout(out)

        out = torch.flatten(out, 1)
        out = self.projection(out)
        return out
