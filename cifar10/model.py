import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sn
import torch.optim
import torchmetrics.functional
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.functional import F


class BaseModel(pl.LightningModule):
    def __init__(self, transform=None, learning_rate=1e-03, weight_decay=0.001, t_max=10, eta_min=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.example_input_array = torch.rand(1, 3, 32, 32)

        self.train_log = []
        self.val_log = []
        self.best_epoch = 0
        self.best_val = 10.0

        self.transform = transform
        self.criterion = nn.NLLLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.eta_min = eta_min

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=self.eta_min)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        if self.transform:
            x = self.transform(x)

        z = self.forward(x)
        loss = self.criterion(z, y)

        pred = torch.argmax(z, dim=1)
        acc = accuracy(pred, y)

        logs = {
            'train_loss': loss,
            'train_acc': acc
        }

        batch_dict = {
            'loss': loss,
            'log': logs,
            'correct': pred.eq(y).sum().item(),
            'total': len(y)
        }

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return batch_dict

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.forward(x)
        loss = self.criterion(z, y)

        pred = torch.argmax(z, dim=1)
        acc = accuracy(pred, y)

        logs = {
            'val_loss': loss,
            'val_acc': acc
        }

        batch_dict = {
            'loss': loss,
            'log': logs,
            'correct': pred.eq(y).sum().item(),
            'total': len(y),
            'preds': z,
            'target': y
        }

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return batch_dict

    def training_epoch_end(self, outputs):
        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = correct / total

        val_loss, val_acc = self.val_log[self.current_epoch]

        if val_loss < self.best_val:
            self.best_val = val_loss
            self.best_epoch = self.current_epoch

            self.log('best/val_loss', val_loss, on_epoch=True, prog_bar=True)
            self.log('best/val_acc', val_acc, on_epoch=True, prog_bar=True)

            self.log('best/train_loss', avg_loss, on_epoch=True, prog_bar=True)
            self.log('best/train_acc', avg_acc, on_epoch=True, prog_bar=True)

            self.log('best/epoch', self.current_epoch, on_epoch=True, prog_bar=True)

        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Accuracy/Train', avg_acc, self.current_epoch)

        # self.tb_histogram_add()

    def validation_epoch_end(self, outputs):
        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = correct / total

        preds = torch.exp(torch.cat([x['preds'] for x in outputs])).cpu()
        targets = torch.cat([x['target'] for x in outputs]).cpu()

        # confusion_matrix = pl.metrics.functional.confusion_matrix(preds, targets, num_classes=10)
        confusion_matrix = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=10, normalize='true')
        df_cm = pd.DataFrame(confusion_matrix.numpy(), index=range(10), columns=range(10))
        plt.figure(figsize=(10, 7))
        fg = sn.heatmap(df_cm, vmin=0, vmax=1, annot=True).get_figure()
        plt.close(fg)

        self.val_log.append((avg_loss, avg_acc))

        self.logger.experiment.add_scalar('Loss/Validation', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Accuracy/Validation', correct / total, self.current_epoch)

        self.logger.experiment.add_figure("Confusion matrix", fg, self.current_epoch)

    def tb_histogram_add(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)


class MLPCifar10(BaseModel):
    def __init__(self, hidden_size, hidden_cnt, dropout=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_size = hidden_size
        self.hidden_cnt = hidden_cnt
        self.dropout = dropout

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
    def __init__(self, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
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

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pooling1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pooling2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pooling3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
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

    def __init__(self, num_blocks, num_classes=10, in_block_channels=16, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.in_block_channels = in_block_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_block_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_block_channels),
            nn.ReLU()
        )

        def make_layer(out_channels, num_block, stride):
            strides = [stride] + [1] * (num_block - 1)
            layers = []
            for stride in strides:
                layers.append(ResCifar10.Block(self.in_block_channels, out_channels, stride))
                self.in_block_channels = out_channels * ResCifar10.Block.expansion
            return nn.Sequential(*layers)

        # self.layers = [make_layer(64 * (2 ** i), block, 1 if i == 0 else 2) for i, block in enumerate(num_blocks)]
        self.l1 = make_layer(in_block_channels, num_blocks[0], stride=1)
        self.l2 = make_layer(in_block_channels * 2, num_blocks[1], stride=2)
        self.l3 = make_layer(in_block_channels * 4, num_blocks[2], stride=2)
        self.l4 = make_layer(in_block_channels * 8, num_blocks[3], stride=2)

        self.projection = nn.Sequential(
            nn.Linear(in_block_channels * 8, in_block_channels * 8),
            nn.ReLU(),
            nn.Linear(in_block_channels * 8, num_classes),
            nn.LogSoftmax(dim=1)
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
    def __init__(self, num_blocks, dropout=0.3, *args, **kwargs):
        super().__init__(num_blocks, *args, **kwargs)

        self.dropout = nn.Dropout2d(dropout)

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
