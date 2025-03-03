from torch import nn


class CNN2D_3L(nn.Module):
    """Model for human-activity-recognition."""

    def __init__(self, DB, num_classes, win_size=32, cnn_channel=256):
        super().__init__()
        self.win_size=32
        self.cnn_channel=64
        # Extract features, 1D conv layers
        kernal = (5, 1)

        self.features = nn.Sequential(
            # nn.Conv2d(1, cnn_channel, kernel_size=kernal),
            nn.Conv2d(DB, self.cnn_channel, kernel_size=kernal),
            nn.GroupNorm(4, self.cnn_channel),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(self.cnn_channel, self.cnn_channel, kernel_size=kernal),
            nn.GroupNorm(4, self.cnn_channel),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(self.cnn_channel, self.cnn_channel, kernel_size=kernal),
            nn.GroupNorm(4, self.cnn_channel),
            nn.ReLU(),
            # nn.AdaptiveMaxPool2d((4, input_channel))
        )
        fc_input_size = (self.win_size - 7) // 1 * self.cnn_channel 
        # fc_input_size = (self.win_size - 28) // 4* self.cnn_channel 
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # input size: (batch_size, 1, channel, win)
        x = x.permute(0, 3, 2, 1)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        out = self.classifier(x)

        return out,x

class build_Conv_Boost:
    def __init__(self,in_chann=3, is_remix=False):
        self.is_remix = is_remix
        self.in_chann=in_chann

    def build(self, num_classes):
        return CNN2D_3L(num_classes=num_classes,DB=self.in_chann)