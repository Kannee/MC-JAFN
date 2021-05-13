
from utils import *
import core


# define network

class Scnn_6(nn.Module):
    def __init__(self):
        super(Scnn_6, self).__init__()
        self.ms_channel = 4  # channel number of ms image
        self.ratio = 4  # scale ratio between ms and pan image
        self.pan_channel = 1
        self.inter_channels = 64

        self.ML1 = MLAFBlock(in_channels_m=self.ms_channel, in_channels_p=self.pan_channel)
        self.ML2 = MLAFBlock(in_channels_m=self.inter_channels, in_channels_p=self.inter_channels)
        self.ML3 = MLAFBlock(in_channels_m=self.inter_channels, in_channels_p=self.inter_channels)
        self.ML4 = MLAFBlock(in_channels_m=self.inter_channels, in_channels_p=self.inter_channels)
        self.ML5 = MLAFBlock(in_channels_m=self.inter_channels, in_channels_p=self.inter_channels)
        self.ML6 = MLAFBlock(in_channels_m=self.inter_channels, in_channels_p=self.inter_channels)
        self.reconstruct = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.ms_channel,
                      kernel_size=3, stride=1, padding=1)
        )

    def forward(self, ms, pan):
        lms = core.imresize(ms, scale=self.ratio)

        w_m, w_p, f1 = self.ML1(lms, pan)
        w_m, w_p, f2 = self.ML2(w_m, w_p)
        w_m, w_p, f3 = self.ML3(w_m, w_p)
        w_m, w_p, f4 = self.ML4(w_m, w_p)
        w_m, w_p, f5 = self.ML5(w_m, w_p)
        w_m, w_p, f6 = self.ML6(w_m, w_p)
        hms = f1 + f2 + f3 + f4 + f5 + f6
        hms = self.reconstruct(hms) + lms
        return hms

