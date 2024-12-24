import torch
import torch.nn as nn
import numpy as np


def data_loading(name, root_path='/home/zzb/NKDM/Data/EEG/EEG_Data/BCI_2b_1c_sn_splite/'):
    # path = root_path + name + '/pure_eeg_' + name + '.npy'
    pure_eeg = torch.from_numpy(np.load(root_path + name + '/pure_eeg_' + name + '.npy'))
    noise_eeg = torch.from_numpy(np.load(root_path + name + '/noise_eeg_' + name + '.npy'))
    label_eeg = torch.from_numpy(np.load(root_path + name + '/label_eeg_' + name + '.npy'))
    print(pure_eeg.shape, noise_eeg.shape, label_eeg.shape)
    return pure_eeg, noise_eeg, label_eeg


class DeepSeparator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=11, padding=5)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=15, padding=7)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x3 = self.act(self.conv3(x))
        x4 = self.act(self.conv4(x))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class Encoder(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.conv_f = nn.Sequential(
            nn.Conv1d(2, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU()
        )

        self.conv_t = nn.Sequential(
            # DeepSeparator(),    # b, 16, l
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1),
            nn.LeakyReLU(),
            DeepSeparator(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1),
            nn.LeakyReLU(),
            DeepSeparator(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            DeepSeparator(),
            # nn.Dropout(0.5),
            # nn.Conv1d(16, 16, 3, 2),
            # nn.LeakyReLU(),
            # nn.Conv1d(16, 128, 1),
            # nn.LeakyReLU()
        )

        self.init_conv = DeepSeparator()

        self.channel_up = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(16, 16, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv1d(16, 128, 1),
            nn.LeakyReLU()
        )

        self.channel_f = nn.Conv1d(256, 256, 1)
        # self.up_dim = nn.ConvTranspose1d(256, 256, 5, 2)
        self.up_dim = nn.Linear(500, 1000)
        self.channel_down = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 1, 1),
            nn.LeakyReLU()
        )

        self.act = nn.LeakyReLU()

    def temporal2freqence(self, x):
        x = torch.cos(x)
        x = torch.fft.rfft(x)
        x = x[:, 0: 1000 // 2]
        # x[:, int(x.shape[0] * 0.4):] = 0
        x_m = torch.unsqueeze(torch.abs(x), dim=1)
        x_p = torch.unsqueeze(torch.angle(x), dim=1)
        assert x_m.shape == x_p.shape, "x_m & x_p dim dismatch"
        return torch.cat((x_m, x_p), dim=1)

    def forward(self, x):
        x_f = self.temporal2freqence(x)
        x_f = self.conv_f(x_f)
        # print(x_f.shape)

        x_t = torch.unsqueeze(x, dim=1)
        x_t = self.init_conv(x_t)
        x_rt = x_t.clone()
        x_t = self.conv_t(x_t)
        x_t += x_rt
        x_t = self.channel_up(x_t)
        # print(x_t.shape)
        # print(x_f.shape)

        x = torch.cat((x_t, x_f), dim=1)
        x = self.act(self.channel_f(x))
        x = self.act(self.up_dim(x))
        x = self.channel_down(x)
        x = x.squeeze()

        return x


class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # self.pool = nn.AvgPool1d(1000, 1)
        # self.l = nn.Linear(256, 2)
        self.l = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.pool(x)
        # x = x.squeeze(-1)
        x = self.l(x)
        return x


class ContrastiveMLP(nn.Module):
    def __init__(self, in_channel, out_channel, in_length, out_length):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, padding=1),
            nn.LeakyReLU(),
            nn.Linear(in_length, out_length),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.down(x)
        return x


class ContrastiveSemantic(nn.Module):
    def __init__(self, ):
        super().__init__()
        channel_list = [1, 16, 32, 64]
        length_list = [1000, 500, 250, 125]
        channel_list = list(zip(channel_list[:-1], channel_list[1:]))
        length_list = list(zip(length_list[:-1], length_list[1:]))

        param_list = list(zip(channel_list, length_list))

        self.convs = nn.ModuleList(
            [ContrastiveMLP(param[0][0], param[0][1], param[1][0], param[1][1]) for param in param_list])

        self.channel_down = nn.Sequential(
            nn.Conv1d(64, 8, 1),
            nn.LeakyReLU(),
        )

        self.l = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
        x = self.channel_down(x)
        x = x.view(x.shape[0], -1)
        x = self.l(x)

        return x


def temporal2freqence(x):
    x = torch.cos(x)
    x = torch.fft.rfft(x)
    x = x[:, 0: 1000 // 2]
    x_m = torch.abs(x)
    x_p = torch.angle(x)
    assert x_m.shape == x_p.shape, "x_m & x_p dim dismatch"
    return torch.cat((x_m, x_p), dim=-1)
    # return x_m, x_p


class ContrastiveSemanticF(nn.Module):
    def __init__(self):
        super().__init__()
        # self.norm = nn.LayerNorm(1000)
        self.norm = nn.BatchNorm1d(1000)

    def forward(self, x):
        x = temporal2freqence(x)
        x = self.norm(x)

        return x


class ContrastiveSemanticTF(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.temporalNet = ContrastiveSemantic()
        self.t2f = ContrastiveSemanticF()
        self.freqenceNet = ContrastiveSemantic()

        self.tf_fusion = nn.Sequential(
            nn.Linear(2000, 1000),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x_t = self.temporalNet(x)
        x_f = self.t2f(x)
        x_f = self.freqenceNet(x_f)
        x = torch.cat((x_t, x_f), dim=-1)
        x = self.tf_fusion(x)

        return x


# new model for contrastive learning
# use conv1d with stride to replace linear layer
class ContrastiveMLP_New(nn.Module):
    def __init__(self, in_channel, out_channel, in_length, out_length):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, padding=1),
            nn.LeakyReLU(),
            # nn.Linear(in_length, out_length),
            nn.Conv1d(out_channel, out_channel, 3, stride=2, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.down(x)
        return x


class ContrastiveSemantic_New(nn.Module):
    def __init__(self, ):
        super().__init__()
        channel_list = [1, 16, 32, 64]
        length_list = [1000, 500, 250, 125]
        channel_list = list(zip(channel_list[:-1], channel_list[1:]))
        length_list = list(zip(length_list[:-1], length_list[1:]))

        param_list = list(zip(channel_list, length_list))

        self.convs = nn.ModuleList(
            [ContrastiveMLP_New(param[0][0], param[0][1], param[1][0], param[1][1]) for param in param_list])

        self.channel_down = nn.Sequential(
            nn.Conv1d(64, 8, 1),
            nn.LeakyReLU(),
        )

        self.l = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
        x = self.channel_down(x)
        x = x.view(x.shape[0], -1)
        x = self.l(x)

        return x


class ContrastiveSemanticTF_New(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.temporalNet = ContrastiveSemantic_New()
        self.t2f = ContrastiveSemanticF()
        self.freqenceNet = ContrastiveSemantic_New()

        self.tf_fusion = nn.Sequential(
            nn.Linear(2000, 1000),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x_t = self.temporalNet(x)
        x_f = self.t2f(x)
        x_f = self.freqenceNet(x_f)
        x = torch.cat((x_t, x_f), dim=-1)
        x = self.tf_fusion(x)

        return x


if __name__ == '__main__':
    # while 1:
    from torchvision.models import resnet18
    from thop import profile
    # model = resnet18()
    model = ContrastiveSemanticTF_New()
    input = torch.randn(1, 1000) #模型输入的形状,batch_size=1
    flops, params = profile(model, inputs=(input, ))
    print(flops/1e9,params/1e6) #flops单位G，para单位M


    #     i = torch.ones(512, 1000).cuda()
    #     m = ContrastiveSemanticTF_New().cuda()
    #     o = m(i).cuda()
    # print(o.shape)
