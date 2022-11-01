import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# generator
class generator(nn.Module):
    def __init__(self, input_dim=100, feats_dim=1024):
        super(generator, self).__init__()
    
        #input: [batch_size, 100]
        self.conv1 = nn.Sequential(
            nn.Linear(input_dim, feats_dim * 4 * 4, bias=False),
            nn.BatchNorm1d(feats_dim * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = self.Block(feats_dim, feats_dim//2)
        self.conv3 = self.Block(feats_dim//2, feats_dim//4)
        self.conv4 = self.Block(feats_dim//4, feats_dim//8)
        self.Gz = nn.Sequential(
            nn.ConvTranspose2d(feats_dim//8, 3, 5, 2, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weight_init)
    
    def Block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1, 4, 4) # [batch_size, 1024, 4, 4]
        x = self.conv2(x) # [batch_size, 512, 8, 8]
        x = self.conv3(x) # [batch_size, 256, 16, 16]
        x = self.conv4(x)  # [batch_size, 128, 32, 32]
        x = self.Gz(x) # [batch_size, 3, 64, 64]

        return x


class Reverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

    def grad_reverse(x, alpha):
        return Reverse.apply(x, alpha)

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, bias=False),
            nn.BatchNorm2d(512)
        )
        # labels_predictor
        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.residual3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.labels_predictor = nn.Sequential(
            nn.Linear(256*3*3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=-1)
        )
        # domain classifier
        self.dl1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LogSoftmax(dim=-1)
        )
        self.dl2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.dl3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(64*2*2, 2),
            # nn.LogSoftmax(dim=-1)
            nn.Softmax(dim=-1)
        )
        self.apply(weight_init)

    def forward(self, x, alpha):
        x = self.layer1(x)
        x1 = self.residual1(x)
        x = (x1 + self.down1(x))
        x = self.layer2(x)
        x1 = self.residual2(x)
        feat = self.relu(x1 + self.down2(x))
        x1 = self.layer3(feat)
        x = self.relu(x1 + self.residual3(x1))
        x = x.view(x.size(0), -1)
        labels = self.labels_predictor(x)
        # domain classifier
        rev_feat = Reverse.apply(feat, alpha)
        x2 = self.dl1(rev_feat)
        x2 = self.relu(x2 + self.dl2(x2))
        x2 = self.dl3(x2)
        x2 = x2.view(x2.size(0), -1)
        domains = self.domain_classifier(x2)
        return labels, domains


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)
        self.down1 = Down(32, 64)
        self.sa1 = SelfAttention(64, 14)
        self.down2 = Down(64, 64)
        self.sa2 = SelfAttention(64, 7)
      
        self.bot1 = DoubleConv(64, 128)
        self.bot2 = DoubleConv(128, 128)
        self.bot3 = DoubleConv(128, 64)

        # self.sa4 = SelfAttention(64, 7)
        self.up2 = Up(128, 32)
        self.sa5 = SelfAttention(32, 14)
        self.up3 = Up(64, 32)
        self.sa6 = SelfAttention(32, 28)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x) # 32, 28, 28
        x2 = self.down1(x1, t) # 64, 14, 14
        x2 = self.sa1(x2) 
        x3 = self.down2(x2, t) # 128, 8, 8
        x3 = self.sa2(x3)

        x3 = self.bot1(x3)
        x3 = self.bot2(x3)
        x3 = self.bot3(x3) # 128, 8, 8

        # x = self.up1(x4, x3, t)
        # x = self.sa4(x3) # 64, 8, 8
        x = self.up2(x3, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

if __name__=='__main__':
    batch = 32
    print('batch size:', batch)
    print('generator:')
    model = generator()
    x = torch.randn(batch, 100)
    out = model(x)
    print('output shape:', out.shape)

    print('DANN model')
    m = DANN()
    x = torch.randn(batch, 3, 28, 28)
    c, d = m(x, 0)
    print('class labels shape:', c.shape)
    print('domain labels shape', d.shape)