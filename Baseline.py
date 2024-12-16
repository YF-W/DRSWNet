import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as resnet_model
import torch.fft as fft
import numpy as np


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AdaptiveGaussianFilter(nn.Module):
    def __init__(self):

        super(AdaptiveGaussianFilter, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(1.0))

    def forward(self, X_fft, shape):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        y, x = torch.meshgrid(torch.arange(rows, device=X_fft.device),
                              torch.arange(cols, device=X_fft.device), indexing='ij')
        y = y - crow
        x = x - ccol

        gaussian_filter = torch.exp(-((x ** 2 + y ** 2) / (2 * self.sigma ** 2)))
        return X_fft * gaussian_filter

class AdaptiveDirectionalFilter(nn.Module):
    def __init__(self, num_directions=4):

        super(AdaptiveDirectionalFilter, self).__init__()
        self.num_directions = num_directions
        self.theta = nn.Parameter(torch.linspace(0, 2 * np.pi, num_directions))
        self.bandwidth = nn.Parameter(torch.tensor(30.0))

    def forward(self, X_fft, shape):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        y, x = torch.meshgrid(torch.arange(rows, device=X_fft.device),
                              torch.arange(cols, device=X_fft.device), indexing='ij')
        y = y - crow
        x = x - ccol

        angle = torch.atan2(y, x)


        filtered_fft = torch.zeros_like(X_fft)
        for i in range(self.num_directions):
            directional_filter = torch.exp(-((angle - self.theta[i]) ** 2) / (2 * self.bandwidth ** 2))
            filtered_fft += X_fft * directional_filter

        return filtered_fft


class ATF(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, num_directions=4):

        super(ATF, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(input_channels * 3, output_channels, kernel_size,
                              padding=kernel_size // 2)
        self.conv_1 = nn.Conv2d(input_channels, output_channels, kernel_size,
                              padding=kernel_size // 2)

        self.gaussian_filter = AdaptiveGaussianFilter()

        self.directional_filter = AdaptiveDirectionalFilter(num_directions)

    def forward(self, x):
        x = x.float()

        batch_size, _, rows, cols = x.shape
        shape = (rows, cols)

      #  original_x = x


        X_fft = torch.fft.fft2(x, dim=(-2, -1))
        X_fft_shift = torch.fft.fftshift(X_fft, dim=(-2, -1))


        high_freq = self.gaussian_filter(X_fft_shift, shape)
        low_freq = X_fft_shift - high_freq

        directional_freq = self.directional_filter(X_fft_shift, shape)

        high_freq_spatial = torch.fft.ifftshift(high_freq, dim=(-2, -1))
        high_freq_spatial = torch.fft.ifft2(high_freq_spatial, dim=(-2, -1)).real

        low_freq_spatial = torch.fft.ifftshift(low_freq, dim=(-2, -1))
        low_freq_spatial = torch.fft.ifft2(low_freq_spatial, dim=(-2, -1)).real

        directional_freq_spatial = torch.fft.ifftshift(directional_freq, dim=(-2, -1))
        directional_freq_spatial = torch.fft.ifft2(directional_freq_spatial, dim=(-2, -1)).real

        combined_features = torch.cat((high_freq_spatial, low_freq_spatial, directional_freq_spatial), dim=1)

        out = self.conv(combined_features)

        return out


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UPConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DepthwiseSeparableConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()

        img_size = pair(img_size)
        patch_size = pair(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        x = x + self.pos_embed
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.h = DC_CRD(12)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        H = self.h(attn)
        out = torch.matmul(attn + H, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0., upsample_dim=None):
        super().__init__()

        self.patch_embedding = PatchEmbed(img_size=image_size, patch_size=patch_size, in_c=channels, embed_dim=dim)

        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False)
        self.conv_head = nn.Conv2d(dim, num_classes, kernel_size=1)

        if upsample_dim is not None:
            self.final_upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=upsample_dim, stride=upsample_dim, padding=0)
        else:
            self.final_upsample = nn.Identity()

    def forward(self, img):
        x = self.patch_embedding(img)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        new_dim = int((x.shape[1]) ** 0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=new_dim, w=new_dim)

        x = self.upsample(x)
        x = self.conv_head(x)

        x = self.final_upsample(x)
        return x

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):

        support = torch.mm(self.weight, features)

        output = torch.spmm(adj, support)

        if active:
            output = F.relu(output)
        return output

class DC_CRD(nn.Module):
    def __init__(self, num_channels):
        super(DC_CRD, self).__init__()
        self.num_channels = num_channels

        self.theta = nn.Parameter(torch.randn(num_channels, num_channels))

        self.gnn_layer = GNNLayer(num_channels, num_channels)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)

        batch_size = x.size(0)
        adj_matrices = []

        for batch in range(batch_size):
            w_batch = w[batch]
            diff = w_batch.unsqueeze(1) - w_batch.unsqueeze(0)
            T_w = torch.abs(1 - torch.exp(-diff) / (1 + torch.exp(-diff))) - 1
            adj_matrix = (T_w + T_w.T) / 2 * self.theta
            adj_matrices.append(adj_matrix)

        A = torch.stack(adj_matrices, dim=0)

        output = torch.zeros_like(x)

        for batch in range(batch_size):
            x_batch = x[batch]
            A_batch = A[batch]

            x_flat = x_batch.view(self.num_channels, -1)

            H = self.gnn_layer(x_flat, A_batch)

            output[batch] = H.view(self.num_channels, x_batch.size(1), x_batch.size(2))

        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.adjust_channels_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.adjust_channels_2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x):

        x_1 = self.conv1x1(x)
        x_1 = self.bn1(x_1)

        x_2 = self.conv1(x)
        x_2 = self.bn2(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.conv2(x_2)
        x_2 = self.bn3(x_2)
        x_2 = self.relu(x_2)

        out = torch.cat((x_1, x_2), dim=1)
        out = self.relu(out)


        x = self.adjust_channels_1(x)
        out = self.adjust_channels_2(out)

        out += x

        return out


class MDDGNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, small_kernel=5, small_padding=2, channels=[32, 64, 128, 256, 512]):
        super(MDDGNet, self).__init__()

        resnet = resnet_model.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.fft_1_1 = ATF(in_channels, channels[0])
        self.down_1_1 = DownConv(channels[0], channels[0])
        self.fft_1_2 = ATF(channels[0], channels[1])
        self.down_1_2 = DownConv(channels[1], channels[1])
        self.fft_1_3 = ATF(channels[1], channels[2])
        self.down_3_1 = DownConv(channels[2], channels[2])

        self.dc_2_1 = ResidualBlock(in_channels, channels[0])
        self.down_2_1 = DownConv(channels[0], channels[0])
        self.dc_2_2 = ResidualBlock(channels[0], channels[1])
        self.down_2_2 = DownConv(channels[1], channels[1])
        self.dc_2_3 = ResidualBlock(channels[1], channels[2])
        self.down_3_2 = DownConv(channels[2], channels[2])

        self.graphvit = ViT(image_size=28, patch_size=2, channels=512, dim=768,
                       num_classes=channels[4], depth=3, heads=12, mlp_dim=512, dropout=0.1, emb_dropout=0.1,
                       upsample_dim=None)

        self.bottleneck_1 = DoubleConv(channels[4], channels[4])

        self.up_3_1 = UPConv(channels[3], channels[2])
        self.up_3_2 = UPConv(channels[3], channels[2])
        self.dc_1_1 = DoubleConv(channels[3], channels[2], small_kernel, small_padding)
        self.dc_2_4 = ResidualBlock(channels[3], channels[2])

        self.down_3_3 = DownConv(channels[2], channels[2])
        self.down_3_4 = DownConv(channels[2], channels[2])

        self.bottleneck_2 = DoubleConv(channels[4], channels[4])

        self.up_3_3 = UPConv(channels[3], channels[2])
        self.up_3_4 = UPConv(channels[3], channels[2])
        self.fft_1_4 = ATF(channels[3] + 128, channels[1])
        self.dc_2_5 = ResidualBlock(channels[3] + 128, channels[1])

        self.up_1_1 = UPConv(channels[1], channels[1])
        self.fft_1_5 = ATF(channels[2] + 64, channels[0])
        self.up_1_2 = UPConv(channels[0], channels[0])
        self.fft_1_6 = ATF(channels[1] + 64, channels[0])

        self.up_2_1 = UPConv(channels[1], channels[1])
        self.dc_2_6 = ResidualBlock(channels[2] + 64, channels[0])
        self.up_2_2 = UPConv(channels[0], channels[0])
        self.dc_2_7 = ResidualBlock(channels[1] + 64, channels[0])

        self.finalConv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.dws = DepthwiseSeparableConv(channels[0] * 2, channels[0])


    def forward(self, x, channels=[32, 64, 128, 256, 512]):
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        x_1_1 = self.fft_1_1(x)
        x_1_1_down = self.down_1_1(x_1_1)
        x_1_2 = self.fft_1_2(x_1_1_down)
        x_1_2_down = self.down_1_2(x_1_2)
        x_1_3 = self.fft_1_3(x_1_2_down)
        x_3_1_down = self.down_3_1(x_1_3)

        x_2_1 = self.dc_2_1(x)
        x_2_1_down = self.down_2_1(x_2_1)
        x_2_2 = self.dc_2_2(x_2_1_down)
        x_2_2_down = self.down_2_2(x_2_2)
        x_2_3 = self.dc_2_3(x_2_2_down)
        x_3_2_down = self.down_3_2(x_2_3)

        x_bottleneck_1 = torch.cat((x_3_1_down, x_3_2_down, e3), dim=1)
        x_bottleneck_1 = self.bottleneck_1(x_bottleneck_1)

        x_up_3_1, x_up_3_2 = torch.split(x_bottleneck_1, channels[3], dim=1)
        x_up_3_1_up = self.up_3_1(x_up_3_1)
        x_up_3_2_up = self.up_3_2(x_up_3_2)
        x_up_1_1 = self.dc_1_1(torch.cat((x_up_3_1_up, x_1_3), dim=1))
        x_up_2_1 = self.dc_2_4(torch.cat((x_up_3_2_up, x_2_3), dim=1))

        x_3_3_down = self.down_3_3(x_up_1_1)
        x_3_4_down = self.down_3_4(x_up_2_1)

        x_bottleneck_2 = torch.cat((x_3_3_down, x_3_4_down, e3), dim=1)
        x_bottleneck_2 = self.bottleneck_2(x_bottleneck_2)

        x_up_3_3, x_up_3_4 = torch.split(x_bottleneck_2, channels[3], dim=1)
        x_up_3_3_up = self.up_3_3(x_up_3_3)
        x_up_3_4_up = self.up_3_4(x_up_3_4)
        x_up_1_4 = self.fft_1_4(torch.cat((x_up_3_3_up, x_up_1_1, e2), dim=1))
        x_up_2_5 = self.dc_2_5(torch.cat((x_up_3_4_up, x_up_2_1, e2), dim=1))
        x_up_1_2_up = self.up_1_1(x_up_1_4)
        x_up_2_2_up = self.up_2_1(x_up_2_5)

        x_up_1_3 = self.fft_1_5(torch.cat((x_up_1_2_up, x_1_2, e1), dim=1))
        x_up_2_3 = self.dc_2_6(torch.cat((x_up_2_2_up, x_2_2, e1), dim=1))
        x_up_1_3_up = self.up_1_2(x_up_1_3)
        x_up_2_3_up = self.up_2_2(x_up_2_3)

        e0 = F.interpolate(e0, scale_factor=2, mode='bilinear', align_corners=True)
        x_up_1_4 = self.fft_1_6(torch.cat((x_up_1_3_up, x_1_1, e0), dim=1))
        x_up_2_4 = self.dc_2_7(torch.cat((x_up_2_3_up, x_2_1, e0), dim=1))

        temp = torch.cat((x_up_1_4, x_up_2_4), dim=1)
        temp = self.dws(temp)
        x = x_up_1_4 + x_up_2_4 + temp
        x = self.finalConv(x)

        return x

if __name__ == "__main__":
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

    x = torch.randn(4, 3, 224, 224)
    model = MDDGNet()
    out = model(x)
    print(out.shape)
