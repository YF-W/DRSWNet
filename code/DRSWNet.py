import torch
from torch import einsum, nn
import torch.nn.functional as Ft
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as resnet_model
import torch.fft as fft
import numpy as np
import pywt
from thop import profile


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


       # original_x = x


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

        self.h = DynamicChannelGraphConvolution(12)
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

class DC_CRD(nn.Module):
    def __init__(self, num_channels):

        super(DC_CRD, self).__init__()
        self.num_channels = num_channels

        self.theta = nn.Parameter(torch.FloatTensor(num_channels, num_channels))

        self.W = nn.Linear(self.num_channels, self.num_channels)

        torch.nn.init.xavier_uniform_(self.theta)

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

            H = F.relu(A_batch @ x_flat)

            H = self.W(H.T)

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


class WaveletTransform2D_Level1(nn.Module):
    def __init__(self, wavelet='haar'):
        super(WaveletTransform2D_Level1, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert height == width

        x_np = x.detach().cpu().numpy()
        subband_images = []

        for b in range(batch_size):
            for c in range(channels):

                coeffs2 = pywt.dwt2(x_np[b, c], wavelet=self.wavelet)
                LL, (LH, HL, HH) = coeffs2
                subbands = [LL, LH, HL, HH]
                subband_images.append(subbands)

        subband_images = np.array(subband_images)
        subband_images = torch.tensor(subband_images, dtype=x.dtype, device=x.device)
        subband_images = subband_images.reshape(batch_size, channels, 2, 2, height // 2, width // 2)


        ll_freq = subband_images[:, :, 0, 0]  # LL
        hl_freq = subband_images[:, :, 0, 1]  # HL
        lh_freq = subband_images[:, :, 1, 0]  # LH
        hh_freq = subband_images[:, :, 1, 1]  # HH

        return ll_freq, hl_freq, lh_freq, hh_freq





class CCFEA_Module(nn.Module):
    def __init__(self, in_channels, reduction_ratio=32, expansion_ratio=4):
        super(CCFEA_Module, self).__init__()
        reduced_channels = reduction_ratio
        expanded_channels = in_channels * expansion_ratio

        self.channel_compress = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.bn_compress = nn.BatchNorm2d(reduced_channels)
        self.conv_e1 = nn.Conv2d(reduced_channels, reduced_channels, kernel_size= 3,padding=1)

        self.feature_expand = nn.Conv2d(reduced_channels, expanded_channels, kernel_size=1)
        self.conv_e2 = nn.Conv2d(expanded_channels,expanded_channels,kernel_size=5,padding=2)
        self.bn_expand = nn.BatchNorm2d(expanded_channels)
        self.relu = nn.ReLU(inplace=True)

        self.channel_restore = nn.Conv2d(expanded_channels, in_channels, kernel_size=1)
        self.bn_restore = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_x = x

        x = self.channel_compress(x)
        x = self.conv_e1(x)
        x = self.bn_compress(x)
        x = self.relu(x)

        x = self.feature_expand(x)
        x = self.conv_e2(x)
        x = self.bn_expand(x)
        x = self.relu(x)

        x = self.channel_restore(x)
        x = self.bn_restore(x)
        attention_weights = self.sigmoid(x)

        enhanced_features = original_x * attention_weights

        return enhanced_features




class FeatureProcessor(nn.Module):
    def __init__(self, in_channels, wavelet='haar'):
        super(FeatureProcessor, self).__init__()
        self.CCFEA_Module = CCFEA_Module(in_channels * 2, reduction_ratio=16, expansion_ratio=4)
        self.conv_adjust = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv_ll = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=4, dilation=2)
        self.bn_ll = nn.BatchNorm2d(in_channels)
        self.conv_hh = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn_hh = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, in_channels)
        self.final_conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.wavelet = wavelet
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)


        self.ll_weight = nn.Parameter(torch.tensor(1.0))
        self.hl_weight = nn.Parameter(torch.tensor(1.0))
        self.lh_weight = nn.Parameter(torch.tensor(1.0))
        self.hh_weight = nn.Parameter(torch.tensor(1.0))


    @staticmethod
    def median_filter(tensor, kernel_size=5):

        padding = kernel_size // 2

        tensor_padded = F.pad(tensor, (padding, padding, padding, padding), mode='reflect')


        filtered = torch.zeros_like(tensor)


        for i in range(kernel_size):
            for j in range(kernel_size):

                shifted = tensor_padded[:, :, i:(i + tensor.size(2)), j:(j + tensor.size(3))]


                filtered += shifted


        filtered, _ = torch.median(filtered, dim=1, keepdim=True)

        return filtered

    def inverse_wavelet_transform(self, ll_freq, hl_freq, lh_freq, hh_freq, wavelet='haar'):

        coeffs2 = (
            ll_freq.detach().cpu().numpy(),
            (hl_freq.detach().cpu().numpy(), lh_freq.detach().cpu().numpy(), hh_freq.detach().cpu().numpy())
        )


        reconstructed = pywt.idwt2(coeffs2, wavelet)

        reconstructed = torch.tensor(reconstructed, dtype=ll_freq.dtype, device=ll_freq.device)

        return reconstructed

    def forward(self, ll_freq, hl_freq, lh_freq, hh_freq):
        hh_freq_or = hh_freq
        hl_freq_or = hl_freq
        lh_freq_or = lh_freq


        ll_freq = self.ll_weight * ll_freq
        hl_freq = self.hl_weight * hl_freq
        lh_freq = self.lh_weight * lh_freq
        hh_freq = self.hh_weight * hh_freq

        hh_freq = self.median_filter(hh_freq)

        original_channels = hh_freq_or.shape[1]
        hh_freq = hh_freq.repeat(1, original_channels, 1, 1)

        hh_freq_l1 = hh_freq + hh_freq_or

        hl_lh_freq = torch.cat((hl_freq, lh_freq), dim=1)
        hl_lh_freq = self.CCFEA_Module(hl_lh_freq)
        hl_freq_l1, lh_freq_l1 = torch.split(hl_lh_freq, hl_freq.size(1), dim=1)


        ll_freq_l1 = self.conv1(ll_freq)
        ll_freq_l1 = self.conv3(ll_freq_l1)
        ll_freq_l1 = self.conv5(ll_freq_l1)

        ll_freq_l2 = torch.cat((ll_freq_l1, ll_freq), dim=1)
        ll_freq_l2 = self.conv_adjust(ll_freq_l2)

        hl_freq_l2 = torch.cat((hl_freq_l1, hl_freq_or), dim=1)
        hl_freq_l2 = self.conv_adjust(hl_freq_l2)

        lh_freq_l2 = torch.cat((lh_freq_l1, lh_freq_or), dim=1)
        lh_freq_l2 = self.conv_adjust(lh_freq_l2)

        hh_freq = F.leaky_relu(self.conv_hh(hh_freq_l1), negative_slope=0.1)
        hh_freq_l2 = torch.tanh(self.conv_ll(hh_freq))

        reconstructed1 = self.inverse_wavelet_transform(ll_freq_l2, hl_freq_l2, lh_freq_l2, hh_freq_l2,
                                                        wavelet=self.wavelet)

        return reconstructed1


class WT_MFP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WT_MFP, self).__init__()
        self.wavelet_transform = WaveletTransform2D_Level1(wavelet='haar')
        self.feature_processor = FeatureProcessor(in_channels)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_final = nn.Conv2d(in_channels * 4, out_channels, kernel_size=3,padding=2, dilation=2)


    def forward(self, x):
        x_or = x
        ll_freq, hl_freq, lh_freq, hh_freq = self.wavelet_transform(x)
        output = self.feature_processor(ll_freq, hl_freq, lh_freq, hh_freq)
        output = x_or * output
        output = self.conv_out(output)
        ll_freq_final, hl_freq_final, lh_freq_final, hh_freq_final = self.wavelet_transform(output)
        freq_final = torch.cat((ll_freq_final, hl_freq_final, lh_freq_final, hh_freq_final), dim=1)
        output = self.conv_final(freq_final)



        return output

class DRSWNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, small_kernel=5, small_padding=2, channels=[32, 64, 128, 256, 512]):
        super(DRSWNet, self).__init__()


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
        self.down_2_1 = WT_MFP(channels[0], channels[0])
        self.dc_2_2 = ResidualBlock(channels[0], channels[1])
        self.down_2_2 = WT_MFP(channels[1], channels[1])
        self.dc_2_3 = ResidualBlock(channels[1], channels[2])
        self.down_3_2 = WT_MFP(channels[2], channels[2])

        self.graphformer_1 = DC_CRD(channels[0])
        self.graphformer_2 = DC_CRD(channels[1])
        self.graphformer_3 = DC_CRD(channels[0])
        self.graphformer_4 = DC_CRD(channels[1])
        self.graphformer_5 = DC_CRD(channels[4])
        self.graphformer_6 = DC_CRD(channels[4])


        self.bottleneck_1 = DoubleConv(channels[4], channels[4])

        self.up_3_1 = UPConv(channels[3], channels[2])
        self.up_3_2 = UPConv(channels[3], channels[2])
        self.dc_1_1 = DoubleConv(channels[3], channels[2], small_kernel, small_padding)
        self.dc_2_4 = ResidualBlock(channels[3], channels[2])


        self.down_3_3 = WT_MFP(channels[2], channels[2])
        self.down_3_4 = WT_MFP(channels[2], channels[2])


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
        x_1_1_skip = self.graphformer_1(x_1_1)
        x_1_1_down = self.down_1_1(x_1_1)
        x_1_2 = self.fft_1_2(x_1_1_down)
        x_1_2_skip = self.graphformer_2(x_1_2)
        x_1_2_down = self.down_1_2(x_1_2)
        x_1_3 = self.fft_1_3(x_1_2_down)
        x_3_1_down = self.down_3_1(x_1_3)

        x_2_1 = self.dc_2_1(x)
        x_2_1_skip = self.graphformer_3(x_2_1)
        x_2_1_down = self.down_2_1(x_2_1)
        x_2_2 = self.dc_2_2(x_2_1_down)
        x_2_2_skip = self.graphformer_4(x_2_2)
        x_2_2_down = self.down_2_2(x_2_2)
        x_2_3 = self.dc_2_3(x_2_2_down)
        x_3_2_down = self.down_3_2(x_2_3)

        x_bottleneck_1 = torch.cat((x_3_1_down, x_3_2_down, e3), dim=1)
        x_bottleneck_1 = self.graphformer_5(x_bottleneck_1)
        x_bottleneck_1 = self.bottleneck_1(x_bottleneck_1)


        x_up_3_1, x_up_3_2 = torch.split(x_bottleneck_1, channels[3], dim=1)
        x_up_3_1_up = self.up_3_1(x_up_3_1)
        x_up_3_2_up = self.up_3_2(x_up_3_2)
        x_up_1_1 = self.dc_1_1(torch.cat((x_up_3_1_up, x_1_3), dim=1))
        x_up_2_1 = self.dc_2_4(torch.cat((x_up_3_2_up, x_2_3), dim=1))


        x_3_3_down = self.down_3_3(x_up_1_1)
        x_3_4_down = self.down_3_4(x_up_2_1)


        x_bottleneck_2 = torch.cat((x_3_3_down, x_3_4_down, e3), dim=1)
        x_bottleneck_2 = self.graphformer_6(x_bottleneck_2)
        x_bottleneck_2 = self.bottleneck_2(x_bottleneck_2)


        x_up_3_3, x_up_3_4 = torch.split(x_bottleneck_2, channels[3], dim=1)
        x_up_3_3_up = self.up_3_3(x_up_3_3)
        x_up_3_4_up = self.up_3_4(x_up_3_4)
        x_up_1_4 = self.fft_1_4(torch.cat((x_up_3_3_up, x_up_1_1, e2), dim=1))
        x_up_2_5 = self.dc_2_5(torch.cat((x_up_3_4_up, x_up_2_1, e2), dim=1))
        x_up_1_2_up = self.up_1_1(x_up_1_4)
        x_up_2_2_up = self.up_2_1(x_up_2_5)

        x_up_1_3 = self.fft_1_5(torch.cat((x_up_1_2_up, x_1_2_skip, e1), dim=1))
        x_up_2_3 = self.dc_2_6(torch.cat((x_up_2_2_up, x_2_2_skip, e1), dim=1))
        x_up_1_3_up = self.up_1_2(x_up_1_3)
        x_up_2_3_up = self.up_2_2(x_up_2_3)

        e0 = F.interpolate(e0, scale_factor=2, mode='bilinear', align_corners=True)
        x_up_1_4 = self.fft_1_6(torch.cat((x_up_1_3_up, x_1_1_skip, e0), dim=1))
        x_up_2_4 = self.dc_2_7(torch.cat((x_up_2_3_up, x_2_1_skip, e0), dim=1))

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
    model = DRSWNet()
    out = model(x)
    print(out.shape)
