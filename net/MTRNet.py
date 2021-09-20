# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


class InputProjection(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, active_layer=nn.LeakyReLU):
        super(InputProjection, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            active_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        # x的维度B, C, H, W

        # 将其HW合成一维
        # x = self.projection(x).flatten(2).transpose(1, 2).contiguous()

        # 将维度转为 B, H, W, C
        x = self.projection(x).permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x



class OutputProjection(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, active_layer=None):
        super(OutputProjection, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        )
        if active_layer is not None:
            self.projection.add_module(active_layer(inplace=True))
        if norm_layer is not None:
            self.norm_layer = norm_layer(out_channel)
        else:
            self.norm_layer = None

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.projection(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x


def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)




# Feedforward Neural Network —— FNN
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, active_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.active = active_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.active(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# LeFF: Local-enhanced Feed-Forward
# 该层促进空间维度上相邻标记之间的相关性, 利用 Depth wise 增强空间维度之间的关联性
class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, active_layer=nn.GELU):
        super(LeFF, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            active_layer()
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            active_layer()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = self.linear1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.depth_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear2(x)
        return x




# 正文中将上下采样转换成双线性插值加卷积（去除转置卷积的块状效应，破坏结构）
class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = x.permute(0, 3, 1, 2).contiguous() # B C H W
        out = self.conv(x).permute(0, 2, 3, 1).contiguous()  # B H W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4
        print("DownSample:{%.2f}" % (flops / 1e9))
        return flops


# UpSample Block
class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous() # B C H W
        out = self.de_conv(x).permute(0, 2, 3, 1).contiguous()  # B H W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("UpSample:{%.2f}" % (flops / 1e9))
        return flops


# final UpSample Block
class FinalUpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FinalUpSample, self).__init__()
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous() # B C H W
        out = self.de_conv(x).permute(0, 2, 3, 1).contiguous()  # B H W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("UpSample:{%.2f}" % (flops / 1e9))
        return flops


# 有监督的注意力模块：Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, in_channels, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(in_channels, in_channels, kernel_size, bias)
        self.conv2 = conv(in_channels, 3, kernel_size, bias)
        self.conv3 = conv(3, in_channels, kernel_size, bias)

    def forward(self, x, origin_x):
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv1(x)
        img = self.conv2(x) + origin_x
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img



## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, norm):
        super(CAB, self).__init__()
        modules_body = [
            conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias=bias)
        ]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.norm = norm(n_feat)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.norm(res)
        res += x
        return res


class Signal_MTR_Model(nn.Module):
    def __init__(self, image_size=128, in_channels=3, embed_dim=32, win_size=8, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attention_drop=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 depths=(2, 2, 2, 2, 2, 2, 2), num_heads=(1, 2, 4, 8, 8, 4, 2), token_projection='linear',
                 token_mlp='ffn', se_layer=False, drop_path_rate=0.1, dowsample=DownSample, upsample=UpSample, csff=False):
        super(Signal_MTR_Model, self).__init__()

        self.u_former = U_former(image_size, in_channels, embed_dim, win_size, mlp_ratio, qkv_bias, qk_scale,
                                  drop_rate, attention_drop, norm_layer, use_checkpoint, depths, num_heads,
                                  token_projection,
                                  token_mlp, se_layer, drop_path_rate, dowsample, upsample, csff=csff)


        self.output_projection = OutputProjection(in_channel= 2*embed_dim, out_channel=in_channels, kernel_size=3, stride=1)

    def forward(self, x):
        encoder_outs, decoder_outs = self.u_former(x)
        res_img = self.output_projection(decoder_outs[2])
        return res_img






# high net architecture
class High_MTR_Model(nn.Module):
    def __init__(self, image_size=128, in_channels=3, embed_dim=32, win_size=8, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attention_drop=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 depths=(2, 2, 2, 2, 2, 2, 2), num_heads=(1, 2, 4, 8, 8, 4, 2), token_projection='linear',
                 token_mlp='ffn', se_layer=False, drop_path_rate=0.1, dowsample=DownSample, upsample=UpSample, csff=True):
        super(High_MTR_Model, self).__init__()

        self.u_former2 = U_former(image_size, in_channels, embed_dim, win_size, mlp_ratio, qkv_bias, qk_scale,
                                  drop_rate, attention_drop, norm_layer, use_checkpoint, depths, num_heads,
                                  token_projection,
                                  token_mlp, se_layer, drop_path_rate, dowsample, upsample, csff=csff)

        self.concat = conv(embed_dim * 2, embed_dim, kernel_size=3, bias=qkv_bias)
        self.input_projection = InputProjection(in_channel=in_channels, out_channel=embed_dim, kernel_size=3, stride=1, active_layer=nn.LeakyReLU)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.output_projection = OutputProjection(in_channel= 2*embed_dim, out_channel=in_channels, kernel_size=3, stride=1)

    def forward(self, x, sam_feature=None, encoder_outs=None, decoder_outs=None):
        y = self.input_projection(x)
        y = self.pos_drop(y)
        if sam_feature is not None:
            y = self.concat(torch.cat([sam_feature, y.permute(0, 3, 1, 2)], 1)).permute(0, 2, 3, 1)
        feat_encoders, res_decoders  = self.u_former2(y, encoder_outs=encoder_outs, decoder_outs=decoder_outs)
        res_img = res_decoders[-1]
        return res_img






# low net architecture
class Low_MTR_Model(nn.Module):
    def __init__(self, image_size=128, in_channels=3, embed_dim=32, win_size=8, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attention_drop=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 depths=(2, 2, 2, 2, 2, 2, 2), num_heads=(1, 2, 4, 8, 8, 4, 2), token_projection='linear',
                 token_mlp='ffn', se_layer=False,
                 drop_path_rate=0.1, dowsample=DownSample, upsample=UpSample):
        super(Low_MTR_Model, self).__init__()
        self.u_former1 = U_former(image_size, in_channels, embed_dim, win_size, mlp_ratio, qkv_bias, qk_scale,
                                  drop_rate, attention_drop, norm_layer, use_checkpoint, depths, num_heads,
                                  token_projection,
                                  token_mlp, se_layer, drop_path_rate, dowsample, upsample, csff=False)
        self.sam = SAM(embed_dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # x.shape - B C H W
        B, C, H, W = x.shape

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches
        x2_top_img = x[:, :, 0:int(H/2), :]
        x2_bot_img = x[:, :, int(H/2):H, :]

        # Four Patches for Stage 4
        x1_ltop_img = x2_top_img[:, :, :, 0:int(W / 2)]
        x1_rtop_img = x2_top_img[:, :, :, int(W / 2):W]
        x1_lbot_img = x2_bot_img[:, :, :, 0:int(W / 2)]
        x1_rbot_img = x2_bot_img[:, :, :, int(W / 2):W]

        feat1_ltop_encoders, res1_ltop_decoders = self.u_former1(x1_ltop_img)
        feat1_rtop_encoders, res1_rtop_decoders = self.u_former1(x1_rtop_img)
        feat1_lbot_encoders, res1_lbot_decoders = self.u_former1(x1_lbot_img)
        feat1_rbot_encoders, res1_rbot_decoders = self.u_former1(x1_rbot_img)

        # Concat deep feature
        feat1_top_encoders = [torch.cat((k,v), 2) for k,v in zip(feat1_ltop_encoders, feat1_rtop_encoders)]
        feat1_bot_encoders = [torch.cat((k, v), 2) for k, v in zip(feat1_lbot_encoders, feat1_rbot_encoders)]
        res1_top_decoders = [torch.cat((k, v), 2) for k, v in zip(res1_ltop_decoders, res1_rtop_decoders)]
        res1_bot_decoders = [torch.cat((k, v), 2) for k, v in zip(res1_lbot_decoders, res1_rbot_decoders)]

        feat1_encoders = [torch.cat((k,v), 1) for k,v in zip(feat1_top_encoders, feat1_bot_encoders)]
        res1_decoders = [torch.cat((k,v), 1) for k,v in zip(res1_top_decoders, res1_bot_decoders)]


        x1_sam_feature, x1_img = self.sam(res1_decoders[-1], x)

        return feat1_encoders, res1_decoders, x1_sam_feature, x1_img






# Overall net architecture
class MTR_Model(nn.Module):
    def __init__(self, image_size=128, in_channels=3, embed_dim=32, win_size=8, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attention_drop=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 depths=(2, 2, 2, 2, 2, 2, 2),num_heads=(1, 2, 4, 8, 8, 4, 2), token_projection='linear', token_mlp='ffn', se_layer=False,
                 drop_path_rate=0.1, dowsample=DownSample, upsample=UpSample):
        super(MTR_Model, self).__init__()

        self.u_former1 = U_former(image_size, in_channels, embed_dim, win_size, mlp_ratio, qkv_bias, qk_scale,
                                  drop_rate, attention_drop, norm_layer, use_checkpoint, depths, num_heads, token_projection,
                                  token_mlp, se_layer, drop_path_rate, dowsample, upsample, csff=False)

        self.u_former2 = U_former(image_size, in_channels, embed_dim, win_size, mlp_ratio, qkv_bias, qk_scale,
                                  drop_rate, attention_drop, norm_layer, use_checkpoint, depths, num_heads, token_projection,
                                  token_mlp, se_layer, drop_path_rate, dowsample, upsample, csff=True)


        self.concat = conv(embed_dim*2, embed_dim, kernel_size=3, bias=qkv_bias)

        self.sam = SAM(embed_dim*2, kernel_size=1, bias=qkv_bias)
        self.output = OutputProjection(in_channel= 2*embed_dim, out_channel=in_channels, kernel_size=3, stride=1)


    def forward(self, x):
        # x.shape - B C H W
        B, C, H, W = x.shape

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches
        x2_top_img = x[:, :, 0:int(H/2), :]
        x2_bot_img = x[:, :, int(H/2):H, :]

        # Four Patches for Stage 4
        x1_ltop_img = x2_top_img[:, :, :, 0:int(W / 2)]
        x1_rtop_img = x2_top_img[:, :, :, int(W / 2):W]
        x1_lbot_img = x2_bot_img[:, :, :, 0:int(W / 2)]
        x1_rbot_img = x2_bot_img[:, :, :, int(W / 2):W]

        feat1_ltop_encoders, res1_ltop_decoders = self.u_former1(x1_ltop_img)
        feat1_rtop_encoders, res1_rtop_decoders = self.u_former1(x1_rtop_img)
        feat1_lbot_encoders, res1_lbot_decoders = self.u_former1(x1_lbot_img)
        feat1_rbot_encoders, res1_rbot_decoders = self.u_former1(x1_rbot_img)

        # Concat deep feature
        feat1_top_encoders = [torch.cat((k,v), 2) for k,v in zip(feat1_ltop_encoders, feat1_rtop_encoders)]
        feat1_bot_encoders = [torch.cat((k, v), 2) for k, v in zip(feat1_lbot_encoders, feat1_rbot_encoders)]
        res1_top_decoders = [torch.cat((k, v), 2) for k, v in zip(res1_ltop_decoders, res1_rtop_decoders)]
        res1_bot_decoders = [torch.cat((k, v), 2) for k, v in zip(res1_lbot_decoders, res1_rbot_decoders)]

        feat1_encoders = [torch.cat((k,v), 3) for k,v in zip(feat1_top_encoders, feat1_bot_encoders)]
        res1_decoders = [torch.cat((k,v), 3) for k,v in zip(res1_top_decoders, res1_bot_decoders)]


        x1_sam_feature, x1_img = self.sam(res1_decoders[-1], x)
        # 原始图像恢复
        feat_encoders, res_decoders= self.u_former2(x, encoder_outs=feat1_encoders, decoder_outs=res1_decoders)
        x_img = self.output(res_decoders[-1]) + x
        return [x1_img, x_img]







class U_former(nn.Module):
    def __init__(self, image_size=128, in_channels=3, embed_dim=32, win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attention_drop=0., norm_layer=nn.LayerNorm, use_checkpoint=False, depths=(2, 2, 2, 2, 2, 2, 2),
                 num_heads=(1, 2, 4, 8, 8, 4, 2), token_projection='linear', token_mlp='ffn', se_layer=False, drop_path_rate=0.1,
                 dowsample=DownSample, upsample=UpSample, csff=False):
        super(U_former, self).__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.resolution = image_size
        self.mlp_ratio = mlp_ratio
        self.mlp = token_mlp
        self.win_size =win_size
        self.csff = csff

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[3]
        dec_dpr = enc_dpr[::-1]


        # Input/Output
        self.input_projection = InputProjection(in_channel=in_channels, out_channel=embed_dim, kernel_size=3, stride=1, active_layer=nn.LeakyReLU)
        # output 的输入channels根据模型架构来调整
        self.output_projection = OutputProjection(in_channel= 2*embed_dim, out_channel=in_channels, kernel_size=3, stride=1)

        # Encoder
        self.encoder_layer0 = TransformerBlocks(dim=embed_dim, input_resolution=(image_size, image_size), depth=depths[0], num_heads=num_heads[0],
                                                win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attention_drop=attention_drop,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])], norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)

        self.encoder_layer1 = TransformerBlocks(dim=embed_dim*2, input_resolution=(image_size // 2, image_size // 2),
                                                depth=depths[1], num_heads=num_heads[1],
                                                win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                qk_scale=qk_scale, drop=drop_rate, attention_drop=attention_drop,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)

        self.encoder_layer2 = TransformerBlocks(dim=embed_dim*4, input_resolution=(image_size // (2**2), image_size // (2**2)),
                                                depth=depths[2], num_heads=num_heads[2],
                                                win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                qk_scale=qk_scale, drop=drop_rate, attention_drop=attention_drop,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)

        # Bottleneck
        self.bottleneck = TransformerBlocks(dim=embed_dim*8, input_resolution=(image_size // (2 ** 3), image_size // (2 ** 3)),
                                      depth=depths[3], num_heads=num_heads[3], win_size=win_size, mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attention_drop=attention_drop,
                                      drop_path=conv_dpr, norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                      token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)


        # Decoder
        self.upsample_0 = upsample(embed_dim*8, embed_dim*4)
        self.decoder_layer0 = TransformerBlocks(dim=embed_dim*8, input_resolution=(image_size // (2**2), image_size // (2**2)),
                                                depth=depths[4], num_heads=num_heads[4],
                                                win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                qk_scale=qk_scale, drop=drop_rate, attention_drop=attention_drop,
                                                drop_path=dec_dpr[:depths[4]], norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint, token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)

        self.upsample_1 = upsample(embed_dim*8, embed_dim*2)
        self.decoder_layer1 = TransformerBlocks(dim=embed_dim*4,
                                                input_resolution=(image_size // 2, image_size // 2),
                                                depth=depths[5], num_heads=num_heads[5],
                                                win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                qk_scale=qk_scale, drop=drop_rate, attention_drop=attention_drop,
                                                drop_path=dec_dpr[sum(depths[4:5]):sum(depths[4:6])],
                                                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)

        self.upsample_2 = upsample(embed_dim*4, embed_dim)
        self.decoder_layer2 = TransformerBlocks(dim=embed_dim*2,
                                                input_resolution=(image_size, image_size),
                                                depth=depths[6], num_heads=num_heads[6],
                                                win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                qk_scale=qk_scale, drop=drop_rate, attention_drop=attention_drop,
                                                drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],
                                                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.final_decoder = OutputProjection(in_channel= 2*embed_dim, out_channel=embed_dim, kernel_size=3, stride=1)
        # CSFF
        if csff:
            self.csff_encoder0 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=qkv_bias)
            self.csff_encoder1 = nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=1, bias=qkv_bias)
            self.csff_encoder2 = nn.Conv2d(embed_dim*4, embed_dim*4, kernel_size=1, bias=qkv_bias)
            self.csff_decoder2 = nn.Conv2d(embed_dim*4, embed_dim*4, kernel_size=1, bias=qkv_bias)
            self.csff_decoder1 = nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=1, bias=qkv_bias)
            self.csff_decoder0 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=qkv_bias)



        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x,  mask=None, encoder_outs=None, decoder_outs=None):
        if not self.csff:
            y = self.input_projection(x)
            y = self.pos_drop(y)
        else:
            y = x
        # Encoder
        conv0 = self.encoder_layer0(y, mask=mask)
        if self.csff and encoder_outs is not None and decoder_outs is not None:
            conv0 = conv0.permute(0,3,1,2) + self.csff_encoder0(encoder_outs[0]) + \
                    self.csff_decoder0(decoder_outs[0])
            conv0 = conv0.permute(0, 2, 3, 1)
        pool0 = self.dowsample_0(conv0)

        conv1 = self.encoder_layer1(pool0, mask=mask)
        if self.csff and encoder_outs is not None and decoder_outs is not None:
            conv1 = conv1.permute(0,3,1,2) + self.csff_encoder1(encoder_outs[1]) + \
                    self.csff_decoder1(decoder_outs[1])
            conv1 = conv1.permute(0, 2, 3, 1)
        pool1 = self.dowsample_1(conv1)

        conv2 = self.encoder_layer2(pool1, mask=mask)
        if self.csff and encoder_outs is not None and decoder_outs is not None:
            conv2 = conv2.permute(0,3,1,2) + self.csff_encoder2(encoder_outs[2]) + \
                    self.csff_decoder2(decoder_outs[2])
            conv2 = conv2.permute(0, 2, 3, 1)
        pool2 = self.dowsample_2(conv2)

        # Bottleneck
        bottleneck = self.bottleneck(pool2, mask)


        # Decoder
        # 这里的连接采用拼接的形式
        up0 = self.upsample_0(bottleneck)
        deconv0 = torch.cat([up0, conv2], -1)
        deconv0 = self.decoder_layer0(deconv0, mask=mask)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv1], -1)
        deconv1 = self.decoder_layer1(deconv1, mask=mask)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv0], -1)
        deconv2 = self.decoder_layer2(deconv2, mask=mask)


        # Output Projection
        if self.csff:
            y = self.output_projection(deconv2)
        else:
            y = self.final_decoder(deconv2).permute(0, 2, 3, 1)

        # return x+y
        return [conv0, conv1, conv2], [deconv0, deconv1, deconv2, y]



class TransformerBlocks(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, win_size, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attention_drop=0., drop_path=None, drop_path_default=0., norm_layer=nn.LayerNorm,
                 use_checkpoint=False, token_projection='linear',token_mlp='ffn',se_layer=False):
        super(TransformerBlocks, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            # 当shift_size为0时是W-MSA，而非0时是SW-MSA
            TransformerVision(dim=dim, input_resolution=input_resolution,
                                  num_heads=num_heads, win_size=win_size,
                                  shift_size=0 if (i % 2 == 0) else win_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop, attention_drop=attention_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path_default,
                                  norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                                  use_se_layer=se_layer)
            for i in range(depth)])


    def forward(self, x, mask=None):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x, mask)
        return x


## Transformer模型方法。
## shift_size：张量元素移位的位数。通常为 window_size // 2
class TransformerVision(nn.Module):
    def __init__(self,dim, input_resolution, num_heads, win_size=8, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attention_drop=0., drop_path=0., active_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, token_projection='linear',token_mlp='leff',use_se_layer=False):
        super(TransformerVision, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        if min(input_resolution) <= win_size:
            self.shift_size = 0
            self.win_size = min(input_resolution)
        assert 0 <= shift_size < win_size, " shift size must less than window size"
        self.norm1 = norm_layer(dim)
        self.attention = WindowAttention(
            dim, win_size=to_2tuple(win_size), num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attention_drop=attention_drop, projection_drop=drop,
            token_projection=token_projection, se_layer=use_se_layer
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, active_layer=active_layer, drop=drop) if token_mlp=='ffn' else LeFF(dim,mlp_hidden_dim,active_layer=active_layer)


    def forward(self, x, mask=None):
        B, H, W, C = x.shape

        ## input mask
        if mask is not None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, window_size, window_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, window_size*window_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(
                1)  # nW, window_size*window_size, window_size*window_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (
                slice(0, -self.win_size),
                slice(-self.win_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            w_slices = (
                slice(0, -self.win_size),
                slice(-self.win_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            # nW, window_size, window_size, 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)
            # nW, window_size*window_size
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)
            # nW, window_size*window_size, window_size*window_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)
            attn_mask = attn_mask or shift_attn_mask
            attn_mask = attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0))

        shortcut = x

        # Layer Norm
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attention(x_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x



# window-based self-attention
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear',qkv_bias=True, qk_scale=None, attention_drop=0., projection_drop=0., se_layer=False):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads)
        )

        # 用于位置坐标索引定位,位置编码（主要是偏移量的计算）
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim//num_heads, bias=qkv_bias)
        elif token_projection == 'linear_concat':
            self.qkv = LinearProjection_Concat_kv(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.attention_drop = nn.Dropout(attention_drop)
        self.projection = nn.Linear(dim, dim)


        self.se_layer = SELayer(dim) if se_layer else nn.Identity()
        self.projection_drop = nn.Dropout(projection_drop)

        # 函数trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.)的目的是用截断的正态分布绘制的值填充输入张量，我们只需要输入均值mean，标准差std，下界a，上界b
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attention_k=None, mask=None):
        B, H, W, C = x.shape
        q, k, v = self.qkv(x, attention_k)
        q = q * self.scale
        # 矩阵叉乘

        attention = image_matrix_mul(q, k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        ratio = attention.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = einops.repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = einops.repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attention = attention.view(B // nW, nW, self.num_heads, H * W, H * W * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, H*W, H * W * ratio)
            attention = self.softmax(attention)
        else:
            attention = self.softmax(attention)

        attention = self.attention_drop(attention)

        x = (attention @ v.view(v.size(0), v.size(1), -1, v.size(-1))).transpose(1, 2).reshape(B, H, W, C)
        x = self.projection(x)
        x = self.se_layer(x)
        x = self.projection_drop(x)
        return x


# (batch_size, head_num, H, W, dim_head)相乘
# 先将(batch_size, head_num, H, W, dim_head)->(batch_size, head_num, L, dim_head)再执行相乘操作
def image_matrix_mul(x, y):
    result = x.view(x.size(0),x.size(1), -1, x.size(-1)) @ \
             y.view(y.size(0), y.size(1), -1, y.size(-1)).transpose(-2, -1)
    return result



class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, bias=True):
        super(ConvProjection, self).__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attention_kv=None):
        attention_kv = x if attention_kv is None else attention_kv
        x = einops.rearrange(x, 'b h w c -> b c h w')
        attention_kv = einops.rearrange(attention_kv, 'b h w c -> b c h w')

        q = self.to_q(x)
        k = self.to_k(attention_kv)
        v = self.to_v(attention_kv)

        q = einops.rearrange(q, 'b (l d) h w -> b l h w d', l=self.heads)
        k = einops.rearrange(k, 'b (l d) h w -> b l h w d', l=self.heads)
        v = einops.rearrange(v, 'b (l d) h w -> b l h w d', l=self.heads)

        return q, k, v


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_heads=64, bias=True):
        super(LinearProjection, self).__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x, attention_kv=None):
        B, H, W, C = x.shape
        attention_kv = x if attention_kv is None else attention_kv
        q = self.to_q(x).reshape(B, H, W, 1, self.heads, C // self.heads).permute(3, 0, 4, 1, 2, 5)
        kv = self.to_kv(attention_kv).reshape(B, H, W, 2, self.heads, C // self.heads).permute(3, 0, 4, 1, 2, 5)

        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class LinearProjection_Concat_kv(nn.Module):
    def __init__(self, dim, heads=8, dim_heads=64, bias=True):
        super(LinearProjection_Concat_kv, self).__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x, attention_kv=None):
        B, H, W, C = x.shape
        attention_kv = x if attention_kv is None else attention_kv
        qkv_dec = self.to_qkv(x).reshape(B, H, W, 3, self.heads, C//self.heads).permute(3, 0, 4, 1, 2, 5)
        kv_enc = self.to_kv(attention_kv).reshape(B, H, W, 2, self.heads, C//self.heads).permute(3, 0, 4, 1, 2, 5)
        q, k_d, v_d = qkv_dec[0], qkv_dec[1], qkv_dec[2]
        k_e, v_e = kv_enc[0], kv_enc[1]
        k = torch.cat((k_d, k_e), dim=2)
        v = torch.cat((v_d, v_e), dim=2)
        return q, k, v



class SepConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, active_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        # groups=in_channels 每一个输入通道都有自己的过滤器（符合 depth_wise的定义）
        self.depth_wise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels)
        self.point_wise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.active_layer = active_layer() if active_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.active_layer(x)
        x = self.point_wise(x)
        return x


# SE 模块主要为了提升模型对 channel 特征的敏感性
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        x = x.permute(0, 2, 3, 1)
        return x


########### window operation#############
# 函数是用于对张量划分窗口，指定窗口大小。
def window_partition(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


## U-Net
class Encoder(nn.Module):
    def __init__(self, embed_dim, kernel_size, reduction, act, norm, bias):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(embed_dim, kernel_size, reduction, bias=bias, act=act, norm=norm) for _ in range(2)]
        self.encoder_level2 = [CAB(embed_dim * 2, kernel_size, reduction, bias=bias, act=act, norm=norm) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(embed_dim * 4, kernel_size, reduction, bias=bias, act=act, norm=norm) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample_S(embed_dim, embed_dim * 2)
        self.down23 = DownSample_S(embed_dim * 2, embed_dim * 4)


    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, embed_dim, kernel_size, reduction, act, norm, bias):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(embed_dim, kernel_size, reduction, bias=bias, act=act, norm=norm) for _ in range(2)]
        self.decoder_level2 = [CAB(embed_dim * 2, kernel_size, reduction, bias=bias, act=act, norm=norm) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(embed_dim * 4, kernel_size, reduction, bias=bias, act=act, norm=norm) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(embed_dim, kernel_size, reduction, bias=bias, act=act, norm=norm)
        self.skip_attn2 = CAB(embed_dim*2, kernel_size, reduction, bias=bias, act=act, norm=norm)

        self.up21 = SkipUpSample(embed_dim, embed_dim*2)
        self.up32 = SkipUpSample(embed_dim*2, embed_dim*4)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]



class SkipUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(out_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x




# 需要在 MPTR 中增加 LayerNorm
class MPTR_SuperviseNet(nn.Module):
    def __init__(self, in_channel=3, embed_dim=32, kernel_size=3, reduction=4, bias=False, image_size=128, in_channels=3, win_size=8, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attention_drop=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 depths=(2, 2, 2, 2, 2, 2, 2), num_heads=(1, 2, 4, 8, 8, 4, 2), token_projection='linear',
                 token_mlp='ffn', se_layer=False, drop_path_rate=0.1, dowsample=DownSample, upsample=UpSample, csff=True):
        super(MPTR_SuperviseNet, self).__init__()
        act = nn.PReLU()
        self.shallow_feat = nn.Sequential(conv(in_channel, embed_dim, kernel_size, bias=bias),
                                           CAB(embed_dim, kernel_size, reduction, bias=bias, act=act, norm=Layer_Norm))
        self.u_former = U_former(image_size, in_channels, embed_dim, win_size, mlp_ratio, qkv_bias, qk_scale,
                                  drop_rate, attention_drop, norm_layer, use_checkpoint, depths, num_heads,
                                  token_projection, token_mlp, se_layer, drop_path_rate, dowsample, upsample, csff=csff)
        self.concat = conv(embed_dim * 2, embed_dim, kernel_size=3, bias=qkv_bias)
        self.input_projection = InputProjection(in_channel=in_channels, out_channel=embed_dim, kernel_size=3, stride=1,
                                                active_layer=nn.LeakyReLU)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.stage_encoder = Encoder(embed_dim, kernel_size, reduction, act, Layer_Norm, bias)
        self.stage_decoder = Decoder(embed_dim, kernel_size, reduction, act, Layer_Norm, bias)
        self.sam = SAM_S(embed_dim,  norm=Layer_Norm,kernel_size=1, bias=bias)
        
    def forward(self, x):
        B, C, H, W = x.shape

        x2top_img = x[:, :, 0:int(H / 2), :]
        x2bot_img = x[:, :, int(H / 2):H, :]
        x1ltop_img = x2top_img[:, :, :, 0:int(W / 2)]
        x1rtop_img = x2top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x2bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x2bot_img[:, :, :, int(W / 2):W]

        x1ltop = self.shallow_feat(x1ltop_img)
        x1rtop = self.shallow_feat(x1rtop_img)
        x1lbot = self.shallow_feat(x1lbot_img)
        x1rbot = self.shallow_feat(x1rbot_img)

        feat1_ltop = self.stage_encoder(x1ltop)
        feat1_rtop = self.stage_encoder(x1rtop)
        feat1_lbot = self.stage_encoder(x1lbot)
        feat1_rbot = self.stage_encoder(x1rbot)


        ## Pass features through Decoder of Stage 1
        res1_ltop_decoders = self.stage_decoder(feat1_ltop)
        res1_rtop_decoders = self.stage_decoder(feat1_rtop)
        res1_lbot_decoders = self.stage_decoder(feat1_lbot)
        res1_rbot_decoders = self.stage_decoder(feat1_rbot)



        ## Concat deep features
        feat_top_encoders = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat_bot_encoders = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]

        res_top_decoders = [torch.cat((k, v), 3) for k, v in zip(res1_ltop_decoders, res1_rtop_decoders)]
        res_bot_decoders = [torch.cat((k, v), 3) for k, v in zip(res1_lbot_decoders, res1_rbot_decoders)]


        feat_encoders = [torch.cat((k, v), 2) for k, v in zip(feat_top_encoders, feat_bot_encoders)]
        res_decoders = [torch.cat((k, v), 2) for k, v in zip(res_top_decoders, res_bot_decoders)]



        sam_feats, stage_img = self.sam(res_decoders[0], x)


        y = self.input_projection(x)
        y = self.pos_drop(y)
        if sam_feats is not None:
            y = self.concat(torch.cat([sam_feats, y.permute(0, 3, 1, 2)], 1)).permute(0, 2, 3, 1)


        feat_encoders, res_decoders = self.u_former(y, encoder_outs=feat_encoders, decoder_outs=res_decoders)

        # uf_decoder = res_decoders[2].permute(0, 3, 1, 2)
        res_img = res_decoders[-1]


        return  stage_img, res_img


class DownSample_S(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample_S, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

## Supervised Attention Module
class SAM_S(nn.Module):
    def __init__(self, n_feat, norm, kernel_size, bias):
        super(SAM_S, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
        self.norm1 = norm(n_feat)
        self.norm2 = norm(3)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        self.norm1(x1)
        self.norm2(img)
        return x1, img


class Layer_Norm(nn.Module):
    def __init__(self, in_channels):
        super(Layer_Norm, self).__init__()
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        y = self.norm(x.permute(0, 2, 3, 1))
        y = y.permute(0, 3, 1, 2)
        return y









