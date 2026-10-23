import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, in_channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)
    def forward(self, x):
        n, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        return x * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w.permute(0, 1, 3, 2)))

class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, rates=(6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        ])
        for r in rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(True)
            ))
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(rates) + 2) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True), nn.Dropout(0.3)
        )
    def forward(self, x):
        size = (int(x.shape[-2]), int(x.shape[-1]))
        res = [conv(x) for conv in self.convs]
        res.append(F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False))
        return self.project(torch.cat(res, dim=1))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_ca=True):
        super().__init__()
        self.ca = CoordinateAttention(skip_ch) if (skip_ch > 0 and use_ca) else None
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        if skip is not None:
            if self.ca is not None:
                skip = self.ca(skip)
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=(int(x.shape[-2]), int(x.shape[-1])), mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ProposedHybridASPPCA(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model('resnet50', pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        chs = self.encoder.feature_info.channels()
        self.aspp = ASPP(chs[4], out_channels=256)
        self.b0 = DecoderBlock(256, chs[3], 256, use_ca=True)
        self.b1 = DecoderBlock(256, chs[2], 128, use_ca=True)
        self.b2 = DecoderBlock(128, chs[1], 64, use_ca=True)
        self.b3 = DecoderBlock(64, chs[0], 32, use_ca=True)
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    def forward(self, x):
        inp_size = (int(x.shape[-2]), int(x.shape[-1]))
        f = self.encoder(x)
        d = self.b0(self.aspp(f[4]), f[3])
        d = self.b1(d, f[2])
        d = self.b2(d, f[1])
        d = self.b3(d, f[0])
        d = F.interpolate(d, size=inp_size, mode='bilinear', align_corners=False)
        return self.final(d)

model = ProposedHybridASPPCA(8, pretrained=False)
model.load_state_dict(torch.load("/home/turjo/Documents/Dataset_Final_Defence/model_showcase_website/models/checkpoints/best_hybrid.pt", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 160, 160)
out_path = "/home/turjo/Documents/Dataset_Final_Defence/model_showcase_website/models/checkpoints/best_hybrid_160.onnx"
torch.onnx.export(model, dummy_input, out_path,
                  export_params=True, opset_version=18, do_constant_folding=True,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}})

from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(out_path, "/home/turjo/Documents/Dataset_Final_Defence/model_showcase_website/models/checkpoints/best_hybrid_160_int8.onnx", weight_type=QuantType.QUInt8)
