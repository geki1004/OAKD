from torchvision import models
import torch.ao.quantization as tq
import torch
from torch import nn
import torch.nn.functional as F

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(GatedConv2d, self).__init__()
        self.conv_feat = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv_feat(x)
        gate = self.sigmoid(self.conv_gate(x))
        out = feat * gate
        out = self.bn(out)
        out = self.relu(out)
        return out

class OGA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(OGA, self).__init__()
        self.conv_feat = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, y):
        feat = self.conv_feat(x)
        gate = self.sigmoid(self.gap(self.conv_gate(y)))
        out = feat * gate
        out = self.bn(out)
        out = self.relu(out)
        return out

class DBGatedConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        conv_layer = [
            GatedConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            GatedConv2d(out_channels, out_channels, kernel_size=3, padding=1),
        ]
        super(DBGatedConv, self).__init__(*conv_layer)

class DBConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        conv_layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(DBConv, self).__init__(*conv_layers)

class ConvBR(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        conv_layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ConvBR, self).__init__(*conv_layers)

class SobelEdgeAttention(nn.Module):
    def __init__(self, in_channels, ksize=3, use_residual=True):
        super().__init__()
        assert ksize == 3, "현재 구현은 3x3 Sobel에 맞춰져 있음"
        self.in_channels = in_channels
        self.use_residual = use_residual

        # Sobel 커널
        gx = torch.tensor([[[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]], dtype=torch.float32)
        gy = torch.tensor([[[-1,-2,-1],
                            [ 0, 0, 0],
                            [ 1, 2, 1]]], dtype=torch.float32)

        self.register_buffer("weight_x", gx.unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.register_buffer("weight_y", gy.unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.refine = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.eps = 1e-6

    @torch.no_grad()
    def _sobel(self, x):
        grad_x = F.conv2d(x, self.weight_x, bias=None, stride=1, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(x, self.weight_y, bias=None, stride=1, padding=1, groups=self.in_channels)
        return grad_x, grad_y

    def forward(self, skip):
        B, C, H, W = skip.shape

        gx, gy = self._sobel(skip)
        gm = torch.sqrt(gx * gx + gy * gy + self.eps)

        edge = gm.mean(dim=1, keepdim=True)

        e_mu = edge.mean(dim=(2,3), keepdim=True)
        e_sd = edge.std(dim=(2,3), keepdim=True) + self.eps
        edge_norm = (edge - e_mu) / e_sd

        att = torch.sigmoid(self.refine(edge_norm))

        if self.use_residual:
            out = skip * (1.0 + self.alpha * att)       # residual gating
        else:
            out = skip * att

        return out

class DDOS_Net(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, pretrained=False):
        super(DDOS_Net, self).__init__()

        vgg16 = models.vgg16(pretrained=pretrained)

        old_conv = vgg16.features[0]

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias

        vgg16.features[0] = new_conv

        self.encoder1 = vgg16.features[:4] # 64, H/2, W/2
        self.encoder2 = vgg16.features[4:9] # 128, H/4, W/4
        self.encoder3 = vgg16.features[9:16] # 256, H/8, W/8
        self.encoder4 = vgg16.features[16:23] # 512, H/16, W/16
        self.encoder5 = vgg16.features[23:30] # 512, H/32, W/32

        self.bottleneck = DBConv(512, 512)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decoder4 = DBConv(512 + 512, 512)
        self.decoder3 = DBConv(512 + 512, 256)
        self.decoder2 = DBConv(256 + 256, 128)
        self.decoder1 = DBConv(128 + 128, 64)
        self.decoder0 = DBConv(64 + 64 , 32)

        self.out_decoder4 = OGA(512 + 512, 512)#ConvBR(512+512, 512)#
        self.out_decoder4_1 = ConvBR(512,512)
        self.out_decoder3 = OGA(512 + 512, 256)#ConvBR(512+512, 256)#
        self.out_decoder3_1 = ConvBR(256,256)
        self.out_decoder2 = OGA(256 + 256, 128)#ConvBR(256+256, 128)#
        self.out_decoder2_1 = ConvBR(128,128)
        self.out_decoder1 = OGA(128 + 128, 64) #ConvBR(128+128, 64)#
        self.out_decoder1_1 = ConvBR(64,64)
        self.out_decoder0 = OGA(64 + 64, 32)
        self.out_decoder0_1 = ConvBR(32,32)

        self.edge_att4 = SobelEdgeAttention(in_channels=512, ksize=3, use_residual=True)
        self.edge_att3 = SobelEdgeAttention(in_channels=256, ksize=3, use_residual=True)
        self.edge_att2 = SobelEdgeAttention(in_channels=128, ksize=3, use_residual=True)
        self.edge_att1 = SobelEdgeAttention(in_channels=64, ksize=3, use_residual=True)


        self.seg_head = nn.Conv2d(32 + 32, num_classes, 1)
        self.ob_head = nn.Conv2d(32, 2, 1)
        self.out_head = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        pass1 = self.encoder1(x)
        pass2 = self.encoder2(pass1)
        pass3 = self.encoder3(pass2)
        pass4 = self.encoder4(pass3)
        pass5 = self.encoder5(pass4)
        neck = self.bottleneck(pass5)

        output1 = torch.cat((neck, pass5), 1)
        output = self.decoder4(output1)
        
        output = self.up(output)
        output2 = torch.cat((output, self.edge_att4(pass4)), 1)
        output = self.decoder3(output2)
        
        output = self.up(output)
        output3 = torch.cat((output, self.edge_att3(pass3)), 1)
        output = self.decoder2(output3)
        
        output = self.up(output)
        output4 = torch.cat((output, self.edge_att2(pass2)), 1)
        output = self.decoder1(output4)
        
        output = self.up(output)
        output5 = torch.cat((output, self.edge_att1(pass1)), 1)
        output = self.decoder0(output5)

        out_output = torch.cat((neck, pass5), 1)
        out_output = self.out_decoder4_1(self.out_decoder4(out_output, output1)) #
        
        out_output = self.up(out_output)
        out_output = torch.cat((out_output, pass4), 1)
        out_output = self.out_decoder3_1(self.out_decoder3(out_output, output2)) #
        
        out_output = self.up(out_output)
        out_output = torch.cat((out_output, pass3), 1)
        out_output = self.out_decoder2_1(self.out_decoder2(out_output, output3)) #
        
        out_output = self.up(out_output)
        out_output = torch.cat((out_output, pass2), 1)
        out_output = self.out_decoder1_1(self.out_decoder1(out_output, output4)) #
        
        out_output = self.up(out_output)
        out_output = torch.cat((out_output, pass1), 1)
        out_output = self.out_decoder0_1(self.out_decoder0(out_output, output5))

        ob = self.ob_head(output)
        out = self.sigmoid(self.out_head(out_output))
        seg = self.seg_head(torch.cat((output, out_output), 1))


        return seg, ob, out, neck

class ContractingPath(nn.Module):
    def __init__(self, in_channels, first_outchannels):
        super(ContractingPath, self).__init__()
        self.conv1 = DBConv(in_channels, first_outchannels)
        self.conv2 = DBConv(first_outchannels, first_outchannels * 2)
        self.conv3 = DBConv(first_outchannels * 2, first_outchannels * 4)
        self.conv4 = DBConv(first_outchannels * 4, first_outchannels * 8)

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        output1 = self.conv1(x)
        output = self.maxpool(output1)
        output2 = self.conv2(output)
        output = self.maxpool(output2)
        output3 = self.conv3(output)
        output = self.maxpool(output3)
        output4 = self.conv4(output)
        output = self.maxpool(output4)
        return output1, output2, output3, output4, output

class SDS_Net(nn.Module):
    def __init__(self, in_channels=4, first_outchannels=32, num_classes=3):
        super(SDS_Net, self).__init__()
        self.contracting_path = ContractingPath(in_channels=in_channels, first_outchannels=first_outchannels)

        self.bottleneck = DBConv(256, 256)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decoder4 = DBConv(256 + 256, 256)
        self.decoder3 = DBConv(256 + 256, 128)
        self.decoder2 = DBConv(128 + 128, 64)
        self.decoder1 = DBConv(64 + 64, 32)
        self.decoder0 = DBConv(32 + 32, 32)

        self.edge_att4 = SobelEdgeAttention(in_channels=256, ksize=3, use_residual=True)
        self.edge_att3 = SobelEdgeAttention(in_channels=128, ksize=3, use_residual=True)
        self.edge_att2 = SobelEdgeAttention(in_channels=64, ksize=3, use_residual=True)
        self.edge_att1 = SobelEdgeAttention(in_channels=32, ksize=3, use_residual=True)

        self.seg_head = nn.Conv2d(first_outchannels, num_classes, 1)

    def forward(self, x):
        pass1, pass2, pass3, pass4, pass5 = self.contracting_path(x)

        neck = self.bottleneck(pass5)

        output1 = torch.cat((neck, pass5), 1)
        output = self.decoder4(output1)

        output = self.up(output)
        output2 = torch.cat((output, self.edge_att4(pass4)), 1)
        output = self.decoder3(output2)

        output = self.up(output)
        output3 = torch.cat((output, self.edge_att3(pass3)), 1)
        output = self.decoder2(output3)

        output = self.up(output)
        output4 = torch.cat((output, self.edge_att2(pass2)), 1)
        output = self.decoder1(output4)

        output = self.up(output)
        output5 = torch.cat((output, self.edge_att1(pass1)), 1)
        output = self.decoder0(output5)

        seg = self.seg_head(output)

        return seg, neck
