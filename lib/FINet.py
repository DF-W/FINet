import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.FIFormer import FIFormer
from ptflops import get_model_complexity_info

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=out_planes)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class UpsampleModule(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1, scale_factor=2):
        super(UpsampleModule, self).__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = BasicConv2d(in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.up(x) 
        x = self.conv(x)
       
        return x
    
class SFI(nn.Module):
    def __init__(self, in_planes):
        super(SFI, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes*2, in_planes, 3, padding=1, groups=in_planes) 
        
        self.conv2 = nn.Conv2d(2, 2, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()     
         
        self.conv3_1 = nn.Conv2d(in_planes, in_planes, 1)
        self.conv3_2 = nn.Conv2d(in_planes, in_planes, 1)
        
        self.conv4_1 = nn.Conv2d(in_planes, in_planes, 1)
        self.conv4_2 = nn.Conv2d(in_planes, in_planes, 1)
 
    def forward(self, x_r, x_b): 
        x = torch.cat([x_r, x_b], dim=1)
        x = self.conv1(x)
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv2(x)
        sa = self.sigmoid(x)
        
        out_r = self.conv4_1(self.conv3_1(x_r * sa[:,0,:,:].unsqueeze(1)) + x_r) 
        out_b = self.conv4_2(self.conv3_2(x_b * sa[:,1,:,:].unsqueeze(1)) + x_b)
        
        return out_r, out_b
    

class FINet(nn.Module):
    def __init__(self):
        super(FINet, self).__init__()

        self.backbone = FIFormer() 
        path = '/opt/data/private/Second/FINet/pretrained/FIFormer.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model['model'].items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.convr4 = BasicConv2d(512, 512, 3, padding=1)
        self.convb4 = BasicConv2d(512, 512, 3, padding=1)
        self.sfi4 = SFI(512)
        
        
        self.upr4_3 = UpsampleModule(in_planes=512, out_planes=256)
        self.upb4_3 = UpsampleModule(in_planes=512, out_planes=256)
        
        self.convr3 = BasicConv2d(512, 256, 3, padding=1)
        self.convb3 = BasicConv2d(512, 256, 3, padding=1)
        self.sfi3 = SFI(256)
        
        
        self.upr3_2 = UpsampleModule(in_planes=256, out_planes=128)
        self.upb3_2 = UpsampleModule(in_planes=256, out_planes=128)
        
        self.convr2 = BasicConv2d(256, 128, 3, padding=1)
        self.convb2 = BasicConv2d(256, 128, 3, padding=1)
        self.sfi2 = SFI(128)
        
        
        self.upr2_1 = UpsampleModule(in_planes=128, out_planes=64)
        self.upb2_1 = UpsampleModule(in_planes=128, out_planes=64)
        
        self.convr1 = BasicConv2d(128, 64, 3, padding=1)
        self.convb1 = BasicConv2d(128, 64, 3, padding=1)
        self.sfi1 = SFI(64)
        
        
        self.outr = nn.Conv2d(64, 1, 1)
        self.outb1 = nn.Conv2d(64, 1, 1)
        self.outb2 = nn.Conv2d(128, 1, 1)
        self.outb3 = nn.Conv2d(256, 1, 1)
        self.outb4 = nn.Conv2d(512, 1, 1)
        

    def forward(self, x):

        # backbone
        ssa = self.backbone(x)
        x1 = ssa[0]
        x2 = ssa[1]
        x3 = ssa[2]
        x4 = ssa[3]

        xr4 = self.convr4(x4)
        xb4 = self.convb4(x4)
        xr4, xb4 = self.sfi4(xr4, xb4)
        
        
        xr4_3 = self.upr4_3(xr4)
        xb4_3 = self.upb4_3(xb4)
        
        xr3 = self.convr3(torch.cat([x3, xr4_3], dim=1))
        xb3 = self.convb3(torch.cat([x3, xb4_3], dim=1))
        xr3, xb3 = self.sfi3(xr3, xb3)
        
        
        xr3_2 = self.upr3_2(xr3)
        xb3_2 = self.upb3_2(xb3)
        
        xr2 = self.convr2(torch.cat([x2, xr3_2], dim=1))
        xb2 = self.convb2(torch.cat([x2, xb3_2], dim=1))
        xr2, xb2 = self.sfi2(xr2, xb2)
        
        
        xr2_1 = self.upr2_1(xr2)
        xb2_1 = self.upb2_1(xb2)
        
        xr1 = self.convr1(torch.cat([x1, xr2_1], dim=1))
        xb1 = self.convb1(torch.cat([x1, xb2_1], dim=1))
        xr1, xb1 = self.sfi1(xr1, xb1)
        
        xr0 = self.outr(xr1)
        
        xr0 = F.interpolate(xr0, scale_factor=4, mode='bilinear', align_corners=True)
        
        ob1 = self.outb1(xb1)
        ob2 = self.outb2(xb2)
        ob3 = self.outb3(xb3)
        ob4 = self.outb4(xb4)
        
        return xr0, ob1, ob2, ob3, ob4
  
        
if __name__ == '__main__':
    model = FINet().cuda()
    
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
