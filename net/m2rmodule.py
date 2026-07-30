import torch.nn as nn
from net.GDN import GDN


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,bias=False)


class conv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1x1=False, kernel_size=3, stride=1, padding=1):
        super(conv_ResBlock, self).__init__()
        self.conv1=conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2=conv(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gdn1=GDN(out_channels)
        self.gdn2=GDN(out_channels)
        self.prelu=nn.PReLU()
        self.use_conv1x1=use_conv1x1
        if use_conv1x1 == True:
            self.conv3=conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            
    def forward(self, x): 
        out=self.conv1(x)
        out=self.gdn1(out)
        out=self.prelu(out)
        out=self.conv2(out)
        out=self.gdn2(out)
        if self.use_conv1x1 == True:
            x=self.conv3(x)
        out=out+x
        out=self.prelu(out)
        return out 


class deconv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(deconv_ResBlock, self).__init__()
        self.deconv1=deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.deconv2=deconv(out_channels, out_channels, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.gdn1=GDN(out_channels)
        self.gdn2=GDN(out_channels)
        self.prelu=nn.PReLU()
        self.sigmoid=nn.Sigmoid()
        self.use_deconv1x1=use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3=deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding)
            
    def forward(self, x, activate_func='prelu'): 
        out=self.deconv1(x)
        out=self.gdn1(out)
        out=self.prelu(out)
        out=self.deconv2(out)
        out=self.gdn2(out)
        if self.use_deconv1x1 == True:
            x=self.deconv3(x)
        out=out+x
        if activate_func=='prelu':
            out=self.prelu(out)
        elif activate_func=='sigmoid':
            out=self.sigmoid(out)
        return out 


class M2RModule_Res(nn.Module):
    def __init__(self, channel):
        super(M2RModule_Res, self).__init__()

        # THIS is a UNet-like nerual network aiming at restoring the masked feature map
        self.channel = channel
        self.hidden_channel1 = channel*2
        self.hidden_channel2 = channel*4
        self.hidden_channel3 = channel*8
        self.avg_pool = nn.AvgPool2d(2)
        self.conv1 = conv_ResBlock(self.channel, self.hidden_channel1, use_conv1x1=True, kernel_size=3, stride=1, padding=1)

        # Downsampling convs
        self.down_conv1 = nn.Sequential(conv_ResBlock(self.hidden_channel1, self.hidden_channel2, use_conv1x1=True, kernel_size=3, stride=2, padding=1),
                                        conv_ResBlock(self.hidden_channel2, self.hidden_channel2, kernel_size=3, stride=1, padding=1))
        self.down_conv2 = nn.Sequential(conv_ResBlock(self.hidden_channel2, self.hidden_channel3, use_conv1x1=True, kernel_size=3, stride=2, padding=1),
                                        conv_ResBlock(self.hidden_channel3, self.hidden_channel3, kernel_size=3, stride=1, padding=1))
        
        # Projection
        self.conv2 = conv_ResBlock(self.hidden_channel3, self.hidden_channel3, kernel_size=1, stride=1, padding=0)

        # Upsampling convs
        self.up_conv2 = nn.Sequential(conv_ResBlock(self.hidden_channel3, self.hidden_channel3, kernel_size=3, stride=1, padding=1), 
                                      deconv_ResBlock(self.hidden_channel3, self.hidden_channel2, use_deconv1x1=True, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.up_conv1 = nn.Sequential(conv_ResBlock(self.hidden_channel2, self.hidden_channel2, kernel_size=3, stride=1, padding=1), 
                                      deconv_ResBlock(self.hidden_channel2, self.hidden_channel1, use_deconv1x1=True, kernel_size=3, stride=2, padding=1, output_padding=1))
        
        # Projection
        self.conv3 = conv_ResBlock(self.hidden_channel1, self.channel, use_conv1x1=True, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        f1 = self.conv1(x)
        f2 = self.down_conv1(f1)
        f3 = self.down_conv2(f2)
        
        f3_hat = self.conv2(f3)
        
        f2_hat = self.up_conv2(f3_hat + f3)
        f1_hat = self.up_conv1(f2_hat + f2)
        
        res = self.conv3(f1_hat + f1)

        return res