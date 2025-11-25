from module.modules import *

class MixMamba_TransNet(nn.Module):

  def __init__(self,backbone):
    super().__init__()
    self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]
    for i in [1, 4, 7, 10]:
        self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

    self.paspp = PASPP(512,512)

    self.connect1 = SkipConnection(512)
    self.connect2 = SkipConnection(320)
    self.connect3 = SkipConnection(128)
    self.connect4 = SkipConnection(64)

    self.se1 = SEBlock(512)
    self.se2 = SEBlock(320)
    self.se3 = SEBlock(128)
    self.se4 = SEBlock(64)

    self.ca1 = CoordAtt(512, 512)
    self.ca2 = CoordAtt(320, 320)
    self.ca3 = CoordAtt(128, 128)
    self.ca4 = CoordAtt(64, 64)

    self.cbam1 = CBAM(512)
    self.cbam2 = CBAM(320)
    self.cbam3 = CBAM(128)
    self.cbam4 = CBAM(64)

    self.up1 = up_conv(512,320)
    self.up2 = up_conv(320,128)
    self.up3 = up_conv(128,64)

    self.upconv1 = conv_block(1024,512)
    self.upconv2 = conv_block(640,320)
    self.upconv3 = conv_block(256,128)
    self.upconv4 = conv_block(128,64)

    self.mapreduce1 = MapReduce(512)
    self.mapreduce2 = MapReduce(320)
    self.mapreduce3 = MapReduce(128)
    self.mapreduce4 = MapReduce(64)

    self.mamba1 = MambaMLPMixer(dim=512)
    self.mamba2 = MambaMLPMixer(dim=320)
    self.mamba3 = MambaMLPMixer(dim=128)
    self.mamba4 = MambaMLPMixer(dim=64)

    self.sigmoid = nn.Sigmoid()

  def get_pyramid(self,x):
      pyramid = []
      B = x.shape[0]
      for i, module in enumerate(self.backbone):
          if i in [0, 3, 6, 9]:
              x, H, W = module(x)
          elif i in [1, 4, 7,10]:
              for sub_module in module:
                  x = sub_module(x, H, W)
          else:
              x = module(x)
              x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
              pyramid.append(x)
      return pyramid

  def forward(self,x):

    H, W = x.size()[2:]

    pyramid=self.get_pyramid(x)


    pyramid3=self.paspp(pyramid[3])                                         #(512,8,8)
    pyramid[3]=self.connect1(pyramid[3])                                    #(512,8,8)
    pyramid[3]=self.se1(pyramid[3])
    pyramid[3]=torch.cat((pyramid[3],pyramid3),dim=1)                       #(1024,8,8)
    pyramid[3]=self.upconv1(pyramid[3])                                     #(512,8,8)
    pyramid[3]=self.ca1(pyramid[3])
    pyramid[3]=self.mamba1(pyramid[3])                                      #(512,8,8)
    pyramid[3]=self.cbam1(pyramid[3])


    x1 = self.mapreduce1(pyramid[3])                                        #(1,8,8)
    x1 = F.interpolate(x1, (H, W), mode="bilinear", align_corners=False)    #(1,256,256)


    pyramid[2]=self.connect2(pyramid[2])                                    #(320,16,16)
    pyramid[2]=self.se2(pyramid[2])
    pyramid[3]=self.up1(pyramid[3])                                         #(320,16,16)
    pyramid[2]=torch.cat((pyramid[3],pyramid[2]),dim=1) #(640,16,16)
    pyramid[2]=self.upconv2(pyramid[2])                                     #(320,16,16)
    pyramid[2]=self.ca2(pyramid[2])
    pyramid[2]=self.mamba2(pyramid[2])                                      #(320,16,16)
    pyramid[2]=self.cbam2(pyramid[2])


    x2 = self.mapreduce2(pyramid[2])                                        #(1,16,16)
    x2 = F.interpolate(x2, (H, W), mode="bilinear", align_corners=False)    #(1,256,256)


    pyramid[1]=self.connect3(pyramid[1])                                    #(128,32,32)
    pyramid[1]=self.se3(pyramid[1])
    pyramid[2]=self.up2(pyramid[2])                                         #(128,32,32)
    pyramid[1]=torch.cat((pyramid[2],pyramid[1]),dim=1)                     #(256,32,32)
    pyramid[1]=self.upconv3(pyramid[1])                                     #(128,32,32)
    pyramid[1]=self.ca3(pyramid[1])
    pyramid[1]=self.mamba3(pyramid[1])                                      #(128,32,32)
    pyramid[1]=self.cbam3(pyramid[1])


    x3 = self.mapreduce3(pyramid[1])                                        #(1,32,32)
    x3 = F.interpolate(x3, (H, W), mode="bilinear", align_corners=False)    #(1,256,256)


    pyramid[0]=self.connect4(pyramid[0])                                    #(64,64,64)
    pyramid[0]=self.se4(pyramid[0])
    pyramid[1]=self.up3(pyramid[1])                                         #(64,64,64)
    pyramid[0]=torch.cat((pyramid[1],pyramid[0]),dim=1)                     #(128,64,64)
    pyramid[0]=self.upconv4(pyramid[0])                                     #(64,64,64)
    pyramid[0]=self.ca4(pyramid[0])
    pyramid[0]=self.mamba4(pyramid[0])                                      #(64,64,64)
    pyramid[0]=self.cbam4(pyramid[0])


    x4 = self.mapreduce4(pyramid[0])                                        #(1,64,64)
    x4 = F.interpolate(x4, (H, W), mode="bilinear", align_corners=False)    #(1,256,256)

    x1 = self.sigmoid(x1)
    x2 = self.sigmoid(x2)
    x3 = self.sigmoid(x3)
    x4 = self.sigmoid(x4)

    return x4, x3, x2, x1