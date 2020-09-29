""" 以下代码仅作为DC-GAN模型的实现参考
"""

import torch
import torch.nn as nn

#  生成器的定义
class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

# 判别器的定义
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):

        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

# DCGAN的训练代码

def train():
    # nz: 隐含变量的维度
    # ngf, ncf: 生成器和判别器的特征维度
    netG = Generator(nz, ngf).to(device)
    netD = Discriminator(nc, ndf).to(device)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr,
        betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,
        betas=(opt.beta1, 0.999))
    real_label = 1 # 真实图像标签
    fake_label = 0 # 生成图像标签
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            # 判别器梯度置为0
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            # 定义输入数据
            label = torch.full((batch_size,), real_label, device=device)
            output = netD(real_cpu)
            # 定义判别器相对于真实图像损失函数
            errD_real = criterion(output, label)
            # 梯度反向传播，相对于真实图像
            errD_real.backward()
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            # 定义判别器相对于生成图像损失函数
            errD_fake = criterion(output, label)
            # 梯度反向传播，相对于生成图像
            errD_fake.backward()
            # 计算判别器总的损失函数：真实图像损失函数+生成图像损失函数
            errD = errD_real + errD_fake
            # 优化判别器
            optimizerD.step()        
    
            # 生成器梯度置为0
            netG.zero_grad()
            # 注意这里是real_label，相对于前面fake_label
            label.fill_(real_label)
            output = netD(fake)
            # 定义判别器相对于真实图像损失函数
            errG = criterion(output, label)
            # 梯度反向传播，相对于生成器
            errG.backward()
            # 优化生成器
            optimizerG.step()

