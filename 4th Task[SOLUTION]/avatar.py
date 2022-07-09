import os

import cv2
import torch
import torchvision
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

BATCH_SIZE = 128
IMAGE_SIZE = (28, 28)
NOISE_DIM = 100
LOW_RES_SIZE = IMAGE_SIZE

class Reshape(nn.Module):
    def __init__(self, new_shape: tuple):
        super().__init__()
        self.new_shape = new_shape
        
    def forward(self, z):
        return z.view(z.size(0), *self.new_shape)

class ADCGenerator(nn.Module):
    def __init__(self, seed_size: int, img_shape: tuple):
        super().__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(seed_size, 4*4*256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4*4*256),
            nn.LeakyReLU(),
            Reshape((256, 4, 4)),
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh(),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=5, stride=3, padding=2, output_padding=2, bias=False),
            nn.Tanh(),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh(),
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.permute(0, 2, 3, 1)  # make channel axis last
        return img

class ADCDiscriminator(nn.Module):
    def __init__(self, img_shape: tuple):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(img_shape[2], 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, momentum=0.7),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.7),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(512*13*13, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, img):
        img_chan = img.permute(0, 3, 1, 2)  # channels first
        return self.model(img_chan)

class AVATARGAN(pl.LightningModule):
    def __init__(self, width, height, channels, seed_size=NOISE_DIM, lr=1.5e-4, b1=0.5, b2=0.999):
        super().__init__()
        self.save_hyperparameters()  # make <arg> available as self.hparams.<arg>
        
        img_shape = (width, height, channels)
        self.generator = ADCGenerator(seed_size=self.hparams.seed_size, img_shape=img_shape)
        self.discriminator = ADCDiscriminator(img_shape=img_shape)
        
        self.validation_z = torch.randn(8, seed_size)
    
    def forward(self, z):
        return self.generator(z)

    def gan_loss(self, y_hat, y):
        loss = nn.BCELoss()
        return  loss(y_hat, y)# TODO: implement GAN loss (hint: binary cross entropy)
    
    def training_step(self, imgs, batch_idx, optimizer_idx):
        z = torch.randn(imgs.size(0), self.hparams.seed_size)
        z = z.type_as(imgs)  # move to same device as imgs
        
        if optimizer_idx == 0:  # generator
            self.generated_imgs = self(z)
            
            valid = torch.ones(imgs.size(0), 1)  # all fake, but we want to be real
            valid = valid.type_as(imgs)
            
            g_loss = self.gan_loss(self.discriminator(self(z)), valid)
            self.g_loss = g_loss
            self.log('g_loss', g_loss, prog_bar=True)
            return {'loss': g_loss}
        
        if optimizer_idx == 1:  # discriminator
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_loss = self.gan_loss(self.discriminator(imgs), valid)# TODO: loss for `discriminator(imgs)` and `valid`
            
            fake = torch.zeros(imgs.size(0), 1)# TODO: zero vector for fake_loss computation
            fake = fake.type_as(imgs)

            fake_loss = self.gan_loss(self.discriminator(self(z).detach()), fake)
            
            d_loss = (real_loss + fake_loss) / 2
            self.d_loss = d_loss
            self.log('d_loss', d_loss, prog_bar=True)
            return {'loss': d_loss}
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        # send results for validation_z to TensorBoard
        z = self.validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self(z).permute(0, 3, 1, 2)  # channels before pixels
        grid = torchvision.utils.make_grid(sample_imgs)
        
        torchvision.utils.save_image(grid, f'img_logs/epoch_{self.current_epoch}.png')
        
        self.logger.experiment.add_scalar('Gen_loss/Train', self.g_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Discr_loss/Train', self.d_loss, self.current_epoch)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)