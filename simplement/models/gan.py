"""Generative Adversarial Network"""
import numpy as np 
import torch 
from torch import nn 
import torch.nn.functional as F


# ***** utils ***** 




# ***** models *****


# -- GAN Implementation -- [Insert previously defined GAN models, loss, and training loop here]
# GAN Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(-1, 28 * 28))

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

criterion = nn.BCELoss()

# GAN training loop
for epoch in range(30):
    for i, (imgs, _) in enumerate(tqdm(train_loader, desc=f"GAN Training Epoch {epoch}")):
        real = imgs.to(device)
        batch_size = real.size(0)

        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, 100).to(device)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch {epoch}, Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")

# GAN sampling
generator.eval()
with torch.no_grad():
    z = torch.randn(10, 100).to(device)
    samples = generator(z).view(-1, 1, 28, 28).cpu()
    show_images(samples, title="GAN Generated Samples")

