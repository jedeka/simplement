"""Variational Autoencoder
Refs:
- https://github.com/pytorch/examples/blob/main/vae/main.py
- https://leimao.github.io/blog/PyTorch-Variational-Autoencoder/

TODO:
- visualize like this https://leimao.github.io/images/blog/2024-06-14-PyTorch-Variational-Autoencoder/sample_using_2d_std_normal_prior_inverse_cdf_29.png

"""

import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ***** utils ***** 
def show_images(images, title=""):
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 2, 2))
    for img, ax in zip(images, axes):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.show()

@torch.no_grad()
def visualize_vae_latent_space(
        model, dataloader, device='cuda', method="tsne", 
        save=False, show=True
    ):
    model.eval()
    latents = []
    labels = []
    for x, y in dataloader:
        x = x.to(device)
        mu, _ = model.encode(x.view(x.size(0), -1))
        latents.append(mu.cpu().numpy())
        labels.append(y.numpy())
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)

    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
    else:
        reducer = PCA(n_components=2)

    reduced = reducer.fit_transform(latents)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title(f"VAE Latent Space ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    if save:
        plt.savefig(f'vae_latent_{method}.png')
    if show:
        plt.show()
    plt.close()


# ***** models *****
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc21 = nn.Linear(400, 20)  # mu
        self.fc22 = nn.Linear(400, 20)  # logvar
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 28 * 28)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_vae(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, 28 * 28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


if __name__ == '__main__':
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    vae = VAE().to(device)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # VAE training loop
    for epoch in range(10):
        vae.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"VAE Training Epoch {epoch}")):
            data = data.to(device)
            optimizer_vae.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_vae(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer_vae.step()
        print(f"Epoch {epoch}, VAE Loss: {train_loss / len(train_loader.dataset):.4f}")

    # VAE sampling
    vae.eval()
    with torch.no_grad():
        z = torch.randn(10, 20).to(device)
        samples = vae.decode(z).view(-1, 1, 28, 28).cpu()
        show_images(samples, title="VAE Generated Samples")

    # Call the visualization function
    visualize_vae_latent_space(vae, train_loader, method="tsne", save=True, show=False)
