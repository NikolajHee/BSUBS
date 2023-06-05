import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_Categorical(x, theta, num_classes):
    x_one_hot = nn.functional.one_hot(
        x.flatten(start_dim=1, end_dim=-1).long(), num_classes=num_classes)
    log_p = torch.sum(
        x_one_hot * torch.log(theta), dim=-1)
    return log_p


def log_Normal(x, mu, log_var):
    D = x.shape[1]
    log_p = -.5 * ((x - mu) ** 2. * torch.exp(-log_var) +
                   log_var + D * torch.log(2 * torch.tensor(np.pi)))
    return log_p


def log_standard_Normal(x):
    D = x.shape[1]
    log_p = -.5 * (x ** 2. + D * torch.log(2 * torch.tensor(np.pi)))
    return log_p


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            32 * self.input_dim * self.input_dim, 2 * latent_dim)

        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.fully_connected.weight)
        self.conv1.bias.data.fill_(1)
        self.conv2.bias.data.fill_(1)
        self.fully_connected.bias.data.fill_(1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels, pixel_range):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.pixel_range = pixel_range

        self.input = nn.Linear(
            latent_dim, 32 * self.input_dim * self.input_dim)
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.fully_connected = nn.Linear(
            channels * self.input_dim * self.input_dim, channels * self.input_dim * self.input_dim * pixel_range)
        self.softmax = nn.Softmax(dim=2)  # changed dim from 1 to 2

    def forward(self, x):
        x = self.input(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32, self.input_dim, self.input_dim)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, self.channels * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        x = x.view(-1, self.channels * self.input_dim *
                   self.input_dim, self.pixel_range)
        x = self.softmax(x)
        return x


class VAE(nn.Module):
    def __init__(self, X, pixel_range, latent_dim, input_dim, channels):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels)
        self.decoder = decoder(input_dim, latent_dim, channels, pixel_range)
        self.pixel_range = pixel_range
        self.latent_dim = latent_dim

        self.data_length = len(X)
        self.eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
        # self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(latent_dim), covariance_matrix=torch.eye(latent_dim))

    def encode(self, x):
        mu, log_var = torch.split(
            self.encoder.forward(x), self.latent_dim, dim=1)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        return mu + torch.exp(0.5*log_var) * self.eps

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var)

        theta = self.decode(z)

        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)
        log_like = log_Categorical(x, theta, self.pixel_range)

        reconstruction_error = - torch.sum(log_like, dim=-1).mean()
        regularizer = - torch.sum(log_prior - log_posterior, dim=-1).mean()

        elbo = reconstruction_error + regularizer

        tqdm.write(
            f"ELBO: {elbo.detach()}, Reconstruction error: {reconstruction_error.detach()}, Regularizer: {regularizer.detach()}")

        return elbo, reconstruction_error, regularizer

    def train_VAE(self, dataloader, epochs, batch_size, lr=10e-5):
        parameters = [param for param in self.parameters()
                      if param.requires_grad == True]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        reconstruction_errors = []
        regularizers = []

        self.train()
        for epoch in tqdm(range(epochs)):
            for batch in tqdm(dataloader):
                x = batch.to(device)
                x = torch.round(x/255)
                optimizer.zero_grad()
                elbo, reconstruction_error, regularizer = self.forward(x)
                reconstruction_errors.append(
                    reconstruction_error.cpu().detach().numpy())
                regularizers.append(regularizer.cpu().detach().numpy())
                elbo.backward(retain_graph=True)
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {elbo.detach()}, Reconstruction Error: {reconstruction_error.detach()}, Regularizer: {regularizer.detach()}")

        mu, log_var = self.encode(x)
        latent_space = self.reparameterization(mu, log_var)

        return self.encoder, self.decoder, reconstruction_errors, regularizers, latent_space


def generate_image(X, encoder, decoder, latent_dim, channels, input_dim, batch_size=1):
    encoder.eval()
    decoder.eval()
    X = X.to(device)
    mu, log_var = torch.split(encoder.forward(X), latent_dim, dim=1)
    eps = torch.normal(mean=0, std=torch.ones(latent_dim))
    z = mu + torch.exp(0.5*log_var) * eps
    theta = decoder.forward(z)
    image = torch.argmax(theta, dim=-1)
    image = image.reshape((batch_size, channels, input_dim, input_dim))
    image = torch.permute(image, (0, 2, 3, 1))
    image = image.numpy()
    return image
