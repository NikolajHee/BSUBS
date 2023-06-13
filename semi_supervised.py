import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn import svm
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_Categorical(x, x_prob, num_classes):
    x_one_hot = nn.functional.one_hot(x, num_classes=num_classes)
    log_p = torch.sum(x_one_hot * torch.log(x_prob), dim=-1)
    return log_p


def log_standard_Categorical(x, num_classes):
    x_one_hot = nn.functional.one_hot(x, num_classes=num_classes).float()
    x_prob = nn.functional.softmax(torch.ones_like(x_one_hot), dim=1)
    log_p = torch.sum(x_one_hot * torch.log(x_prob + 1e-7), dim=1)
    return log_p


def log_Normal(x, mu, log_var):
    D = x.shape[1]
    log_p = -.5 * ((x - mu) ** 2. * torch.exp(-log_var) +
                   log_var + D * np.log(2 * np.pi))
    return log_p


def log_standard_Normal(x):
    D = x.shape[1]
    log_p = -.5 * (x ** 2. + D * np.log(2 * np.pi))
    return log_p


class encoder(nn.Module):
    def __init__(self, input_dim, middel_dim, channels):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            32 * self.input_dim * self.input_dim, middel_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, middel_dim, channels):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels

        self.input = nn.Linear(
            middel_dim, 32 * self.input_dim * self.input_dim)
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.fully_connected = nn.Linear(
            channels * self.input_dim * self.input_dim, 2 * channels * self.input_dim * self.input_dim)
        self.softmax = nn.Softmax(dim=2)

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
        return x


class classifier(nn.Module):
    def __init__(self, classes, input_dim=68, channels=3):
        super(classifier, self).__init__()
        self.classes = classes
        self.input_dim = input_dim
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, 8, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            16 * self.input_dim * self.input_dim, self.classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 16 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        x = self.softmax(x)
        return x


class Semi_supervised_VAE(nn.Module):
    def __init__(self, classes, latent_dim, input_dim, channels):
        super(Semi_supervised_VAE, self).__init__()

        self.latent_dim = latent_dim
        self.classes = classes
        self.channels = channels
        self.input_dim = input_dim
        self.alpha = 0.1
        self.middel_dim = 32 * input_dim * input_dim

        self.eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)

        self.encoder = encoder(input_dim, self.middel_dim, channels)
        self.decoder = decoder(input_dim, self.middel_dim, channels)
        self.classifier = classifier(classes, input_dim, channels)

        self.middel_1 = nn.Linear(self.middel_dim + classes, latent_dim)
        self.middel_2 = nn.Linear(self.middel_dim + classes, latent_dim)
        self.middel_3 = nn.Linear(latent_dim + classes, self.middel_dim)

    def encode(self, x, y_hat):
        m = self.encoder.forward(x)
        mu = self.middel_1(torch.cat((m, y_hat), dim=1))
        log_var = self.middel_2(torch.cat((m, y_hat), dim=1))
        return mu, log_var

    def reparameterization(self, mu, log_var):
        return mu + torch.exp(0.5*log_var) * self.eps

    def decode(self, z, y_hat):
        z = self.middel_3(torch.cat((z, y_hat), dim=1))
        mu, log_var = torch.split(
            self.decoder.forward(z), self.channels * self.input_dim * self.input_dim, dim=1)
        std = torch.exp(log_var)
        return mu, std

    def classify(self, x):
        return self.classifier.forward(x)

    def ELBO_unlabelled(self, y_prob, reconstruction_error, KL):
        H = - torch.sum(torch.log(y_prob) * y_prob, dim=-1).mean()
        K_1 = - log_standard_Categorical(torch.tensor([0]), self.classes)

        Lxy = reconstruction_error + KL + K_1
        q_Lxy = torch.sum(y_prob * Lxy.view(-1, 1), dim=-1).mean()

        ELBO = q_Lxy + H

        return ELBO

    def ELBO_labelled(self, y, y_prob, reconstruction_error, KL):
        K_1 = - log_standard_Categorical(y, self.classes)
        log_like_y = - log_Categorical(y, y_prob, self.classes)

        Lxy = (reconstruction_error + KL + K_1).mean()

        ELBO = Lxy + self.alpha * log_like_y.mean()

        return ELBO

    def forward(self, x, y):
        # y_onehot = nn.functional.one_hot(y, num_classes=self.classes).float()

        idx = y != 0  # "DMSO"

        # y_labelled = y_onehot[idx]
        y_labelled = y[idx]

        y_hat = self.classify(x)
        mu, log_var = self.encode(x, y_hat)

        z = self.reparameterization(mu, log_var)

        decode_mu, decode_var = self.decode(z, y_hat)

        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)
        log_like = (1 / (2 * decode_var) * nn.functional.mse_loss(decode_mu, x.flatten(
            start_dim=1, end_dim=-1), reduction="none"))

        reconstruction_error = torch.sum(log_like, dim=-1)
        KL = - torch.sum(log_prior - log_posterior, dim=-1)

        y_prob_unlabelled = y_hat[~idx]
        y_prob_labelled = y_hat[idx]

        reconstruction_error_unlabelled = reconstruction_error[~idx]
        reconstruction_error_labelled = reconstruction_error[idx]

        KL_unlabelled = KL[~idx]
        KL_labelled = KL[idx]

        if y_prob_unlabelled.shape[0] == 0:
            ELBO_labelled = self.ELBO_labelled(
                y_labelled, y_prob_labelled, reconstruction_error_labelled, KL_labelled)
            ELBO_unlabelled, reconstruction_error_unlabelled, KL_unlabelled = 0, 0, 0

        elif y_prob_labelled.shape[0] == 0:
            ELBO_unlabelled = self.ELBO_unlabelled(
                y_prob_unlabelled, reconstruction_error_unlabelled, KL_unlabelled)
            ELBO_labelled, reconstruction_error_labelled, KL_labelled = 0, 0, 0

        else:
            ELBO_unlabelled = self.ELBO_unlabelled(
                y_prob_unlabelled, reconstruction_error_unlabelled, KL_unlabelled)
            ELBO_labelled = self.ELBO_labelled(
                y_labelled, y_prob_labelled, reconstruction_error_labelled, KL_labelled)

        reconstruction_error = reconstruction_error.mean()
        KL = KL.mean()

        ELBO = ELBO_unlabelled + ELBO_labelled

        tqdm.write(
            f"ELBO: {ELBO.item()}, Reconstruction error: {reconstruction_error.item()}, Regularizer: {KL.item()}")

        return ELBO, reconstruction_error, KL

    def initialise(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.sparse_(m.weight, sparsity=1)
                if m.bias is not None:
                    m.bias.data.fill_(3)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.dirac_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(_init_weights)

    def train_VAE(self, dataloader, epochs, lr=10e-6):
        parameters = [param for param in self.parameters()
                      if param.requires_grad == True]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        self.train()
        self.initialise()

        ELBOs = []
        reconstruction_errors = []
        KLs = []

        self.alpha = 0.1 * dataloader.__len__()
        for epoch in tqdm(range(epochs)):
            for x_batch, y_batch in tqdm(dataloader):
                x = x_batch.to(device)
                y = y_batch.to(device)

                optimizer.zero_grad()

                ELBO, reconstruction_error, KL = self.forward(x, y)

                ELBOs.append(ELBO.item())
                reconstruction_errors.append(reconstruction_error.item())
                KLs.append(KL.item())

                ELBO.backward()
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {ELBO.item()}, Reconstruction Error: {reconstruction_error.item()}, Regularizer: {KL.item()}")

        mu, log_var = self.encode(x)
        latent_space = self.reparameterization(mu, log_var)

        return self.encoder, self.decoder, reconstruction_errors, KL, latent_space


latent_dim = 50
epochs = 5
batch_size = 10

pixel_range = 256
input_dim = 28
channels = 1
classes = 10

train_size = 1000
test_size = 1000

trainset = datasets.MNIST(
    root='./MNIST', train=True, download=True, transform=None)
testset = datasets.MNIST(
    root='./MNIST', train=False, download=True, transform=None)
Xy_train = TensorDataset(trainset.data[:train_size].reshape(
    (train_size, channels, input_dim, input_dim)).float(), trainset.targets[:train_size])
Xy_train = DataLoader(Xy_train, batch_size=batch_size, shuffle=True)
Xy_test = TensorDataset(testset.data[:test_size].reshape(
    (test_size, channels, input_dim, input_dim)).float(), testset.targets[:test_size])
Xy_test = DataLoader(Xy_test, batch_size=batch_size, shuffle=True)

# X = np.load("image_matrix.npz")["images"][:1000]
# X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)

VAE = Semi_supervised_VAE(classes=classes, latent_dim=latent_dim,
                          input_dim=input_dim, channels=channels).to(device)

# print("VAE:")
# The summary function from torchsummary.py has been modified to work with the code by changing the following lines (ln 101):
# total_input_size = abs(np.prod([np.prod(in_size) for in_size in input_size]) * batch_size * 4. / (1024 ** 2.))
# summary(VAE, input_size=[(batch_size, channels, input_dim, input_dim), (batch_size,)])

encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE.train_VAE(
    dataloader=Xy_train, epochs=epochs)
