import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_Categorical(x, theta, num_classes):
    x_one_hot = nn.functional.one_hot(
        x.flatten(start_dim=1, end_dim=-1).long(), num_classes=num_classes)
    log_p = torch.sum(
        x_one_hot * torch.log(torch.clamp(theta, 10e-4, 1.-10e-4)), dim=-1)
    return log_p


def log_Normal(x, mu, log_var):
    D = x.shape[-1]
    log_p = -.5 * ((x - mu) ** 2. * torch.exp(-log_var) +
                   log_var + D * np.log(2 * np.pi))
    return log_p


def log_standard_Normal(x):
    D = x.shape[-1]
    log_p = -.5 * (x ** 2. + D * np.log(2 * np.pi))
    return log_p


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            32 * self.input_dim * self.input_dim, 2 * latent_dim)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.fully_connected.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.input = nn.Linear(
            latent_dim, 32 * self.input_dim * self.input_dim)
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.fully_connected = nn.Linear(
            channels * self.input_dim * self.input_dim, channels * self.input_dim * self.input_dim * 256)
        self.softmax = nn.Softmax(dim=1)

        nn.init.xavier_normal_(self.input.weight)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.fully_connected.weight)

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
        x = x.view(-1, self.channels * self.input_dim * self.input_dim, 256)
        x = self.softmax(x)
        return x


class VAE(nn.Module):
    def __init__(self, X, pixel_range, latent_dim, input_dim, channels):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels)
        self.decoder = decoder(input_dim, latent_dim, channels)
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

    def ELBO(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var)
        theta = self.decode(z)
        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)
        log_con_like = log_Categorical(x, theta, self.pixel_range)
        reconstruction_error = (torch.sum(log_con_like, dim=-1)).mean()
        regularizer = (- torch.sum(log_posterior - log_prior, dim=-1)).mean()
        elbo = - (reconstruction_error + regularizer)
        # print(elbo, reconstruction_error, regularizer)
        return elbo, reconstruction_error, regularizer

    def train_VAE(self, X, epochs, batch_size, lr=10e-10):
        parameters = [param for param in self.parameters()
                      if param.requires_grad == True]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        reconstruction_errors = []
        regularizers = []
        for epoch in tqdm(range(epochs)):
            for i in tqdm(range(0, self.data_length, batch_size)):
                x = X[i:i+batch_size].to(device)
                optimizer.zero_grad()
                elbo, reconstruction_error, regularizer = self.ELBO(x)
                reconstruction_errors.append(
                    reconstruction_error.detach().cpu().numpy())
                regularizers.append(regularizer.detach().cpu().numpy())
                elbo.backward(retain_graph=True)
                optimizer.step()
            if epochs == epoch + 1:
                mu, log_var = self.encode(x)
                latent_space = self.reparameterization(mu, log_var)
            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {elbo}, Reconstruction Error: {reconstruction_error}, Regularizer: {regularizer}")
        return self.encoder, self.decoder, reconstruction_errors, regularizers, latent_space


def generate_image(X, encoder, decoder, latent_dim, channels, input_dim):
    X = X.to(device)
    mu, log_var = torch.split(encoder.forward(X), latent_dim, dim=1)
    eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
    z = mu + torch.exp(0.5*log_var) * eps
    theta = decoder.forward(z)
    image = torch.argmax(theta, dim=-1)
    image = image.reshape((channels, input_dim, input_dim))
    image = torch.permute(image, (1, 2, 0))
    image = image.cpu().numpy()
    return image



latent_dim = 200
epochs = 10
batch_size = 30

pixel_range = 256
input_dim = 28
channels = 1


trainset = datasets.MNIST(
    root='./MNIST', train=True, download=True, transform=None)
testset = datasets.MNIST(
    root='./MNIST', train=False, download=True, transform=None)
X_train = trainset.data.reshape(
    (len(trainset), channels, input_dim, input_dim)).float()
X_test = testset.data.reshape(
    (len(testset), channels, input_dim, input_dim)).float()

# X = np.load("image_matrix.npz")["images"][:1000]

VAE = VAE(X_train, pixel_range=pixel_range,
          latent_dim=latent_dim, input_dim=input_dim, channels=channels).to(device)
encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE.train_VAE(
    X=X_train, epochs=epochs, batch_size=batch_size)


np.savez("latent_space.npz", latent_space=latent_space.detach().cpu().numpy())
np.savez("reconstruction_errors.npz", reconstruction_errors=np.array(reconstruction_errors))

np.savez("regularizers.npz", regularizers=np.array(regularizers))


generated_images = []
for i in range(9):
    image = generate_image(X_test[i], encoder_VAE, decoder=decoder_VAE,
                           latent_dim=latent_dim, channels=channels, input_dim=input_dim)
    generated_images.append(image)

np.savez("generated_images.npz", generated_images=np.array(generated_images))

torch.save(encoder_VAE, "encoder.pt")
torch.save(decoder_VAE, "decoder.pt")
