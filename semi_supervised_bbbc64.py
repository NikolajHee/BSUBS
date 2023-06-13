import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
import os


# tod o: seed


# todo: latent space
# todo: decoder/encoder
# todo: elbo / REs / KLs
# todo: save test_elbo, test_re, test_kl

# to(device) in unlabbeled_elbo and labelled_elbo

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
        self.fully_connected = nn.Linear(
            16 * self.input_dim * self.input_dim, middel_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, 16 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, middel_dim, channels):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels

        self.input = nn.Linear(
            middel_dim,   16 * self.input_dim * self.input_dim)
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.output = nn.Linear(channels * self.input_dim * self.input_dim,
                                2 * channels * self.input_dim * self.input_dim)


    def forward(self, x):
        x = self.input(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, 16, self.input_dim, self.input_dim)
        x = self.conv2(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1,  self.channels * self.input_dim * self.input_dim)
        x = self.output(x)
        return x



class classifier(nn.Module):
    def __init__(self, classes, input_dim=68, channels=3):
        super(classifier, self).__init__()
        self.classes = classes
        self.input_dim = input_dim
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            16 * self.input_dim * self.input_dim, self.classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.01)(x)
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
        self.middel_dim = 8 # input_dim * input_dim

        self.eps = torch.normal(mean=0, std=torch.ones(self.latent_dim)).to(device)

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
        K_1 = - (log_standard_Categorical(torch.tensor([0]), self.classes)).to(device)

        Lxy = reconstruction_error + KL + K_1
        q_Lxy = torch.sum(y_prob * Lxy.view(-1, 1), dim=-1).mean()

        ELBO = q_Lxy + H

        return ELBO

    def ELBO_labelled(self, y, y_prob, reconstruction_error, KL):
        K_1 = - (log_standard_Categorical(y, self.classes)).to(device)
        log_like_y = - log_Categorical(y, y_prob, self.classes)

        Lxy = (reconstruction_error + KL + K_1).mean()

        ELBO = Lxy + self.alpha * log_like_y.mean()

        return ELBO

    def forward(self, x, y, save_latent=False):
        # y_onehot = nn.functional.one_hot(y, num_classes=self.classes).float()

        idx =  y != 0 # "DMSO"

        # y_labelled = y_onehot[idx]
        y_labelled = y[idx]

        y_hat = self.classify(x)
        mu, log_var = self.encode(x, y_hat)

        z = self.reparameterization(mu, log_var)

        decode_mu, decode_var = self.decode(z, y_hat)

        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)
        log_like = (1 / (2 * decode_var) * nn.functional.mse_loss(decode_mu, x.flatten(
            start_dim=1, end_dim=-1), reduction="none")) + 0.5 * torch.log(decode_var) + 0.5 * torch.log(2 * torch.tensor(np.pi))

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

        return (ELBO, reconstruction_error, KL) if not save_latent else (ELBO, reconstruction_error, KL, z)

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

    def train_VAE(self, dataloader, epochs, lr=10e-5):
        parameters = [
            param for param in self.parameters() if param.requires_grad == True
        ]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        REs = []
        KLs = []
        ELBOs = []

        #self.initialise()
        self.train()
        for epoch in tqdm(range(epochs)):
            for batch in dataloader:
                x = batch['image'].to(device)
                y = batch['moa'].to(device)

                optimizer.zero_grad()
                elbo, RE, KL = self.forward(x, y)
                
                REs, KLs, ELBOs = REs + [RE.item()], KLs + [KL.item()], ELBOs + [elbo.item()]

                elbo.backward()
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {elbo.item()}, Reconstruction Error: {RE.item()}, Regularizer: {KL.item()}"
            )

        return (
            self.encoder,
            self.decoder,
            REs,
            KLs,
            ELBOs
        )

    def test_VAE(self, dataloader, save_latent=False, results_folder=''):
        # only wors if len of dataloader is divisible by batch_size
        REs, KLs, ELBOs = [], [], []

        moa, compound = [], []

        latent = np.zeros((self.latent_dim, len(dataloader) * dataloader.batch_size)).T
        self.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader)):
                x = batch["image"].to(device)
                y = batch["moa"].to(device)

                moa, compound = moa + batch["moa_name"], compound + batch["compound"]

                if save_latent:
                    elbo, RE, KL, z = self.forward(x, y, save_latent=save_latent)
                    z = z.cpu().detach().numpy()
                    latent[i*dataloader.batch_size:(i+1)*dataloader.batch_size, :] = z

                else:
                    elbo, RE, KL = self.forward(x, y)

                REs, KLs, ELBOs = REs + [RE.item()], KLs + [KL.item()], ELBOs + [elbo.item()]

            if save_latent: 
                np.savez(results_folder + "latent_space.npz", z=latent, labels=moa, compound=compound)

            return REs, KLs, ELBOs
        



def generate_image(X, vae, latent_dim, channels, input_dim, batch_size=1):
    vae.eval()
    X = X.to(device)
    y_hat = vae.classify(X)
    mu, log_var = vae.encode(X, y_hat)
    eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
    z = mu + torch.exp(0.5 * log_var) * eps
    mean, var = vae.decode(z, y_hat)
    image = torch.normal(mean=mean, std=torch.sqrt(var)).to(device)
    image = image.view(channels, input_dim, input_dim)
    image = image.clip(0,1).detach().cpu().numpy()
    return image


def plot_1_reconstruction(image, 
                        vae,
                        name, 
                        latent_dim, 
                        channels, 
                        input_dim, 
                        results_folder=''):
    recon_image = generate_image(
        image,
        vae=vae,
        latent_dim=latent_dim,
        channels=channels,
        input_dim=input_dim,
    )


    fig, ax = plt.subplots(1, 2, figsize=(10,6))

    ax = ax.flatten()
    ax[0].imshow(image.reshape((input_dim,input_dim,channels)), cmap="gray")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Original',  size=22,fontweight="bold")
    ax[1].imshow(recon_image.reshape((input_dim,input_dim,channels)), cmap="gray")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Reconstruction', size=22,fontweight="bold")
    fig.suptitle(name,  size=26, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_folder + name +'.png')
    plt.show()
    plt.close()

def plot_ELBO(REs, KLs, ELBOs, name, results_folder):
    fig = plt.figure(figsize=(10,6))
    plt.plot(REs, label='Reconstruction Error', color='red', 
             linestyle='--', linewidth=2, alpha=0.5)
    
    plt.plot(KLs, label='Regularizer', color='blue', 
             linestyle='--', linewidth=2, alpha=0.5)
    
    plt.plot(ELBOs, label='ELBO', color='black', 
             linestyle='-', linewidth=4, alpha=0.5)
    plt.legend(fontsize=15)
    plt.title(name, size=26, fontweight='bold')
    plt.xlabel('Iterations',  size=22, fontweight='bold')
    plt.ylabel('Loss',  size=22, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig(results_folder + name + '.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    latent_dim = 300
    epochs = 100
    batch_size = 100

    classes = 13

    input_dim = 68
    channels = 3

    train_size = 10_000
    test_size = 2_000

    # latent_dim = 10
    #epochs, batch_size, train_size = 2, 10, 10

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    from dataloader import BBBC

    main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"
    # main_path = "/Users/nikolaj/Fagprojekt/Data/"


    exclude_dmso = False
    shuffle = True

    subset = (train_size, test_size)

    dataset_train = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                            meta_path=main_path + "metadata.csv",
                            subset=subset,
                            test=False,
                            exclude_dmso=exclude_dmso,
                            shuffle=shuffle)

    dataset_test = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                            meta_path=main_path + "metadata.csv",
                            subset=subset,
                            test=True,
                            exclude_dmso=exclude_dmso,
                            shuffle=shuffle)



    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)
    


    classes_, indexes = np.unique(dataset_test.meta[dataset_test.col_names[-1]], return_index=True)
    boolean = len(classes_) == 13 if not exclude_dmso else len(classes_) == 12

    if not boolean: 
       raise Warning("The number of unique drugs in the test set is not 13")


    print('initialized dataloaders')

    VAE = Semi_supervised_VAE(classes=classes, latent_dim=latent_dim,
                            input_dim=input_dim, channels=channels).to(device)
    
    print('initialized VAE')

    trained_encoder, trained_decoder, train_REs, train_KLs, train_ELBOs = VAE.train_VAE(
        dataloader=loader_train, epochs=epochs)
    
    print('trained VAE')

    test_REs, test_KLs, test_ELBOs = VAE.test_VAE(dataloader=loader_test)

    print('tested VAE')

    results_folder = 'semi_bbbc_little/'
    if not(os.path.exists(results_folder)):
        os.mkdir(results_folder)

    train_images_folder = results_folder +'train_images/'

    test_images_folder = results_folder + 'test_images/'

    if not(os.path.exists(train_images_folder)):
        os.mkdir(train_images_folder)

    if not(os.path.exists(test_images_folder)):
        os.mkdir(test_images_folder)


    plot_ELBO(train_REs, train_KLs, test_ELBOs, 'semi', results_folder)


    np.savez(results_folder + "train_ELBOs.npz", train_ELBOs=train_ELBOs)
    np.savez(results_folder + "train_REs.npz", train_REs=train_REs)
    np.savez(results_folder + "train_KLs.npz", train_KLs=train_KLs)
    np.savez(results_folder + "test_ELBOs.npz", test_ELBOs=test_ELBOs)
    np.savez(results_folder + "test_REs.npz", test_REs=test_REs)
    np.savez(results_folder + "test_KLs.npz", test_KLs=test_KLs)


    torch.save(trained_encoder, results_folder + "encoder.pt")
    torch.save(trained_decoder, results_folder + "decoder.pt")


    for i, image in enumerate(loader_train.dataset):
        if i == 10:
            break
        plot_1_reconstruction(image['image'],
                            vae = VAE,
                            name="Train " + str(image['id']), 
                            results_folder=train_images_folder, 
                            latent_dim=latent_dim, 
                            channels=channels, 
                            input_dim=input_dim)
    
    if boolean:
        for index in indexes:
            image = dataset_test[index]
            plot_1_reconstruction(image['image'],
                                vae = VAE,
                                name=image['moa_name'] + ' ' + str(image['id']), 
                                results_folder=test_images_folder, 
                                latent_dim=latent_dim, 
                                channels=channels, 
                                input_dim=input_dim)
            
    
    
    #from interpolation import interpolate_between_two_images, interpolate_between_three_images
    
    #interpolate_between_two_images(VAE, 452305, 475106, main_path, results_folder=results_folder)
    #interpolate_between_three_images(VAE, 452305, 475106, 273028, main_path, results_folder=results_folder)










