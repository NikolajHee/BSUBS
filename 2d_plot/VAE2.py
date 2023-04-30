import torch
from VAE import log_Categorical, log_Normal, log_standard_Normal, encoder, decoder, VAE, generate_image
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# gridsearch for latent_dim

save_folder_path = "" # 4, 8, 16, 32, 64, 128

latent_dim = 2
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

# plotting the 9 test images
# fig, axs = plt.subplots(3, 3)
# for i in range(3):
#     for j in range(3):
#         axs[i, j].imshow(X_test[i*3+j].reshape((input_dim, input_dim)), cmap='gray')
# plt.show()



VAE_ = VAE(X_train, pixel_range=pixel_range,
        latent_dim=latent_dim, input_dim=input_dim, channels=channels).to(device)
encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space, error_log = VAE_.train_VAE(
    X=X_train, epochs=epochs, batch_size=batch_size)

name = "latent_dim_" + str(latent_dim) + "_epochs_" + str(epochs) + "_batch_size_" + str(batch_size) + "_"



torch.save("encoder.pt")
torch.save("decoder.pt")
