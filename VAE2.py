import torch
from VAE import log_Categorical, log_Normal, log_standard_Normal, encoder, decoder, VAE, generate_image
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# gridsearch for latent_dim

save_folder_path = "gridsearch_results/"

latent_dims = [200, 10, 20, 50, 100]
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


for latent_dim in latent_dims:
    print("latent_dim: ", latent_dim)

    VAE_ = VAE(X_train, pixel_range=pixel_range,
            latent_dim=latent_dim, input_dim=input_dim, channels=channels).to(device)
    encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE_.train_VAE(
        X=X_train, epochs=epochs, batch_size=batch_size)

    name = "latent_dim_" + str(latent_dim) + "_epochs_" + str(epochs) + "_batch_size_" + str(batch_size) + "_"

    np.savez(save_folder_path + name + "latent_space.npz", latent_space=latent_space.detach().cpu().numpy())
    np.savez(save_folder_path + name + "reconstruction_errors.npz", reconstruction_errors=np.array(reconstruction_errors))
    np.savez(save_folder_path + name + "regularizers.npz", regularizers=np.array(regularizers))


    generated_images = []
    for i in range(9):
        image = generate_image(X_test[i], encoder_VAE, decoder=decoder_VAE,
                            latent_dim=latent_dim, channels=channels, input_dim=input_dim)
        generated_images.append(image)

    np.savez(save_folder_path + name + "generated_images.npz", generated_images=np.array(generated_images))


torch.save(save_folder_path + name + encoder_VAE, "encoder.pt")
torch.save(save_folder_path + name + decoder_VAE, "decoder.pt")
