import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# dataloader
from torch.utils.data import DataLoader
import numpy as np
from dataloader import BBBC
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for batch_idx, batch in tqdm(enumerate(train_loader)):
        data = batch['image'].to(device)
        target = batch['moa'].to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    save_pred = np.zeros((len(test_loader.dataset)))
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            data = batch['image'].to(device)
            target = batch['moa'].to(device).long()
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            save_pred[i*batch_size: (i+1)*batch_size] = pred.detach().cpu().numpy()
            correct += (pred == target).sum().item()
    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset), save_pred




train_size = 100_000
test_size = 30_000
batch_size = 100

# latent_dim = 10
#epochs, batch_size, train_size, test_size = 2, 10, 10, 10

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

from dataloader import BBBC

main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"
## main_path = "/Users/nikolaj/Fagprojekt/Data/"



exclude_dmso = True
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




model = classifier(classes=13, input_dim=68, channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()


train_loss = train(model, loader_train, optimizer, criterion, device)

print('train_loss:', train_loss)

test_loss, accuracy, save_pred = test(model, loader_test, criterion, device)

print('test_loss:', test_loss)
print('accuracy:', accuracy)

# np.save("saveª_pred_SC.npy", save_pred)

torch.save(model, "model_SC.pt")
