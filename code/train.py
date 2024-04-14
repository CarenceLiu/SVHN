from model import MyModel
from SVHNdataset import SVHNFullDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ctc_decoder import ctc_decode
import matplotlib.pyplot as plt

#parametor
max_epoch = 15
lr = 1e-2

# read dataset
transformer = transforms.Compose([
    transforms.Resize((32, 128)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
train_data = SVHNFullDataset("../data/train", "train", transformer)
test_data = SVHNFullDataset("../data/test", "test", transformer)
train_loader = DataLoader(train_data, batch_size=64, shuffle = True)
test_loader = DataLoader(test_data, batch_size=64, shuffle = False)

# load model
model = MyModel()
dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("GPU")
else:
    print("CPU")
model = model.to(dev)
criterion = nn.CTCLoss(blank=10, reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 

#train
train_loss = []
train_acc = []
model.train()
for epoch in range(max_epoch):
    total_loss = 0
    total = 0
    correct = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images, labels = images.to(dev), labels.to(dev)

        results = model(images)
        results = nn.functional.log_softmax(results, dim=2)

        result_len = torch.full(size=(results.size(1),), fill_value=results.size(0), dtype=torch.long)
        label_len = torch.full(size=(labels.size(0),), fill_value=labels.size(1), dtype=torch.long)
        
        loss = criterion(results, labels, result_len, label_len)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += images.size(0)

        #decode for accuracy
        pred_labels = ctc_decode(results, method="beam_search", beam_size=1)
        real_labels = labels.cpu().numpy().tolist()
        real_label_len = label_len.cpu().numpy().tolist()
        label_cnt = 0
        for pred_label, length in zip(pred_labels, real_label_len):
            real_label = real_labels[label_cnt: label_cnt+length]
            label_cnt += length
            if pred_label == real_label:
                correct += 1

        if batch_idx%50 == 0:
            print("epoch: %d, batch: %d/%d, loss: %.4f %%"%(epoch, batch_idx, len(train_loader), loss.item()/images.size(0)))
    print("[epoch %d]loss: %.4f %%, acc: %.4f %%"%(100*total_loss/total, 100*correct/total))
    train_loss.append(100*total_loss/total)
    train_acc.append(100*correct/total)
    
        
torch.save(model.state_dict(), "../result/model.pth")

#plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_acc)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()
