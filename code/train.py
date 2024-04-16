from model import MyModel
from SVHNdataset import SVHNFullDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from ctc_decoder import ctc_decode
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

#parametor
max_epoch = 30
lr = 1e-3

# read dataset
transformer = transforms.Compose([
    transforms.Resize((32, 128)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
full_train_data = SVHNFullDataset("../data/train", transformer)
train_size = int(0.8 * len(full_train_data))
validation_size = len(full_train_data) - train_size
train_data, validation_data = random_split(full_train_data, [train_size, validation_size])
train_loader = DataLoader(train_data, batch_size=128, shuffle = True)
validation_loader = DataLoader(validation_data, batch_size=256, shuffle=True)
test_data = SVHNFullDataset("../data/test", transformer)
test_loader = DataLoader(test_data, batch_size=512, shuffle = False)


# load model
model = MyModel()
dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("GPU")
else:
    print("CPU")
model = model.to(dev)
criterion = nn.CTCLoss(blank=10, reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

# 

#train
train_loss = []
train_acc = []
val_acc = []
model.train()
for epoch in range(max_epoch):
    total_loss = 0
    total = 0
    correct = 0

    # train
    model.train()
    for batch_idx, (images, labels, label_lens) in enumerate(train_loader):
        optimizer.zero_grad()
        real_label = "".join(labels)
        labels = torch.tensor([int(c) for s in labels for c in s])
        images, labels, label_lens = images.to(dev), labels.to(dev), label_lens.to(dev)

        results = model(images)
        # 16*bs*num_classes
        results = nn.functional.log_softmax(results, dim=2)

        result_lens = torch.full(size=(results.size(1),), fill_value=results.size(0), dtype=torch.long)
        # print(results, labels, result_lens, label_lens)
        
        loss = criterion(results, labels, result_lens, label_lens)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += images.size(0)

        #decode for accuracy
        pred_labels = ctc_decode(results, method="greedy", beam_size=10, blank=10)
        real_labels = labels.cpu().numpy().tolist()
        real_label_len = label_lens.cpu().numpy().tolist()
        label_cnt = 0
        for pred_label, length in zip(pred_labels, real_label_len):
            real_label = real_labels[label_cnt: label_cnt+length]
            label_cnt += length
            # print(pred_labels, real_label)
            if pred_label == real_label:
                correct += 1
        if batch_idx%100 == 0:
            print("epoch: %d, batch: %d/%d, loss: %.4f"%(epoch, batch_idx, len(train_loader), loss.item()/images.size(0)))
    scheduler.step()

    # val 
    model.eval()
    validation_loss = 0
    validation_correct = 0
    validation_total = 0
    with torch.no_grad():
        for batch_idx, (images, labels, label_lens) in enumerate(validation_loader):
            real_label = "".join(labels)
            labels = torch.tensor([int(c) for s in labels for c in s])
            images, labels, label_lens = images.to(dev), labels.to(dev), label_lens.to(dev)

            results = model(images)
            # 8*bs*num_classes
            results = nn.functional.log_softmax(results, dim=2)

            result_lens = torch.full(size=(results.size(1),), fill_value=results.size(0), dtype=torch.long)
            loss = criterion(results, labels, result_lens, label_lens)

            validation_loss += loss.item()
            validation_total += images.size(0)

            #decode for accuracy
            pred_labels = ctc_decode(results, method="greedy", beam_size=10, blank=10)
            real_labels = labels.cpu().numpy().tolist()
            real_label_len = label_lens.cpu().numpy().tolist()
            label_cnt = 0
            for pred_label, length in zip(pred_labels, real_label_len):
                real_label = real_labels[label_cnt: label_cnt+length]
                label_cnt += length
                # print(pred_labels, real_label)
                if pred_label == real_label:
                    validation_correct += 1



    print("[epoch %d]loss: %.4f, train acc: %.4f %%, validation acc: %.4f %%"%(epoch, total_loss/total, 100*correct/total, 100*validation_correct/validation_total))
    train_loss.append(total_loss/total)
    train_acc.append(correct/total)
    val_acc.append(validation_correct/validation_total)

# test
total = 0
correct = 0
model.eval()
for batch_idx, (images, labels, label_lens) in enumerate(test_loader):
    real_label = "".join(labels)
    labels = torch.tensor([int(c) for s in labels for c in s])
    images, labels, label_lens = images.to(dev), labels.to(dev), label_lens.to(dev)

    results = model(images)
    # 16*bs*num_classes
    results = nn.functional.log_softmax(results, dim=2)

    result_lens = torch.full(size=(results.size(1),), fill_value=results.size(0), dtype=torch.long)

    total += images.size(0)

    #decode for accuracy
    pred_labels = ctc_decode(results, method="greedy", beam_size=10, blank=10)
    real_labels = labels.cpu().numpy().tolist()
    real_label_len = label_lens.cpu().numpy().tolist()
    label_cnt = 0
    for pred_label, length in zip(pred_labels, real_label_len):
        real_label = real_labels[label_cnt: label_cnt+length]
        label_cnt += length
        # print(pred_labels, real_label)
        if pred_label == real_label:
            correct += 1
print("test acc: %.4f %%"%(100*correct/total))
        
torch.save(model.state_dict(), "../result/model.pth")

# print(train_loss, train_acc, val_acc)

#plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_acc, label="train")
plt.plot(val_acc, label="validation")
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
