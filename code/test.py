from model import MyModel
from SVHNdataset import SVHNFullDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ctc_decoder import ctc_decode
import matplotlib.pyplot as plt

# read dataset
transformer = transforms.Compose([
    transforms.Resize((32, 128)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
test_data = SVHNFullDataset("../data/test", transformer)
test_loader = DataLoader(test_data, batch_size=512, shuffle = False)

model = MyModel()
model.load_state_dict(torch.load('../result/model.pth'))
dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("GPU")
else:
    print("CPU")
model = model.to(dev)
model.eval()

total = 0
correct = 0
wrong = 0
idx = 0
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
            # if correct <= 10:
            #     originImage = test_data.getOriginImage(idx)
            #     originImage.save("../result/correct_%d_pred_%s_real_%s.jpg" % (correct, "".join([str(s) for s in pred_label]), "".join([str(s) for s in real_label])))
        else:
            wrong += 1
            # if wrong <= 10:
            #     originImage = test_data.getOriginImage(idx)
            #     originImage.save("../result/wrong_%d_pred_%s_real_%s.jpg" % (wrong, "".join([str(s) for s in pred_label]), "".join([str(s) for s in real_label])))
        idx += 1
                


print("test acc: %.4f %%"%(100*correct/total))