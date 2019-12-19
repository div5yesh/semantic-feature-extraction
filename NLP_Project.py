# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().system('pip install pytorch_pretrained_bert')


# %%
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from read_data import Embeddings
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_excel('./CONCS_FEATS_concstats_brm.xlsx')
embeds = Embeddings()

# %%
concepts = list(df[["Concept","Feature"]].groupby("Concept").groups.keys())
features = list(df[["Concept","Feature"]].groupby("Concept")["Feature"].apply(list))

#%%
x = []
for i in features:
    x += i
    
x = np.array(x)
val, co = np.unique(x, return_counts=True)
val_ = val[np.argsort(co)[-50:]]
co_ = co[np.argsort(co)[-50:]]

# %%
plt.figure(figsize=(10, 5))
plt.title("Top 50 hypernym distribution")
plt.xlabel("Features")
plt.ylabel("Frequency")
plt.tight_layout()
plt.xticks(rotation='vertical')
plt.bar(np.flip(val_), np.flip(co_))

#%%
dict_features = []
for feat in features:
    dft = feat
    # dft = list(filter(lambda x: x in val_, feat))
    dict_features += [dict.fromkeys(dft, True)]

# %%
dv = DictVectorizer(sparse=False)
Y = dv.fit_transform(dict_features)
Y = torch.tensor(Y).double()
print(max(Y.sum(axis=1)))

# %%
embeddings = []
for con in concepts:
    em = embeds.getEmbeddings(con.split("_")[0]).tolist()
    embeddings.append(em)

embeddings = torch.tensor(embeddings)

# %%
embeddings = embeddings.double()
trainX = embeddings[:449]
devX = embeddings[450:499]
testX = embeddings[500:]

trainY = Y[:449]
devY = Y[450:499]
testY = Y[500:]

# %%
labels = 2526
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12 * 768, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, labels)

    def forward(self, x):
        x = x.view(-1, 12 * 768)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.get_device_name(device))
net = Net()
net.to(device)
net.double()
# torch.autograd.set_detect_anomaly(True)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.2)
# pos_weight = torch.ones(Y.shape[1]) * 5
criterion = nn.BCELoss()
# criterion.to(device)

epochs = 100
batch_size = 4
run_loss = []
batches = int(len(trainX)/batch_size)
for epoch in range(epochs):
    for batch_idx in range(batches):
        start = batch_size * batch_idx
        end = batch_size * (batch_idx + 1)
        data, target = Variable(trainX[start:end]), Variable(trainY[start:end])

        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out.reshape((-1,labels)), target.reshape((-1,labels)))
        
        # if epoch % 10 == 0:
            # print('Before backward pass: \n', net[0].weight)
            # s = torch.sum(net.fc3.weight.data)
        loss.backward()
        optimizer.step()
        # if (epoch+1) % 5 == 0:
            # print('After backward pass: \n', net[0].weight)
            # s = torch.sum(net.fc3.weight.data)

        
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.data))
    run_loss.append(loss.item())

# %%
plt.title("Loss curve - training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(list(range(0,epochs)), run_loss)

# %%
test_loss = 0
correct = 0
precision = 0
threshold = 0.99
label_threshold = 0.1
recall = 0

with torch.no_grad():
    data, target = Variable(devX), Variable(devY)
    if torch.cuda.is_available():
        data, target = data.to(device), target.to(device)

    net_out = net(data)
    test_loss += criterion(net_out, target).data
    out = net_out.data
    x  = out.clone()
    # print(out.min(axis=1))
    out[out > label_threshold] = 1
    out[out <= label_threshold] = 0

for idx, act in enumerate(target):
    pred = out[idx]
    
    match = sum((pred == act) * 1)

    if match.tolist()/2526 >= threshold:
        correct += 1

    idx_pred = np.where(np.array(out[idx].tolist()) == 1)
    idx_act = np.where(np.array(target[idx].tolist()) == 1)

    tn_fp = out[idx].sum()
    tn_tp = target[idx].sum()

    precision += len(np.intersect1d(idx_act, idx_pred))/tn_fp
    recall += len(np.intersect1d(idx_act, idx_pred))/tn_tp


print("Accuracy:", correct/len(devX))
print("Precision:", precision/len(devX))
print("Recall:", recall/len(devX))

# %%
out = net(devX.cuda())

#%%
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
idx = 0
for row in ax:
    for col in row:
        col.scatter(out[idx].tolist(),range(2526))
        col.plot([0.15,0.15],[0,2526], c="r")
        idx += 1
plt.tight_layout()
plt.show()

# %%
