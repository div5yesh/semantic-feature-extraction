#%%
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.feature_extraction import DictVectorizer
from read_data import Embeddings
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch

#%%
df = pd.read_excel('/home/flipper/divyesh/semantic-feature-extraction/CONCS_FEATS_concstats_brm.xlsx')
embeds = Embeddings()

# %%
concepts = list(df[["Concept","Feature"]].groupby("Concept").groups.keys())
features = list(df[["Concept","Feature"]].groupby("Concept")["Feature"].apply(list))

dict_features = []
for feat in features:
    dict_features += [dict.fromkeys(feat, True)]

# %%
dv = DictVectorizer(sparse=False)
Y = dv.fit_transform(dict_features)
Y = torch.tensor(Y, requires_grad=True).double()

# %%
embeddings = []
for con in concepts:
    em = embeds.getEmbeddings(con.split("_")[0]).tolist()
    embeddings.append(em)

embeddings = torch.tensor(embeddings, requires_grad=True)

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12 * 768, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 2526)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x)

#%%
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
net = Net()
net.to(device)
net.double()
torch.autograd.set_detect_anomaly(True)
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.2)
criterion = nn.BCELoss()

#%%
embeddings = embeddings.double()
trainX = embeddings[:449]
devX = embeddings[450:499]
testX = embeddings[500:]

trainY = Y[:449]
devY = Y[450:499]
testY = Y[500:]

# %%
epochs = 10
for epoch in range(epochs):
    for batch_idx, data in enumerate(trainX):
        data, target = Variable(data), Variable(trainY[batch_idx])
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        # print(data.shape, target.shape)
        # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
        # data = data.view(-1, 12*768)
        optimizer.zero_grad()
        net_out = net(data)        
        # net_out[net_out <= 0] = 0
        # net_out[net_out > 0] = 1        
        loss = criterion(net_out.reshape((1,2526)), target.reshape((1,2526)))
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx , len(trainX),
                           100. * batch_idx / len(trainX), loss.data))


# %%
test_loss = 0
correct = 0
for idx, data in enumerate(devX):
    data, target = Variable(data, volatile=True), Variable(devY[idx])
    if torch.cuda.is_available():
        data, target = data.to(device), target.to(device)
    # data = data.view(-1, 28 * 28)
    net_out = net(data)
    # sum up batch loss
    test_loss += criterion(net_out, target).data
    # print(test_loss)
    pred = net_out.data # get the index of the max log-probability
    # pred[pred <= 0] = 0
    # pred[pred > 0] = 1
    
    correct += pred.eq(target).sum()
    # print(correct)

test_loss /= len(devX)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(devX),
        100. * correct / len(devX)))

# %%
