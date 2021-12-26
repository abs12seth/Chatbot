import json
import numpy as np
import random
import nltk
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
#nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
	return nltk.word_tokenize(sentence)

def stem(word):
	return stemmer.stem(word.lower())

def bag_of_words(sentence,all_words):
	sentence = [stem(w) for w in sentence]
	bag = np.zeros(len(all_words))
	#print(sentence)
	#print(all_words)
	for index,w in enumerate(all_words):
		if w in sentence:
			#print(w)
			bag[index] = 1.0

	return bag		


with open("intents.json","r") as f:
	intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
	tag = intent['tag']
	tags.append(tag)
	for pattern in intent['patterns']:
		w = tokenize(pattern)
		all_words.extend(w)
		xy.append((w,tag))

ignore_words = ['?','!',',','.']

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []
for (patt, tag) in xy:
	bag = bag_of_words(patt,all_words)
	X_train.append(bag)

	label = tags.index(tag)
	Y_train.append(label)
#print(X_train)
#print(Y_train)
X_train = np.array(X_train)
Y_train = np.array(Y_train)	


class ChatDataset(Dataset):
	def __init__(self):
		self.n_samples = len(X_train)
		self.x_data = X_train
		self.y_data = Y_train

	def __getitem__(self,index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.n_samples

batch_size = 8
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
no_of_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset,batch_size = batch_size, shuffle = True, num_workers=0)		

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)	

for epoch in range(no_of_epochs):
	for (words, labels) in train_loader:
		words = words.to(device)
		labels = labels.to(dtype=torch.long).to(device)
		#print(words)
		outputs = model(words.float())
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if(epoch+1)%100 == 0:
		print(f'epoch {epoch+1}/{no_of_epochs}, loss={loss.item():.5f}')	

print(f'final loss, loss={loss.item():.5f}')

data = {
	"model_state": model.state_dict(),
	"input_size": input_size,
	"hidden_size": hidden_size,
	"output_size": output_size,
	"all_words": all_words,
	"tags": tags
}

FILE = "data.pth"
torch.save(data,FILE)