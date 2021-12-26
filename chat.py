import random
import json
import torch
import numpy as np
from model import NeuralNet
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


with open("intents.json","r") as f:
	intents = json.load(f)	

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "BOT"

while True:
	sentence = input("You: ")
	if(sentence == "quit"):
		break

	sentence = tokenize(sentence)
	x = bag_of_words(sentence,all_words)
	x = x.reshape(1,x.shape[0])
	x = torch.from_numpy(x).to(device)

	output = model(x.float())

	_, predicted = torch.max(output, dim = 1)

	tag = tags[predicted.item()]

	probs = torch.softmax(output,dim = 1)
	prob = probs[0][predicted.item()]

	if prob.item() > 0.75:
		for intent in intents['intents']:
			if tag == intent["tag"]:
				print(f"{bot_name}: {random.choice(intent['responses'])}")			
	else:	
				print(f"{bot_name}: Sorry! I do not understand.\nIf you wish to talk to our customer executive please click here and we will contact you shortly.\nThank You!")
		
