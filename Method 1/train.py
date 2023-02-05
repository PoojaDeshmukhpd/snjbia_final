import json
from snjbia_intents import tokenize,stem,bag_of_words
import numpy as np
import torch
import torch.nn as nn
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import numpy as np
import speech_recognition as sr
import pyttsx3
import subprocess as s
import datetime
from datetime import time
import os
import sqlite3
import nltk
import string
# nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


all_words=[]
tags=[]
xy=[]
with open('intents.json', "r") as f:
    intents = json.load(f)

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    w = intent['pattern']
    all_words.append(w)
    xy.append((w, tag))
print(all_words)
print(tags)
print(xy)
placementInfo = "in 2021 22 placement count was 38 candidates from computer department, in 2020 21 placement count was 48 candidates from computer department "
collegeInfo = "The Jain Gurukul campus has various faculties, out of which the SNJB's Late Sau Kantabai Bhavarlalji Jain College of Engineering, which is approved by the All India Council of Technical Education (AICTE)"

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 170)
engine.setProperty('volume', 0.2)

engine.setProperty('voice', 'english+f5')

def talk(text):
    engine.say(text)
    engine.runAndWait()


def take_command():
    try:
        with sr.Microphone() as source:
            listener.pause_threshold = 1
            print("listening...")
            voice = listener.listen(source, timeout=3, phrase_time_limit=5)
            r = sr.Recognizer()
            r.energy_threshold = 20000
            r.adjust_for_ambient_noise(source, 1.2)
            command = listener.recognize_google(voice, language='en-in')
            command = command.lower()
            if 'snjbya' in command:
                command = command.replace('snjbya', '')

    except:
        print("error")
        print("Unable to Recognize your voice.")
        return "None"

    return command


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        talk("Good Morning!")

    elif hour >= 12 and hour < 18:
        talk("Good Afternoon!")

    else:
        talk("Good Evening!")

    assname = ("SNJBEEA")
    talk("I am your Voice Assistant SNJBEEA")


clear = lambda: os.system('cls')
con = sqlite3.connect("madata.db")
cur = con.cursor()
clear()
wishMe()

while True:
        command = take_command().lower()
        # print(command)

        ntokens = word_tokenize(command)
        stop_words = set(stopwords.words('english'))
        print(stop_words)
        filter_sentence = [w for w in ntokens if not w.lower() in stop_words]
        filter_sentence = []
        for w in ntokens:
            if w not in stop_words:
                filter_sentence.append(w)

        print(ntokens, "\n")
        print(filter_sentence, "\n")
        punc = string.punctuation
        # print(punc)
        txt_pun = str(" ".join([c for c in filter_sentence if c not in string.punctuation]))
        print(txt_pun)







# ignore_words=['?','!', '.', ',']
# all_words=[stem(w) for w in all_words if w not in ignore_words]
# all_words=sorted(set(all_words))
# tage=sorted(set(tags))
# print(tags)
# print(all_words)
# print(xy)
#
# x_train=[]
# y_train=[]
# for (pattern_setence,tag) in xy:
#     bag=bag_of_words(pattern_setence,all_words)
#     x_train.append(bag)
#     label=tags.index(tag)
#     y_train.append(label) #crossEntrophyLoss
#
# x_train=np.array(x_train)
# y_train=np.array(y_train)
#
# class ChatDataset(Dataset):
#     def __init__(self):
#         self.n_samples=len(x_train)
#         self.x_data=x_train
#         self.y_data=y_train
#
#     # dataset[idx]
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]
#     def __len__(self):
#         return self.n_samples
#
# #  Hyperparamater
# batch_size=8
# hidden_size=8
# output_size=len(tage)
# input_size=len(x_train[0])
# learning_rate=0.001
# num_epochs=1000
#
# dataset=ChatDataset()
# train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2)
#
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model=NeuralNet(input_size, hidden_size, output_size).to(device)
#
# # loss and optimizer
# criterion=nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# # optimizer= torch.optim.Adam(model.paramaters(),lr=learning_rate)
# for epoch in range(num_epochs):
#     for (words,labels) in train_loader:
#         words=words.to(device)
#         labels(labels.to(device))
#
#         outputs=model(words)
#         loss=criterion(outputs,labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.stop()
#     if (epoch+1)%100 == 0:
#         print(f'epoch {epoch+1}/{num_epochs},loss={loss.item():.4f}')
#
# print(f'final loss, loss={loss.item():.4f}')
