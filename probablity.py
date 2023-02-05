import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import time
import difflib
import numpy
import tflearn
import tensorflow
import json
import pickle
import random
import os
import speech_recognition as sr
import pyttsx3
# preprocessing

with open("intents.json") as file:
    data=json.load(file)

try:
    with ("data.pickle","rb") as f:
        words,label,training,output = pickle.load(f)
except:
    words = []
    label = []
    docs_patt = []
    docs_tag = []

# tokenization and stemming

    for intent in data["intents"]:
        for pattern in intent["pattern"]:
            wrds = nltk.word_tokenize(pattern)
            for item in wrds:
                words.extent(wrds)
                docs_patt.append(wrds)
                docs_tag.append(intent["tag"])
                if intent["tag"] not in label:
                    label.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))
    label = sorted(label)

    training = []
    output = []

    out_empty = [0 for _ in range(len(label))]


    # bag of words feature engineering

    for x, doc in enumerate(docs_patt):
        bag = []
        wrds = [stemmer.stem(w.lower())for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[label.index(docs_tag[x])] =1
        training.append(bag)
        output.append(output_row)
    training=numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle","wb") as f:
        pickle.dump((words,label,training,output),f)


# model building

from tensorflow.python.framework import ops
ops.reset_default_graph()
net = tflearn.input_data(shape = [None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")

net= tflearn.regrassion(net)
model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model=tflearn.DNN(net)
    history=model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
    model.save("model.tflearn")
#input preprocessing
def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w==se:
                bag[i]=1
    return numpy.array(bag)

def word_to_list(s):
    a=[]
    ns=""
    s=s+" "
    for i in range(len(s)):
        if s[i]==" ":
            a.append(ns)
            ns=""
        else:
            ns=ns+s[i]
    a=list(set(a))
    return a
def json_to_dictionary(data):
    dict=[]
    fil_dic=[]
    voc=[]
    for i in data["intents"]:
        for pattern in i["pattern"]:
            voc.append(pattern.lower())
    for i in voc:
        dict.append(words_to_list(i))
    for i in range(len(dict)):
        for word in dict[i]:
            fil_dic.append(word)
    return list(set(fil_dic))
chatbot_vocabulary = json_to_dictionary(data)

def word_checker(s):
    correct_string = ""
    for word in s.casefold().split():
        if word not in chatbot_vocabulary:
            suggestion = difflib.get_close_matches(word,chartbot_vocabulary)
            for x in suggestion:
                pass
            if len(suggestion)==0:
                pass
            else:
                correct_string = correct_string+" "+str(suggestion[0])
        else:
            correct_string = correct_string+" "+str(word)
    return correct_string

# speech recognition
r = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 170)
engine.setProperty('volume', 0.5)
engine.setProperty('voice', 'english+f5')


def bot_speaking(message):
    engine.say(message)
    engine.runAndWait()
    if engine._inLoop:
        engine.endLoop()

def get_input():
    with sr.Microphone() as source:
        bot_speaking("Hey mates Say Somthing")
        audio = r.listen(source,timeout=0)
        bot_speaking("Perfect, Thanks!!")
    try:
        msg = r.recognize_google(audio)
        print("text: "+msg);
        bot_speaking("you said "+ msg)
        return  msg
    except:
        bot_speaking("Sorry mate!! its not working")

def chat():
    print("Snjebia I am your college assistant and I am here to answer the query on snjb")
    while True:
        inp = input("You: ")
        if inp.lower()=="quit" or inp == None:
            break
        inp_x = word_checker(inp)

        results = model.predict([bag_of_words(inp_x,words)])[0]

        results_index = numpy.argmax(results)

        tag = label[results_index]

        if results[results_index]>=0.9:
            for tg in data["intents"]:
                if tg['tag']==tag:
                    responses = tg['responses']
                    ms = random.choice(responses)
                    print("Snjbia: "+ms)
                    bot_speaking(ms)

        else:
            print("Snjbia: Sorry")
            bot_speaking("Sorry")

chat()
