import speech_recognition as sr
import time
import os
import playsound
import pyttsx3
from gtts import gTTS

engine=pyttsx3.init()
#
# rate=engine.getProperty("rate")
# print(rate)

engine.setProperty("rate",120)
engine.setProperty("volume",1.5)
voices=engine.getProperty("voices")
engine.setProperty("voice",format(voices[1].id))
engine.say("hello in python")
engine.runAndWait()
def speak(text):
    tts=gTTS(text=text,lang="en")
    filename="./voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)
speak("hello tim")