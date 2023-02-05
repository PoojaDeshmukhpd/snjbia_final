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
# nltk.download('punkt')

from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')

placementInfo = "in 2021 22 placement count was 38 candidates from computer department, in 2020 21 placement count was 48 candidates from computer department "
collegeInfo = "The Jain Gurukul campus has various faculties, out of which the SNJB's Late Sau Kantabai Bhavarlalji Jain College of Engineering, which is approved by the All India Council of Technical Education (AICTE)"
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 170)
engine.setProperty('volume', 0.5)
engine.setProperty('voice', 'english+f5')

def username():
    talk("What should i call you")
    uname = take_command()
    talk("Welcome")
    talk(uname)
    talk("How can i Help you")

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
            r.energy_threshold = 300
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


text = ""

if __name__ == '__main__':
    clear = lambda: os.system('cls')
    con = sqlite3.connect("madata.db")
    cur = con.cursor()
    clear()
    wishMe()
    username()
    

    while True:
        command = take_command().lower()
        # print(command)

        ntokens = word_tokenize(command)
        stop_words = set(stopwords.words('english'))
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
        query = "select lower(tag),lower(pattern),lower(response),lower(link) from que where pattern=?;"
        cur.execute(query, (txt_pun,))
        ans = cur.fetchall()
        print(ans)

        if (ans):
            url = str(ans[0][-1])
            # openfile = s.Popen(["midori", "-a", url])
            q = ans[0]
            print(q)
            talk(ans[0][2])
            # openfile.terminate()
        else:
            print("sorry i did't get that try again")
            talk("sorry i did't get that try again")

        # if 'placement count' in command:
        #     s.Popen(["midori", "-a", "http://www.snjb.org/engineering/images/department/Update_Placement_Summery_Computer_Per.png"])
        #     print(placementInfo)
        #     talk(placementInfo)
        # elif command=="none":
        #     talk("hey i didn't heard properly come again?")

        # elif 'how are you' in command:
        #     talk("I am fine, Thank you")
        #     talk("How are you, Sir")

        # elif 'departments' in command:
        #     talk("Computer Engineering, Civil Engineering, Electronics and Telecommunication, Mechanical Engineering, Artificial Intelligence and Data Science")

        # elif 'branches' in command:
        #     talk("Computer Engineering, Civil Engineering, Electronics and Telecommunication, Mechanical Engineering, Artificial Intelligence and Data Science")
        # elif 'how many branches' in command:
        #     s.Popen(["midori", "-a", "http://www.snjb.org/engineering/About/about_engineering#about"])

        #     talk(collegeInfo)

        # elif 'college information' in command:
        #     s.Popen(["midori", "-a", "http://www.snjb.org/engineering/About/about_engineering#about"])
        #     talk(collegeInfo)

        # elif 'DSE' in command:
        #     s.Popen(["midori", "-a", "http://www.snjb.org/engineering/Admission/fees_structure_dse"])

        # elif 'direct second year admission' in command:
        #     s.Popen(["midori", "-a", "http://www.snjb.org/engineering/Admission/fees_structure_dse"])

        # elif 'principal' in command:
        #     print("Dr.M.D.Kokate is Principal of SNJB's KBJ College of Engineering Chandwad")
        #     talk("Dr.M.D.Kokate is Principal of SNJB's KBJ College of Engineering Chandwad")
        #     s.Popen(["midori", "-a", "http://www.snjb.org/engineering/About/engineering_mba_principals_message"])

        # elif 'vice principal' in command:
        #     print("Sanghavi Sir is vice principle of SNJB's KBJ College of Engineering Chandwad")
        #     talk("Sanghavi Sir is vice principle of SNJB's KBJ College of Engineering Chandwad")

        # elif 'account section' in command:
        #     s.Popen(["midori", "-a", "http://www.snjb.org/engineering/images/dynamic_page/Account_Section.jpg"])

        # elif 'vision and mission' in command:
        #     talk("Vision is Transform young aspirant learners towards creativity and professionalism for societal growth through quality technical education.")
        #     talk("Mission is To share values, ideas, beliefs by encouraging faculties and students for welfare of society.")

        # elif 'college campus' in command:
        #     s.Popen(["midori", "-a", "https://youtu.be/sweoKSBXeNc"])

        # elif 'how many labs in computer department' in command:
        #     print("Total 12 Labs are available in computer department")
        #     talk("Total 12 Labs are available in computer department")
        #     s.Popen(["midori", "-a", "snjb.org/engineering/Computer_engineering/computer_engineering_laboratories"])

        # elif 'be computer engineering result' in command:
        #     print("Academic Year 2021-22  100%  2018-19 96.34%")
        #     talk("Academic Year 2021-22  100%  2018-19 96.34%")
        #     s.Popen(["midori", "-a", "http://www.snjb.org/engineering/Computer_engineering/computer_engineering_results#r2"])

        # elif 'achievements of computer department' in command:
        #     s.Popen(["midori", "-a", "http://www.snjb.org/engineering/About/about_engineering#about"])
        #     talk(collegeInfo)

        # else:
        #     talk("hey i didn't heard properly come again?")