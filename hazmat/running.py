#import speech_recognition as sr
from time import ctime
import time
import os
import sys
#from gtts import gTTS
import subprocess



global running
running = False

global haz
global gen
global zbar

f = open("output.txt", "r")
for i in f:
    haz = i.split()[0]
    gen = i.split()[1]
    zbar = i.split()[2]
f.close()

def write():
    global gen
    global haz
    f = open("output.txt", "w")
    x = str(haz) + " " + str(gen) + " " + str(zbar)
    f.write(x)
    f.close()


def recordAudio():
    data = input("command: ")

    return data

# initialization

while 1:
    data = recordAudio()
    
    if "start detection" in data.lower():
        if running == False:
            print("starting detection")

            p = subprocess.Popen([sys.executable, os.path.join(os.getcwd(), "detectvideo(all).py")])
        running = True
        
    elif "end detection" in data.lower() or "and detection" in data.lower():
        print("ending detection")

        p.kill()
        running = False

    #hazmat general

    if "enable hazmat" in data.lower():
        haz = "True"
        write()
    elif "disable hazmat" in data.lower():
        haz = "False"
        write()
    if "enable general" in data.lower():
        gen = "True"
        write()
    elif "disable general" in data.lower():
        gen = "False"
        write()
    if "enable qr" in data.lower():
        zbar = "True"
        write()
    elif "disable qr" in data.lower():
        zbar = "False"
        write()
    if "help" in data.lower():
        print("hazmat, general, qr")
    if data.lower() == "quit":
        quit()
    
