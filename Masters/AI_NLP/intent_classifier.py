#RUN this in the command prompt to execute this program properly
#python intent_classifier.py

#For this assignment, I am calling Optus mobile phone customer support, 
#demonstrating a GILIICAPRA exchange. Firstly, the program will automatically 
#play an audio that "Greets and Invites" the customer to say his/her intent. 
#Next, the program allows the customer to say his/her intent by recording the 
#customer's response or allowing the customer to play a recorded clip containing 
#his/her intent. Then, the program "Listens" to the customer's intent. Afterwards,
#the program will "Identify the Intent" of the customer by determining if the 
#intent of the customer falls under the following: Sales, Support, Complaints, 
#Account Management, Billing and Payments, and will "Confirm" identified intent 
#with the customer by playing another audio clip corresponding to the identified 
#intent. Next, the program will allow the customer to "Accept" by recording 
#another response or playing another audio clip that contains customer response. 
#Lastly, based on the customer response, the program will enact its "Placeholder 
#Reply and Act" by printing the identified intent of the customer and playing a 
#music queue.

#The user can try the following phrases:

#'I need to change my address details' - Account Management

#'I think I like a new phone' - Sales

#'The representative I talked to was very rude' - This will be considered 
#"very negative" and as a result, the program will print "OPERATOR" and will 
#abruptly end after playing an audio telling the user that the user will be 
#connected to an expert.

#'I would like to file a complaint' - Complaints

#'Hallelujah' - should be an unknown intent, and program will print CONFUSED and should abruptly end 
#the program after informing user that the user will be connected with an expert

#'I cannot connect to my mobile data' - Support

#'Please connect me to Billing and Payments' - Billing and Payments

#'Hindi kita maintindihan'(Tagalog for I don't understand you) - this will identify an intent of "unknown" and shall print an end result of "CONFUSED" and will just play an audio clip telling the customer that the customer will be connected with an expert.

#Marker can also try not recording anything, typing random characters when asked
#to supply file path for an audio to play, then just manually type all the transcripts.

#Marker can also try not recording anything, then just supply file paths to an audio file all throughout.

#Marker can also try saying No to all confirmations and see that the program will print "CONFUSED".

#got start time to know how long the program runs
import time
prog_start_time = time.time()

print('Program is loading. Please wait...')
print('Assuming user was able to install everything already.The whole program usually runs around 2.5 - 3.5 minutes depending on the user intent.')
print('Please also note that majority of the runtime is exhausted in identifying the user intent. Please bear with us...')

#imported necessary packages
from pydub import AudioSegment
import pygame
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from transformers import pipeline
import logging
import ollama
import json
import time
import sys

#defined necessary functions

#defined function that converts the audio recordings in aac file to wav file formats

def convert_aac_to_wav(input_file, output_file):
    #loads the AAC audio file
    audio = AudioSegment.from_file(input_file, format="aac")
    
    #exports the audio into WAV format
    audio.export(output_file, format="wav")

#defined function that plays an audio file

def play_audio(file_path):
    pygame.init()
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(file_path)#loads the audio file
        pygame.mixer.music.play()#plays the audio file
        while pygame.mixer.music.get_busy():#waits until the audio playback is finished before continuing with the rest of the program
            pygame.time.Clock().tick(10)
    except pygame.error as e:
        print("Error occurred while playing audio:", e) #prints an error message should the program fails to be executed
    pygame.quit()
    
    
#defined function that can play audio through the speakers (or headphones)

def play(input_file,output_filename):
    #called function that converts aac audio file to wav file
    convert_aac_to_wav(input_file, output_filename)
    #called function that plays the newly converted wav audio file
    play_audio(output_filename)
    
#created a function that saves an audio file as a wav file
##This will be used for the recording so that the recorded audio can be transcribed properly as a wav file
def save_as_wav(audio_data, filename, fs):
    # Scale audio data to 16-bit integer range (-32768 to 32767)
    scaled_audio = np.int16(audio_data * 32767)
    # Write audio data to WAV file
    wavfile.write(filename, fs, scaled_audio)

#defined function that records audio requiring a duration input

def record_audio(duration):
    #set sampling frequency
    fs = 44100
    print("Recording started...")
    # Record audio
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()  #waits until recording is finished
    print("Recording stopped.")
    
    #saved the recording as a wav file so that it will be transcriber-ready
    filename = "recorded_audio.wav"
    save_as_wav(audio_data, filename, fs)
    print(f"Audio saved as {filename}")
    
    return audio_data

#defined function that when called, plays the recorded audio
def record_play(audio_data):
    #set sampling frequency
    fs = 44100
    print("Playing recorded audio...")
    sd.play(audio_data, samplerate=fs)
    sd.wait()  #waits until playback is finished
    print("Playback finished.")
    
#Using pipeline, created a transcriber that when called, converts an audio file and transcribes it into text
transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")

#Configures logging to write logs to a file named transcript.log
logging.basicConfig(filename='transcript.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#defined a function that would utilize the transcriber
#in case the transcriber fails, the user will be asked to manually type the script
def transcribe(input_file):
    print("Transcribing. Please wait...")
    try:
        transcript = transcriber(input_file)#calls the transcriber
        logging.info("Transcript: %s", transcript)  #Logs the transcript
        print("Transcript:", transcript)  #Prints the transcript
    except Exception as e:#should transcriber fail, the messages below will be displayed and the user will be prompted to enter transcript manually
        logging.error("Error occurred: %s", e)  # Log the error
        print("Error occurred:", e)  # Print the error message
        print("Please type the transcript manually:")
        transcript = input()
        print("Manually typed transcript:", transcript)
        logging.info("Transcript: %s", transcript)  #Logs the transcript
        transcript = {'text': transcript}#put transcript back into dictionary format as the other functions require dictionary inputs

    return transcript

#pulled gemma:2b
ollama.pull('gemma:2b')

#defined function that identifies an input message's intent using the model gemma:2b
def identify_intent(user_message):
    start_time = time.time()
    print("Identifying intent of the message: " + user_message)
    print('Assuming user was able to install everything already.The whole program usually runs around 2.5 - 3.5 minutes depending on the user intent.')
    print('Please also note that majority of the runtime is exhausted in identifying the user intent. Please bear with us...')

    response = ollama.chat(model='gemma:2b', messages=[
      {
        'role': 'user',
        'content': """ Use the following examples to guide identifying the intent of the message.

    The intent of the message should be classified as 'Sales' if the message contains statements similar to the following:
    "I'm interested in..."
    "I'd like to buy..."
    "What are my options for..."
    "Do you offer..."
    "How much does it cost?"


    The intent of the message should be classified as 'Support' if the message contains statements similar to the following:
    "I'm experiencing an issue with..."
    "My [product/service] isn't working properly."
    "I'm having trouble with..."
    "I can't [perform a specific action]."
    "There's an error message saying..."
    "I need help resolving..."


    The intent of the message should be classified as 'Complaints' if the message contains statements similar to the following:
    "I'm unhappy with..."
    "I'm dissatisfied with..."
    "I have a problem with..."
    "I'm frustrated because..."
    "This is unacceptable..."
    "I'm disappointed that..."

    The intent of the message should be classified as 'Account Management' if the message contains statements similar to the following:
    "I need help setting up my account."
    "I'm having trouble logging in."
    "I forgot my password/username."
    "I want to update my account information."
    "Can you help me change my email address?"
    "I need to update my address."



    The intent of the message should be classified as 'Billing and Payments' if the message contains statements similar to the following:
    "bill."
    "invoice/statement."
    "charges."
    "dispute a charge on my account."
    "payment."
    
    The intent of the message should be classified as 'idk' if the message contains statements similar to the following:
    "hindi kita maintindihan"
    "gsdajgasngfangsfansdgn"
    "buhbuhbuhbuh"
    unusual sounds
    if you don't understand


        """ 
    f'Now classify the intent of the following message,{user_message}'

      },
    ], format = 'json')
    end_time = time.time()
    running_time = end_time - start_time
    min_running_time = running_time/60
    #print("The running time lasts " + str(running_time) + " seconds or " + str(min_running_time) + " minutes")
    print(response['message']['content'])
    return response

#Used pipeline for sentiment analysis using RoBERTa-base model I found on huggingface because
## I found distilbert/distilbert-base-uncased-finetuned-sst-2-english to lean towards negative outcomes despite statements being neutral
from transformers import pipeline

classify_sentiment = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

#defined function that when called plays the audio appropriate to the identified intent
def confirm_intent(intent_value):
    if intent_value == 'Sales':
        #assigned file path to input_file variable
        input_file = 'recordings/Sales.aac'

        #assigned desired output filename including file extension(.wav)
        output_filename = 'Sales.wav'

        #called the play function
        play(input_file,output_filename)

    elif intent_value == 'Support':
        #assigned file path to input_file variable
        input_file = 'recordings/Support.aac'

        #assigned desired output filename including file extension(.wav)
        output_filename = 'Support.wav'

        #called the play function
        play(input_file,output_filename)

    elif intent_value == 'Complaints':
        #assigned file path to input_file variable
        input_file = 'recordings/Complaints.aac'

        #assigned desired output filename including file extension(.wav)
        output_filename = 'Complaints.wav'

        #called the play function
        play(input_file,output_filename)

    elif intent_value == 'Account Management':
        #assigned file path to input_file variable
        input_file = 'recordings/Account Management.aac'

        #assigned desired output filename including file extension(.wav)
        output_filename = 'Account_Management.wav'
        
        #called the play function
        play(input_file,output_filename)

    elif intent_value == 'Billing and Payments':
        #assigned file path to input_file variable
        input_file = 'recordings/Billing and Payments.aac'

        #assigned desired output filename including file extension(.wav)
        output_filename = 'Billing_Payments.wav'

        #called the play function
        play(input_file,output_filename)

    else:
        print('CONFUSED')
        #assigned file path to input_file variable
        input_file = 'recordings/dontknowhattodo.aac'

        #assigned desired output filename including file extension(.wav)
        output_filename = 'dontknowhattodo.wav'

        #called the play function
        play(input_file,output_filename)
        
        #abruptly ends the call after telling user that the user will be connected to an expert
        sys.exit()
    
#defined function that determines if the confirmation provided by the user is affirmative or negative
def verify_confirmation(user_message):
    start_time = time.time()
    print("Verifying confirmation of the user: " + user_message)

    response = ollama.chat(model='gemma:2b', messages=[
      {
        'role': 'user',
        'content': f""" 
    
    Classify the 'type' of the following confirmation as 'affirmative' or 'negative',{user_message}'

        """ 
      },
    ], format = 'json')
    end_time = time.time()
    running_time = end_time - start_time
    min_running_time = running_time/60
    #print("The running time lasts " + str(running_time) + " seconds or " + str(min_running_time) + " minutes")
    print(response['message']['content'])
    return response
    

#defined a function that when called checks if the confirmation is affirmative or not, and will perform corresponding actions depending on the result
def is_confirmation_affirmative(type_value):
    if type_value == 'Affirmative' or type_value == 'affirmative':
        print('Intent: ' + intent_value)
        
        #assigned file path to input_file variable
        input_file = 'recordings/finalqueue.aac'

        #assigned desired output filename including file extension(.wav)
        output_filename = 'finalqueue.wav'

        #called the play function
        play(input_file,output_filename)



    else:
        print('CONFUSED')
        #assigned file path to input_file variable
        input_file = 'recordings/dontknowhattodo.aac'

        #assigned desired output filename including file extension(.wav)
        output_filename = 'dontknowhattodo.wav'

        #called the play function
        play(input_file,output_filename)
    

#Now it's just a matter of stringing these things together into one big program

#played the greeting audio by calling the functions created above
#assigned file path to input_file variable
input_file = 'recordings/Greet_Invite.aac'
#assigned desired output filename including file extension(.wav)
output_filename = 'greet_invite.wav'
#called the play function
play(input_file,output_filename)

#Record the request
#set duration to record in 3 seconds as instructed
duration = 3
#recorded audio by calling the record_audio function above
recorded_data = record_audio(duration)
#called function record_play to play recorded audio afterwards
record_play(recorded_data)

#Transcribe it
#assigned file path of the recording used earlier to input_file variable
input_file = 'recorded_audio.wav'
#called transcribe function
transcript = transcribe(input_file)

#Your program has some option so that instead of recording audio from the sound device, it can read an audio file from disk.
if transcript['text'] == 'you' or transcript['text'] == ' you': #sets condition that detects if no recording has been made
    #play audio telling user to play an audio
    #assigned file path to input_file variable
    input_file = 'recordings/upload_audio.aac'
    #assigned desired output filename including file extension(.wav)
    output_filename = 'upload_audio.wav'
    #called the play function
    play(input_file,output_filename)
    #prints text tellling user to play an audio
    print('It seems that you have not provided any input. Please provide the file path of the audio containing what you need instead.')
    #prompts user to provide the path for the audio to be played
    audio_path = input()
    split = audio_path.split('.') #splits string by dots so we can isolate the file extension
    audio_format = split[-1]#selects the last split string to always get the file extension regardless of the number of dots in the filename
    #checks format if aac, the "play" function which converts the file first to wav format is used, if not, play_audio function is used
    if audio_format == 'aac':
        play(audio_path,'user_intent.wav')
        #Transcribes the audio file
        print('Transcribing audio file. Please wait...')
        #assigned file path to input_file variable
        input_file = audio_path
        #called transcribe function
        transcript = transcribe(input_file)
    else:
        play_audio(audio_path)
        #Transcribes the audio file
        print('Transcribing audio file. Please wait...')
        #assigned file path to input_file variable
        input_file = audio_path
        #called transcribe function
        transcript = transcribe(input_file)
  
#Check to see if it is very negative using your sentiment classifier        
try:
    sentiment = classify_sentiment(transcript['text'])
    print(sentiment)
    score = sentiment[0]['score'] #assigns score to score variable  
except:
    sentiment = 'negative'
    print(sentiment)
#extracts the label from the sentiment
try:
    sentiment_label = sentiment[0]['label']
    print(sentiment_label)
except:
    sentiment_label = sentiment
    print(sentiment_label)
if sentiment_label == 'negative':
    if score >= 0.85: #if this condition is met, then statement should be considered very negative
        print("OPERATOR")#prints operator as instructed
        
        #plays recording telling user that he will be transferred to an expert
        #assigned file path to input_file variable
        input_file = 'recordings/dontknowhattodo.aac'

        #assigned desired output filename including file extension(.wav)
        output_filename = 'dontknowhattodo.wav'

        #called the play function
        play(input_file,output_filename)
        sys.exit() #stops the program from executing and exits
else:
    print("User is calm.")
    

#Find out the intent
intent = identify_intent(transcript['text'])
#Accessed the value associated with the "content" key and parsed it as JSON so we can get the intent
content_json = json.loads(intent['message']['content'])
#Accessed the value of Intent Key to get the Intent value
try:
    intent_value = content_json["Intent"]
    print(intent_value)  
except:
    try:
        intent_value = content_json["intent"]
        print(intent_value)
    except:
        intent_value = "idk"
        print(intent_value)

#called function to confirm intent
confirm_intent(intent_value)

#Record Confirmation
#set duration to record in 3 seconds as instructed
duration = 3
#recorded audio by calling the record_audio function above
recorded_data = record_audio(duration)
#called function record_play to play recorded audio afterwards
record_play(recorded_data)
#transcribed the recording similar to what was done to the previous recording
#assigned file path of the recording used earlier to input_file variable
input_file = 'recorded_audio.wav'
#called transcribe function
transcript = transcribe(input_file)

#Your program has some option so that instead of recording audio from the sound device, it can read an audio file from disk.
#re-used the function above to allow the user to play an audio file instead of recording
#prints a text and plays an audio getting acceptance from the user

if transcript['text'] == 'you' or transcript['text'] == ' you': #sets condition that detects if no recording has been made
    
    #play audio telling user to play an audio
    #assigned file path to input_file variable
    input_file = 'recordings/get_acceptance.aac'

    #assigned desired output filename including file extension(.wav)
    output_filename = 'get_acceptance.wav'

    #called the play function
    play(input_file,output_filename)
    
    #prints text tellling user to play an audio
    print('I did not get a confirmation from you. Please provide the file path of the audio with your confirmation instead.')
    
    #prompts user to provide the path for the audio to be played
    audio_path = input()
    split = audio_path.split('.') #splits string by dots so we can isolate the file extension
    audio_format = split[-1]#selects the last split string to always get the file extension regardless of the number of dots in the filename
    #checks format if aac, the "play" function which converts the file first to wav format is used, if not, play_audio function is used
    if audio_format == 'aac':
        play(audio_path,'user_acceptance.wav')
        
        #Transcribes the audio file
        print('Transcribing audio file. Please wait...')
        #assigned file path to input_file variable
        input_file = audio_path
        #called transcribe function
        transcript = transcribe(input_file)
        
    else:
        play_audio(audio_path)
        
        #Transcribes the audio file
        print('Transcribing audio file. Please wait...')
        #assigned file path to input_file variable
        input_file = audio_path
        #called transcribe function
        transcript = transcribe(input_file)

#called function to check if confirmation is affirmative or negative
verification = verify_confirmation(transcript['text'])
#Accessed the value associated with the "content" key and parsed it as JSON so we can get the confirmation type
content_json = json.loads(verification['message']['content'])
#Accessed the value of the type Key to get the type value
try:
    type_value = content_json["Type"]
    print(type_value)
except:
    try:
        type_value = content_json["type"]
        print(type_value)
    except:
        type_value = "idk"
        print(type_value)

#called the function to check whether the confirmation is affirmative or not
is_confirmation_affirmative(type_value)

#Computed for the whole program running time
prog_end_time = time.time()
prog_running_time = prog_end_time - prog_start_time
prog_min_running_time = prog_running_time/60
print("The whole program runs for " + str(prog_running_time) + " seconds or " + str(prog_min_running_time) + " minutes")

        



    
    