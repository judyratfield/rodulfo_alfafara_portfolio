#RUN this in the command prompt to execute this program properly
#python classify.py --input-file (tv description filename.txt) --output-json-file (genre predictions filename.json) --encoding UTF-8 --explanation-output-dir (./subdirectory name)

#In this classifier program, we will be creating an inference system that will use
#the model we created previously to predict television show genres

#First of all, we are going to preprocess the text to make the prediction more
#accurate

#defined preprocess function
def preprocess(tv_desc):#preprocess function follows 4 steps(clean html tags, re,tokenize,stopwords handling)
    res1 = clean_html_tags(tv_desc)
    res2 = step1(res1)
    res3 = step2(res2)
    res4 = step3(res3)

    processed_text = res4#step3(res2)
    return processed_text

#defined function that cleans html tags
def clean_html_tags(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    cleaned_text = soup.get_text()
    return cleaned_text

def step1(res1):# replaces characters that are not words with spaces.
    import re#imported re function
    res2 = re.sub("[^\w]"," ",tv_desc)
    return res2

def step2(res2):#splits the text into words as tokens
    from nltk.tokenize import word_tokenize #imported nltk word_tokenize
    res3 = word_tokenize(res2.lower())#changes words into lowercase
    return res3

def step3(res3):#filters out stopwords which are words that do not provide much information
    #for this part, we will not be using the nltk stopwords library since it removes a lot of important words
    
    #These are the only words we will be considering as stop words as these are the words with the most number of occurances that do not provide any information. Moreover, when we analyzed the explanation of the model, some of these stopwords kept on ocurring. We did use the stopwords form nltk since it contains some important words like "not" among other things.
    stop_words = set(['the', ',', '<', '>', 'and', 'of', 'a', '.', 'p', 
                      '/p', 'is', "'s", 'with', 'as', 'that', 'who', 'b', 
                      '/b', 'by', 'they', 'he', 'an', 'series', 'are', 'it', 
                      'will', 'at', 'this', 'has', "''", '``', 'was', 
                      'be', 'have', 'when', 'about', ')', '(', ':', '?',
                      'which', 'them', 'what', 'i', 'I', 'we', '-', 'you',
                      'would', 'us', 'every', 'but', 't', 'But', 'because',
                      'day', 'to', 's', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                      'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 
                      'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'de' ])#we also removed single occurences of the alphabet, as these letters alone hold no meaning, these need to form a word at the very least to have meaning.

    #used list comprehension for faster filtering
    res4 = [word for word in res3 if word not in stop_words]#words that are not stopwords are retained
    return res4

#After processing the text, we will be using the trained model to predict the
#genres of the processed tv show description

#defined the function for the inference system
def predict_genre(processed_text):

    #imported necessary packages to load the model
    from keras.models import load_model
    import pickle
    import pandas as pd
    
    #loaded the model
    inference_model = load_model('trained_model')
    
    #loaded the dictionary previously saved
    inference_category_lookup = pickle.loads(open('genres.pkl', 'rb').read())#rb for read binary
    
    # Set the option to display all rows and columns. This setting was used to be able to view all 27 results in the console to check that the result is as expected.
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    #executed the inference system to predict the genre of the provided tv show description
    prediction = inference_model.predict([processed_text])
    
    #saved the prediction results into a data frame
    prediction_df = pd.DataFrame(data = prediction, columns = inference_category_lookup)
    print(prediction_df)
    return prediction_df
    
#After making predictions, we also have to enable the model to provide
#an explanation as to why it made such predictions.

#defined function that will explain the prediction
def explain_prediction(processed_text):
    #imported necessary packages
    import os
    import lime.lime_text
    from keras.models import load_model
    
    #loaded the model
    model = load_model('trained_model')
    
    #Set lime text explainer to text_explainer variable
    #Set class names to the genres
    text_explainer = lime.lime_text.LimeTextExplainer(class_names = ['Action','Adult','Adventure','Anime',
                                                                 'Children','Comedy','Crime','DIY','Drama','Espionage',
                                                                 'Family','Fantasy','Food','History','Horror','Legal',
                                                                 'Medical','Music','Mystery','Nature','Romance','Science-Fiction', 
                                                                 'Sports','Supernatural','Thriller','Travel','War','Western'],
                                                  char_level = False,#set to False because we are operating on words instead of on individual characters
                                                  bow = True)#set to True because the position of the string is not important for our prediction
    
    if args.explanation_output_dir:
        #executed text explainer to explain the prediction for the passed processed text
        #set the labels to 0-27 which represents the 28 genres
        explanation = text_explainer.explain_instance(processed_text, model.predict, labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,
                                                                                               13,14,15,16,17,18,19,20,21,22,
                                                                                               23,24,25,26,27])
        
        #Creates the output directory if it doesn't exist
        os.makedirs(args.explanation_output_dir, exist_ok=True)
        
        #saved the explanation to an html file                                                                                       
        #explanation.save_to_file('explanation_output.html') 
        explanation.save_to_file(os.path.join(args.explanation_output_dir, 'explanation_output.html'))




#imported argparse to aid in consuming the passed file containing the tv description
import argparse

if __name__== "__main__":
    
    parser = argparse.ArgumentParser(description="Classify TV show genres based on description")
    parser.add_argument("--input-file", required=True, help="Path to the input file with TV show description")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for genres")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")
    parser.add_argument("--explanation-output-dir", help="Directory for explanation output")

    args = parser.parse_args()
    
    input_filename = args.input_file
    
    #read the passed tv description
    with open(input_filename, 'r') as f:
        tv_desc = f.read()#sets the read description to variable tv_desc
    
    #called the functions above on the read tv description
    processed_list = preprocess(tv_desc)#called preprocess function to preprocess the tv description
    processed_text = ' '.join(processed_list)#converted the preprocessed description from a list to a string so that the vectorizer can process the description
    predicted_genres = predict_genre(processed_text)#called the predict_genre prediction to execute the predictions for the processed text
    explanation_prediction = explain_prediction(processed_text)#called the explain_prediction function on the processed text to come up with an explanation on the predictions made
    
    #imported heapq library to aid in getting the highest prediction values from the Pandas DataFrame
    import heapq
    
    #got the 3 highest prediction values and saved the results into the top_genres variable
    threshold = 0.05#we applied a threshold of 0.05, any prediction below than the threshold value should not be included as a prediction as 0.05 is the widely used significance value in statistics
    top_genres = [genre for genre in predicted_genres.iloc[0].index if predicted_genres.iloc[0][genre] > threshold]
    top_genres = heapq.nlargest(3, top_genres, key=lambda genre: predicted_genres.iloc[0][genre])
    
    #imported pickle to be able to load the genres dictionary previously created using pickle
    import pickle
    
    #loaded the dictionary previously saved
    inference_category_lookup = pickle.loads(open('genres.pkl', 'rb').read())#rb for read binary
    
    #created an empty list that will later on carry the actual genre prediction names
    final_prediction = []
    #iterated through the 3 highest prediction values saved earlier
    for genre in top_genres:
        genre_prediction = inference_category_lookup[genre]#matched the genre prediction numerical key to its genre prediction name value
        final_prediction.append(genre_prediction)#appended the genre prediction name value to the final_prediction list
    
    #imported json because we want to output a json file that contains the predictions made
    import json

    #specified the output JSON file name that contains the genre predictions
    output_filename = args.output_json_file#"genre_prediction.json"
    
    #saved the genre predictions into to a JSON file
    with open(output_filename, 'w') as json_file:
        json.dump(final_prediction, json_file)
    print(final_prediction)






