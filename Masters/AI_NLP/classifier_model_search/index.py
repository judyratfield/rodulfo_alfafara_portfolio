#Run this in the command prompt to execute this program
#python index.py --raw-data sqlite-file.sqlite

#Since we already conducted an analysis previously on the schema of the tvmaze.sqlite file,
#and considering that the schema will be the same for the sqlite file to be used
#by the markers for evaluation, then we will be using the same details obtained
#from the schema analysis previously done from training the model.
#Moreover, since we only need description and not genre, then this time around,
#we need not merge the tvmaze_genre table and only focus on tvmaze table
#which contains description column

if __name__== "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index data from an SQLite database")
    parser.add_argument("--raw-data", required=True, help="Path to the SQLite database file")


    args = parser.parse_args()
    
    raw_data_filename = args.raw_data


    #imported sqlite 3
    import sqlite3
    
    #path  below should be replaced with the path of the sqlite file to be used
    database_file = raw_data_filename #in our case, this is passed by args.raw_data
    
    #established connections same as what had been done previously
    connection = sqlite3.connect(database_file)
    cursor = connection.cursor()
    
    #Defined query and selected the table tvmaze
    select_query = "SELECT * FROM tvmaze;"
    
    #Executed the query to fetch data from 'tvmaze'
    cursor.execute(select_query)
    
    #Fetched all rows from the result set and named it data
    data = cursor.fetchall()
    
    cursor.close()
    connection.close()
    
    #Similar to the approach from train.py, we will also be leveraging on the pandas dataframe
    
    #imported pandas
    import pandas as pd
    
    #Converted dataset into a pandas dataframe
    df = pd.DataFrame(data)
    
    # Defined a list of new column names for our dataframe
    new_column_names = ['tvmaze_id', 'update_timestamp', 'showname', 'first_airing',
                        'imdb', 'lang', 'description']
    
    # Assigned the new column names to the DataFrame
    df.columns = new_column_names
    
    # Set the option to display all rows and columns. This setting was used to be able to view all columns in the console to check that the result is as expected.
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    #Similar to train.py, we also remove records with missing data for description
    cleaned_df = df.copy().dropna(subset=['description'])
    
    #Since we removed records with missing values, we also reset the index of our dataset
    cleaned_df.reset_index(drop=True, inplace=True)
    
    #improted necessary packages
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    
    #defined preprocess function to help with the search execution later on
    def preprocess(tv_desc):#defined the steps to be performed in preprocessing
        res1 = step1(tv_desc)
        res2 = step2(res1)
        res3 = step3(res2)
        processed_text = step4(res3)
        return processed_text
    
    def step1(tv_desc):# replaces characters that are not words with spaces.
        import re#imported re function
        res1 = re.sub("[^\w]"," ",tv_desc)
        return res1
    
    def step2(res1):#splits the text into words as tokens
        from nltk.tokenize import word_tokenize #imported nltk word_tokenize
        #res2 = word_tokenize(res1)
        res2 = [word.lower() for word in word_tokenize(res1)]#converts the words into lowercase to avoid searching sensitive to case, then tokenize
        return res2
    
    def step3(res2):#filters out stopwords which are words that do not provide much information
        from nltk.corpus import stopwords#imported stopwords library from nltk
    
        #created a set of stopwords for faster lookup
        stop_words = set(stopwords.words("english"))
    
        #used list comprehension for faster filtering
        res3 = [word for word in res2 if word not in stop_words]#words that are not stopwords are retained
        return res3
    
    def step4(res3):#lemmatizes or converts words into their root word forms
    
        #created lemmatizer
        from nltk.stem import WordNetLemmatizer#imported lemmatizer
        lemmatizer = WordNetLemmatizer()
        #Similar to step 3, used list comprehension for faster processing
        res4 = [lemmatizer.lemmatize(word) for word in res3]#executed lemmatization or conversion of words
        return res4
    
    #executed the preprocessing of the tv descriptions
    #called the preprocess function for each description
    cleaned_df['description'] = cleaned_df['description'].map(preprocess)
    
    #Saved result into a csv file for search.py
    cleaned_df.to_csv('index_output.csv')
    
    #ALL in all, this program runs for about a little over than 2 mins, 
    #Please be patient