#RUN this in the command prompt to execute this program properly
#python train.py --training-data sqlite-file.sqlite

if __name__== "__main__":
    import argparse#imported argaprs
    #set arguments similar to the skeleton doc
    parser = argparse.ArgumentParser(description="Train a classification model from SQLite data")
    parser.add_argument("--training-data", required=True, help="Path to the SQLite database file")

    args = parser.parse_args()
    
    training_data_filename = args.training_data

    #Since the dataset is a .sqlite, database file, we will be using the sqlite3 package
    
    #For the first part, we will first simply investigate the contents of the tvmaze
    ##database since we currently have no information on it.
    ##We will read the database using sqlite3 then get the columns and the 
    ##contents of the columns so we can strategize
    
    #imported necessary packages for the reading of the dataset
    import sqlite3
    import pandas as pd
    
    #replace 'training_data_filename' with the path to SQLite file
    #this is also going to be passed from args.training_data
    database_file = training_data_filename
    
    connection = sqlite3.connect(database_file)
    cursor = connection.cursor()
    
    #Defined the query to merge genres for each TV series
    merge_query = """
    CREATE TABLE merged_data AS
    SELECT tvmaze.tvmaze_id,
           tvmaze.update_timestamp,
           tvmaze.showname,
           tvmaze.first_airing,
           tvmaze.imdb,
           tvmaze.lang,
           tvmaze.description,
           GROUP_CONCAT(tvmaze_genre.genre, ', ') AS genre
    FROM tvmaze
    LEFT JOIN tvmaze_genre ON tvmaze.tvmaze_id = tvmaze_genre.tvmaze_id
    GROUP BY tvmaze.tvmaze_id;
    """
    
    #Executed the merge_query to create the new table
    cursor.execute(merge_query)
    
    #Defined the query to select the merged data
    select_query = "SELECT * FROM merged_data;"
    
    #Executed the query to fetch data from 'merged_data'
    cursor.execute(select_query)
    
    #Fetched all rows from the result set
    data = cursor.fetchall()
    
    cursor.close()
    connection.close()
    
    #Converted dataset into a pandas dataframe
    df = pd.DataFrame(data)
    
    # Defined a list of new column names for our dataframe
    new_column_names = ['tvmaze_id', 'update_timestamp', 'showname', 'first_airing',
                        'imdb', 'lang', 'description', 'genre']
    
    # Assigned the new column names to the DataFrame
    df.columns = new_column_names
    
    #removed records with missing data for description
    cleaned_desc = df.copy().dropna(subset=['description'])
    
    #removed records with missing data for genre
    cleaned_df = cleaned_desc.copy().dropna(subset=['genre'])
    
    #We will be handling the whitespaces to be sure
    cleaned_df['genre'] = cleaned_df['genre'].str.strip()
    
    #improted necessary packages
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    
    #defined function that cleans html tags
    def clean_html_tags(html):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        cleaned_text = soup.get_text()
        return cleaned_text
    
    #Before anything else, we'll remove the html tags first
    cleaned_df['description'] = cleaned_df['description'].map(clean_html_tags)
    
    def step1(tv_desc):#splits the text into words as tokens
        from nltk.tokenize import word_tokenize #imported nltk word_tokenize
        res1 = word_tokenize(tv_desc)
        #res1 = [word for word in word_tokenize(tv_desc)]#converts the words into lowercase to avoid searching sensitive to case, then tokenize
        return res1
        
    #then we tokenize
    cleaned_df['description'] = cleaned_df['description'].map(step1)
    
    #Next step before we train our data, we have to further clean up
    ##based on description. We will be removing records with short descriptions that
    ##provide little information. Here, we removed those with descriptions below 20 words because the 25th quantile is of the number of words in a description is 35, so 20 words is a good benchmark for short descriptions that can be removed
    
    cleaned_df = cleaned_df[cleaned_df['description'].apply(len) >= 20]
    
    #converted into strings
    cleaned_df['description'] = cleaned_df['description'].apply(lambda x: ' '.join(x))
    
    #Recall that we removed records earlier so we have to
    #now reset the indices of our dataset
    
    #reset the index of our dataset
    cleaned_df.reset_index(drop=True, inplace=True)
    
    #Now before we train, we need to convert genre values into one-hot encoded
    #values because keras needs a one-hot encoded array of targets
    
    #imported numpy
    import numpy as np
    #sorted the genres
    #split the concatenated genres into individual genres and stored them in a list
    split_genres = []#created an empty list which will contain the split genres
    for genre_list in cleaned_df['genre'].unique():#iterated through the genres from df
        split_genres.extend(genre_list.split(', '))
    
    #sorted the unique genre names
    sorted_genres = np.sort(np.unique(split_genres))

    #created numpy array target which will later on contain the genre taggings
    target = np.zeros((cleaned_df.shape[0],len(sorted_genres)))
    
    
    #created dictionary for forward lookups on genre
    genres_forward_lookup = {c:i for (i,c) in enumerate(sorted_genres)}
    
    #created dictionary for reverse lookups on genre
    genres_reverse_lookup = {i:c for (i,c) in enumerate(sorted_genres)} 
    
    #assigned values to our target numpy array which signifies the genres of our dataset 
    for i, genres_list in enumerate(cleaned_df['genre']):#iterated through the df genres
        #split the concatenated genres into a list
        genres = genres_list.split(', ')
        for genre in genres:#iterated through each genre
            genre_number = genres_forward_lookup[genre]#referred to the lookup to know the genre number equivalent
            target[i, genre_number] = 1.0#assigned value of 1 to the genres of the show

    #Now, we can begin training
    
    #imported necessary packages
    import sklearn
    from sklearn.model_selection import train_test_split
    
    #we split in two because we have 2 sources for our model, the dataset and the target numpy array
    pseudo_train1, test_data1, pseudo_train2,test_data2 = sklearn.model_selection.train_test_split(cleaned_df.description,target, test_size = 0.2)
    
    #splitting further into train and validation
    train_data1, validation_data1, train_data2, validation_data2 = sklearn.model_selection.train_test_split(pseudo_train1,pseudo_train2, test_size = 0.2)#further split the train into train and validation
    
    
    max_tokens = 10000#set tokens to 10000 
    output_sequence_length = 150#set the output sequence length to 150 because when we applied the counter function on the lengths of the descriptions, the most number of occurances were ranging from 100-300, so it's reasonable to use the middle value 
    embedding_dim = 128#set to 128 since we are only working on a small number of categories around 28 but since it is multidimensional and can have 3 final results, 128 seems to be a good spot
    
    #imported necessary packages
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import TextVectorization, Embedding, Dense, GlobalAveragePooling1D, Input
    import keras
    import tensorflow as tf
    
    #Set Vectorizer so the text can be converted into numerical format for training
    vectorizer = TextVectorization(max_tokens = max_tokens, output_sequence_length=output_sequence_length)
    vectorizer.adapt(train_data1)#adapted on train data of tv descriptions
    
    inputs = Input(shape=(1,),dtype=tf.string)#input layer#set the input shape of (1,) to treat each input as a document. Inputs expected are a variable-length string as opposed to setting shape to the vocab size which only works if we convert the input already to fixed-length vectors or arrays. In our case, this will be performed when we embed
    vectorized = vectorizer(inputs)
    embedded = Embedding(max_tokens + 1, embedding_dim)(vectorized)#created an embedding layer
    averaged = GlobalAveragePooling1D()(embedded)#text is represented by 1D this is why we use this # applies the GlobalAveragePooling1D layer to the embedded data where the average of each feature (vector component) is computed for all embedded word vectors
    #set relu as the activation function
    thinking = Dense(128,activation = "relu")(averaged)#uses the output of the Global pooling as an input to the dense layer
    #dropout layer 
    dropout = keras.layers.Dropout(0.5)(thinking)#set dropout layer to control overfitting
    #set the number of neurons of the output layer to the number of genres to be predicted and set activation function to softmax since we will be predicting on multiple classes
    output = Dense(len(sorted_genres),activation = "softmax")(dropout) #we used the output of the hidden later as the input of this output layer #we do not use sigmoid because we have lots of categories to possibly be tagged
    

    model = Model(inputs=inputs, outputs=output)
    #compiled the model and set the loss function to binary crossentropy
    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam' )#used accuracy as metrics and adam for optimizer
    model.summary()
    
    #imported callbacks
    import keras.callbacks
    #set callbacks to monitor val_loss, set a patience of 10 and enabled setting to restore to best weights
    callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights = True)
    
    #one last step before we train, given that when we analyzed the dataset, we found that the dataset is imbalanced;
    ##for the succeeding parts, we will be applying new class weights to address this imbalance
    
    #set the weights for each category. Weights have been manually computed from another python file using the formula total number of samples / (number of classes * count of class)
    new_weights_dict = {'Drama': 0.3743970315398887, 'Anime': 0.46052031036056595, 'Comedy': 0.46052031036056595, 'Action': 0.5260688216892596, 'Adventure': 0.5674915635545557, 'Mystery': 0.6704318936877076, 'Fantasy': 0.6896787423103212, 'Romance': 0.7546746447270007, 'Science-Fiction': 0.7791505791505792, 'Crime': 0.863130881094953, 'Children': 0.923992673992674, 'History': 0.960952380952381, 'Thriller': 0.9805636540330418, 'Family': 1.0009920634920635, 'Supernatural': 1.029591836734694, 'Horror': 1.0369989722507709, 'Music': 1.6379870129870129, 'Sports': 2.0301810865191148, 'Travel': 2.183982683982684, 'Medical': 2.252232142857143, 'War': 2.324884792626728, 'Nature': 2.3629976580796255, 'Food': 2.5739795918367347, 'Espionage': 3.1335403726708075, 'Adult': 3.2759740259740258, 'Western': 3.6035714285714286, 'Legal': 4.003968253968254, 'DIY': 7.207142857142857}
    
    #created an empty dictionary for the new weights
    new_weights = {}
    for genre, weight in new_weights_dict.items():#iterated through the dictionary of the new weights
        genre_idx = genres_forward_lookup[genre]#referred to the genres lookup to convert the genre texts to the numerical equivalents
        new_weights[genre_idx] = weight#assigned eack key value pair of genre(numerical value) and corresponding weight to the dictionary
        
    #fitted the model
    history = model.fit(train_data1, train_data2, validation_data = (validation_data1, validation_data2), callbacks = callback,
                        class_weight = new_weights,#set the class weight to use the newly computed weights
                        epochs = 500)#set epochs to 500 since it is better to train as much as possible as long as validation loss(error) is still decreasing; should it continuously increase, then the training should be stopped
    
    #evaluated the model
    model.evaluate(test_data1,test_data2)
    
    #Saved model for classify.py
    model.save('trained_model')
    
    #Saved dictionary for classify.py
    import pickle#imported pickle for this
    
    with open('genres.pkl','wb') as f:
        f.write(pickle.dumps(genres_reverse_lookup))

