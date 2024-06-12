#RUN this in the command prompt to execute this program properly
#python search.py --input-file (search term filename.txt) --output-json-file (output filename.json) --encoding UTF-8

#For this search program, we will be performing a lemmatized full-text search by lemmatizing both the search term and the database

#imported necessary packages
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

#set database to be the output of index.py
database = pd.read_csv("index_output.csv")


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

#created lemmatizer same as the one done from index.py
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#defined lemmatizing function for the search term
def lemmatizing_query(search_term):
    return [lemmatizer.lemmatize(x).lower() for x in word_tokenize(search_term)]#set to lowercase and tokenized based on words similar to what was done in the tv descriptions from index.py

#defined word search function
def word_search(database_descriptions, word_to_search_for):
    def word_is_present(where):#checks if word_to_search_for is present in the input 
        return word_to_search_for in where
    return database_descriptions.map(word_is_present)#returns the match

#defined multiword search function
def multiword_search(database_descriptions, search_term):
    search_result = pd.Series(index=database_descriptions.index, data=True)#preps the index of the result
    for word in search_term:#iterates through each word found in the search term
        search_result = search_result & word_search(database_descriptions, word)#calls the word search function
    return search_result

#defined search result function
def search_result(ranked_descriptions):
    search_results = []#created an empty list that will contain the search results
    result_counter = 0#initialized counter
    for idx in ranked_descriptions.sort_values().index:#iterated through each of the indices of the sorted(ascending order) ranking scores of the descriptions
        result = {}#created an empty dictionary that will hold the search results
        #got tvmaze_id of idx
        tv_id = int(database['tvmaze_id'][idx])
        #got showname of idx
        show = database['showname'][idx]
        result['tvmaze_id'] = tv_id #created key-value pair for the tv maze id
        result['showname'] = show #created key value pair for the showname
        search_results.append(result) #appends the search result for each iteration on the ranked_descriptions
        result_counter = result_counter + 1 #increases result counter by 1 per appended search result
        if result_counter >= 3:#this condition limits the search results to be displayed by 3
            break
    return search_results

#defined the same set of functions used from index.py. This will be called for the search terms so that both the database and the search terms will be lemmatized and processed in the same way
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


#Take note that to optimize our search results, we also need to take into account repeating search terms from the description
##To handle this, our strategy will be to compute for the standard deviations only for the first occurences of the search terms
## and then, for every repetition of the search term from the tv description, we impose a penalty of 0.1 from the computed standard deviation of that description
##This approach puts more value in descriptions with more matches or hits as the penalty is a deduction from the standard deviation value 
##which means that the description with more hits or matches will have a smaller standard deviation, 
##and in turn, will be ranked higher

#There is no need to do this for similar search terms because usually repeated search terms
##connote to very specific titles or descriptions where the repetition is valuable.
##In such cases, the repetition should be treated as one search term, for example: 'run, chicken, run', 'rage,rage, against the dying of the light', 'knock, knock', 'double,double, toil and trouble', etc

#imported statistic for the computation of the standard deviation
import statistics

#defined multiword ranking function that ranks each matched description
def multiword_ranking(new_matched_descriptions, processed_search_term):
    if len(processed_search_term) == 1:#For single word searches, computes a ranking based on the ratio of the length of the description to the count of occurrences of the search term in that description.
        ratio_list = [] #created an empty ratio list which will contain all the ratio values to be computed
        for description in new_matched_descriptions:#iterated through all the processed matched descriptions
            count_matches = 0#initialized the count of matches
            for term in processed_search_term:#iterated through the search terms
                for word in description:#iterated through the words found in the description
                    if word == term:#condition that matches the word currently being assessed with the search term currently being assessed
                        count_matches = count_matches + 1#adds an additional count per match or hit
            if count_matches == 0:#this condition is set to ensure that should there be no matches, the denominator for the ratio computation will not be 0, which results to undefined and fails the program
                count_matches = 0.01
            desc_ratio = len(description) / count_matches #computed for the ratio of the description
            ratio_list.append(desc_ratio)#this now contains the list of ratios for each description
        
        #Next step for us is to pair these with the indices of the descriptions so we can rank the descriptions

        #first we get the indices of the matched descriptions, then we convert it into a list
        descriptions_indices = list(matched_descriptions.index)

        #then we create a new pandas series combining these indices with their corresponding ratios
        ratio_descriptions = pd.Series(ratio_list, index=descriptions_indices)

        #then, we sort this in ascending order, because the smaller the ratio, the better. 
        ##A small ratio means that the description is being overpopulated by the searched term

        sorted_ratio_descriptions = ratio_descriptions.sort_values()
        return sorted_ratio_descriptions

    else:#for searches with more than one word, ranking strategy is based on standard deviation
        #created an empty list which will contain all the computed ranking values based on standard deviation
        std_list = []
        for description in new_matched_descriptions:#iterated through all descriptions
            positions_list = []#created an empty list which will contain all the positions of the matched searches
            penalty = 0#initialized the penalty
            for term in processed_search_term:#iterated through all the processed search terms
                count_term = 0 #initialized the count term
                for idx, word in enumerate(description):#iterated through all the words found in the description
                    if word == term:#this defines what the program will do if there is a match
                        if count_term < 1:#this ensures that repetitions of the matched term and word from the description will not be part of the standard deviation computation
                            #got the position of the word
                            word_position = int(idx)#got the position of the word from the descripton and converted to int
                            positions_list.append(word_position)#appended the position of the word to the positions list
                            count_term = count_term + 1#increases the count of matched terms by 1 
                        else:
                            penalty = penalty + 1#if the word currently being assessed is already a repeated match, then instead of getting its position as an input for the computation of standard deviation, it will instead increase the penalty count by 1

            if len(positions_list) >=2:#this ensures that those that will undergo standard deviation computations contain at least 2 elements because having only one element or none does not allow for the computation of standard deviation
                
                #computed std
                desc_std = statistics.stdev(positions_list) - penalty*0.1#as mentioned earlier, we deduct the number of repeated matches multiplied by 0.1
                #appended to the std list
                std_list.append(desc_std)#this now contains the ranking values based on standard deviations for each description
            else:#should there be positions_list with only one element or none, then, it will follow the same handling as the single search term searches where the ratio of the length of the description to the count of occurrences of the search term will be computed
                ratio_list = []
                for description in new_matched_descriptions:
                    count_matches = 0
                    for term in processed_search_term:
                        for word in description:
                            if word == term:
                                count_matches = count_matches + 1
                    if count_matches == 0:
                        count_matches = 0.01
                    desc_ratio = len(description) / count_matches
                    ratio_list.append(desc_ratio)#this now contains the list of ratios for each description
                
                #Next step for us is to pair these with the indices of the descriptions so we can rank the descriptions

                #first we get the indices of the matched descriptions, then we convert it into a list
                descriptions_indices = list(matched_descriptions.index)

                #then we create a new pandas series combining these indices with their corresponding ratios
                ratio_descriptions = pd.Series(ratio_list, index=descriptions_indices)

                #then, we sort this in ascending order, because the smaller the ratio, the better. 
                ##A small ratio means that the description is being overpopulated by the searched term

                sorted_ratio_descriptions = ratio_descriptions.sort_values()
                return sorted_ratio_descriptions

        #Next step for us is to pair these with the indices of the descriptions so we can rank the descriptions

        #first we get the indices of the matched descriptions, then we convert it into a list
        descriptions_indices = list(matched_descriptions.index)

        #then we create a new pandas series combining these indices with their corresponding standard deviations
        std_descriptions = pd.Series(std_list, index=descriptions_indices)

        #then, we sort this in ascending order, 
        ##because the description with the smallest standard deviation means that 
        ##the search terms are closest and should be ranked higher

        sorted_descriptions = std_descriptions.sort_values()
        return sorted_descriptions

#imported argparse to aid in consuming the passed file containing the search term
import argparse

if __name__== "__main__":
    
    parser = argparse.ArgumentParser(description="Search for TV shows based on a query")
    parser.add_argument("--input-file", required=True, help="Path to the input file with the search query")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for matched shows")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")

    args = parser.parse_args()
    
    input_filename = args.input_file
    
    #read the passed search term
    with open(input_filename, 'r') as f:
        search_term = f.read()#sets the search term to variable search_term
    
    interpreted = preprocess(search_term) #performs the same preprocessing steps done on the database in order to sucessfully execute a Lemmatized full-text search
    search_matches = multiword_search(database.description, interpreted)#for each description, True or False is returned if there is a match
    matched_descriptions = database.description[search_matches]#returns descriptions from the database where the corresponding value in the search_matches is True. Essentially, this retrieves descriptions that met the search criteria 
    
    #We had to clean up to properly rank because the formatting was changed from the series of preprocessing steps done previously
    new_matched_descriptions = []
    for description in matched_descriptions:
        cleaned_description = description.replace("[", "").replace("]", "").replace("'","").strip()
        list_description = cleaned_description.split(',')
        corrected_word_list = []
        for word in list_description:
            new_word = word.strip()
            corrected_word_list.append(new_word)
        new_matched_descriptions.append(corrected_word_list) #this is now a list of lists(descriptions)
    
    
    ranked_descriptions = multiword_ranking(new_matched_descriptions, interpreted)#for each description, rank value is provided
    

    #called the search result function to get the search results
    top_searches = search_result(ranked_descriptions)
    
    #imported json because we want to output a json file that contains the top 3 search results
    import json
    
    #specified the output JSON file name that contains the search results
    output_filename = args.output_json_file
    
    #saved the search results into to a JSON file
    with open(output_filename, 'w') as json_file:
        json.dump(top_searches, json_file)
    print(top_searches)



