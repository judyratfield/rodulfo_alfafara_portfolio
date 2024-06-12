
The search engine works by performing a lemmatized full-text search on a 
given database, by lemmatizing both the search term and the database.

#Index
The process begins with indexing by running this in the command prompt 
python index.py --raw-data sqlite-file.sqlite to execute the program 
properly, where the database is read, and records undergo a series of 
preprocessing steps where non-word characters are replaced with spaces, 
tv descriptions are tokenized as words, words are converted to lowercase, 
and stopwords are filtered out.This time around, contrary to what was done 
during training and classification,stopwords library from nltk is utilized 
because our search function will not make use of a training model that also
tries to learn meanings and connections among the words. In other words, 
semantic analysis is not being done here, hence, for our purposes of executing
a search engine, leveraging on the stopwords nltk library is beneficial. 
Then, lemmatization is applied to convert words into their root word forms. 
Lastly, the processed text are saved into a CSV file, which is later used 
for searching. 

### Please take note, this program runs for about a little over than 2 mins
 
#Search

The process begins with search by running this in the command prompt python
search.py --input-file (search term filename.txt) --output-json-file (output
filename.json) --encoding UTF-8 In the search phase, the same preprocessing
functions are applied to both the search terms and the database to ensure 
lemmatization consistency. The search result function iterates through each 
description in the database, computes ranking scores based on the proximity 
of the search terms in the passed tv description, number of matches, and 
applies a strategy to optimize search results by considering repeating 
search terms. This is handled by computing for the standard deviations only
for the first occurences of the search terms. Then, for every repetition of
the search term from the tv description, we impose a penalty of 0.1 from 
the computed standard deviation of that description. This approach puts more
value in descriptions with more matches or hits as the penalty is a deduction 
from the standard deviation value which means that the description with more
hits or matches will have a smaller standard deviation,and in turn, will be
ranked higher. There is no need to do this for similar search terms because
usually, repeated search terms connote to very specific titles or descriptions 
where the repetition is valuable. In such cases, the repetition should be 
treated as one search term, for example: 'run, chicken, run', 'rage,rage, against the dying of the light', 'knock, knock', 'double,double, toil and trouble', etc
For single word searches on the other hand, the program computes a ranking 
based on the ratio of the length of the description to the count of occurrences 
of the search term in that description. Then, we sort this in ascending order, 
because the smaller the ratio, the better. A small ratio means that the 
description is being overpopulated by the searched term. Next, we also sort the
standard deviation computations in ascending order, because the description 
with the smallest standard deviation means that the search terms are closest 
and should be ranked higher, which is similar to the description-to-occurences 
ratios where, the smaller the value, the better.
