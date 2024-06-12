# Training, Evaluation

### For training, it's around 8-10 mins, apologies for the wait.

The process begins by running this in the command prompt to execute the program properly
python train.py --training-data sqlite-file.sqlite. The program then loads the dataset,
splits, trains, evaluates and saves a model. Prior to training, we investigated and 
correspondingly handled the dataset issues on imbalances in the genre distribution,  
missing data, invalid data such as HTML tags, and unnecessary whitespaces.Then, we
removed records with short descriptions that provide little information. Initially, we
applied a lot more preprocessing steps than these, but after evaluating the performance
of the model several times, we found out that a lot of information is being lost when
too much preprocessing is involved, hence, making the model perform poorly. One important
step was how we handled the imbalanced dataset. Given that the difference among the counts
of each genre were huge, we addressed it by applying new class weights that would give 
larger weights to genres with fewer occurances and smaller weights to genres with more
occurances. Weights have been manually computed from another python file using the formula
total number of samples / (number of classes * count of class)   

After preprocessing,
we proceeded to convert genre labels into one-hot encoded values, and split the dataset 
into training, validation, and test sets. Then, the description text was converted into 
numerical format for training with the help of vectorization and embeddings. The model 
cosists of only one hidden layer, and one dropout layer that helps curb overfitting. 
The output sequence length has been set to 150 because when we applied the counter 
function on the lengths of the descriptions, the most number of occurances were ranging 
from 100-300, hence, it's reasonable to use the middle value of 150. The max_tokens and 
number of neurons has been set to 128 despite working only on a small number of categories
and a relatively small dataset because the prediction task is multidimensional. The input 
layer was assigned with an input shape of (1,) to treat each input as a document as inputs
expected are a variable-length string as opposed to setting shape to the vocab size which 
only works if we convert the input already to fixed-length vectors or arrays. In our case,
this is performed during embedding. The model has been applied with the GlobalAveragePooling1D 
layer to the embedded data where the average of each feature (vector component) is 
computed for all embedded word vectors. Then, we set the number of neurons of the output layer 
to the number of genres to be predicted and set activation function to softmax since 
the model will be predicting on multiple classes. Then, we set callbacks to monitor 
val_loss, a patience of 10, 500 epochs and enabled setting to restore to best weights. This
was the approach done because it is better to train as much as possible as long as 
validation loss(error) is still decreasing. The model should only stop when validation loss(error)
has been continuously increasing or if the number of epochs have been exhausted. This was 
the design as this proved to be the combination that produced the highest accuracy among 
all the other many attempts. Lastly, the model was saved for later use in the classification program.

# Classification
The process begins by running this in the command prompt to execute the program properly
python classify.py --input-file (tv description filename.txt) --output-json-file (genre 
predictions filename.json) --encoding UTF-8 --explanation-output-dir (./subdirectory name)
The classifier program, then, loads the trained model and preprocesses the 
tv descriptions to help the model make more accurate predictions. It then utilizes the 
Lime framework to provide explanations for the predictions made by the model. We included
a threshold for prediction values of 0.05 which ensures that only significant predictions 
are considered. The preprocessing stepds done were to clean html tags, remove unimportant
non-alphabet characters,tokenization, cconversion to lowercase and stopwords handling.
For the removal of stopwords, the nltk library was not used because it contains some 
important words like "not" among other things, which can drastically change the meaning of
a text. We used a stopwords list based on the words of the tv descriptions with the most 
number of occurances that do not provide any information. Moreover, when analysis has been
made on the explanation of the model, some of the unimportant words that kept on ocurring 
were added to the stopwords to be filtered out. Lastly, the program gets the 3 highest 
prediction values, following a threshold application of 0.05, where any prediction below 
the threshold value is excluded as a prediction. This was the approach done because
0.05 is the widely used significance value in statistics.

# EXTRA READ---Overview of the Analysis Activities Conducted

## Initial Dataset Structure:
I initially worked with a dataset that had one row per genre. For example, if a show had 
two genres, like action and comedy, I created separate rows for each genre 
(row 1: show1 - action, row 2: show1 - comedy). I did this to increase the dataset size, 
even though it introduced duplicates. My rationale was that a larger dataset might help 
the model learn better,and allow the model to at least assign higher probability scores to
the correct predictions even if it meant sacrificing the accuracy for the incorrect ones. 
This approach seemed reasonable since our primary goal was to secure the three highest 
predictions.

## Testing the Duplicates:
To validate my approach,  I conducted a series of manual tests. Firstly, I created another model that worked with a 
dataset without duplicates (one row per show, regardless of the number of genres). Then, I 
classified 30 TV show descriptions with both models and compared the results. 
Surprisingly, the model with duplicates, despite having lower accuracy (around 30%) as 
compared to the model without duplicates (around 44%), seemed to perform better. I 
hypothesized that this was because the duplicate model only tags one genre prediction per row.
For instance, given a show that has three genres, the model will only make one correct prediction on
the three records containing the same show:
      Actual    Prediction Scores    Final Prediction
show1 Action    Action - 60%         Action  
                Comedy - 30 %
                Sci-fi - 10%

show1 Comedy    Action - 60%         Action
                Comedy - 30 %
                Sci-fi - 10%

show1 Sci-fi    Action - 60%         Action
                Comedy - 30 %
                Sci-fi - 10%

Referring to the example above, we only get 1/3 which is around 33%, which is also the accuracy of the model,
so it made sense for me at the beginning, especially bearing in mind that our goal is not to have a high accuracy,
but to be able to make up to 3 correct genre predictions for a tv show. I just needed to place 
additional handling on classify.py to get the 3 highest scores. Then, if I did this in this example,
instead of getting only one correct prediction, despite getting a low accuracy of 30%, I will
make all the predictions correctly (Action, Comedy, Sci-fi)

## Improvements and Model Selection:
As I improved both models by adjusting parameters, layers, dimensions, and preprocessing 
steps, the model without duplicates showed a significant accuracy improvement, reaching 
up to 50%, while the duplicate model only improved slightly (1-2%). Upon manually retesting, the 
model without duplicates outperformed the duplicate model. Thus, I decided to proceed 
with the model that worked with the dataset with one row per TV show, which I used for 
my initial submission on October 30. I'll refer to this as the "original model."

# Deadline Extension and Pre-Trained Embedding:
Initially, I aimed to implement a pre-trained embedding but lacked time. However, with 
the deadline extension to November 3, I dedicated time to implementing it.

Word2Vec Training: I initially attempted training Word2Vec on our dataset and applied it 
to the model. Then, I tried using 'word2vec-google-news-300' pre-trained model. Both performed 
way worse than my original model. Next, because it was hard for me to accept that it wasn't doing 
any good considering that it has already been trained on a very large dataset and is widely being 
used around the world, not to mention, it has 1.6gb size consisting of only word vector weights;
I wanted to give it another chance so I used 'word2vec-google-news-300' pre-trained model again 
but this time around on a sequential model. It performed better than the other two, but still not 
better than my original model. My final attempt to give it another chance was to tweak the parameters, 
layers, and dimensions. After that, it now performed better than my original model but only
by a small margin. Therefore, instead of using this approach which takes about more than 2 
hours to train on a non-GPU device, I instead stuck with my original model.

## The Importance of Dataset Size:
My key takeaway from this exercise is the significance of having a large dataset. 
No matter how complex the model or the quality of embeddings, limited data points 
hinder model performance. Our dataset, initially consisting of only 65,988 records, and 
fewer(a little under than 40,000) after preprocessing, posed challenges in achieving 
substantial improvements to the model's performance.