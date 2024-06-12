# Integrating Audio Analysis with NLP for Enhanced Sentiment and Intent Recognition in Customer Interactions

__Alfafara, Rodulfo II Dela Paz__ 
__Guzman, Jelene Delos Santos__ 

## Statement of the Problem

Current NLP semantic analysis solutions have significant limitations, primarily because they focus exclusively on analyzing text context. This narrow approach often leads to miscategorization of intents, as it disregards intonations and other non-verbal audio cues. For instance, a statement like "Can you help me please?" might be labeled as neutral by most NLP contact center solutions, but if the speaker is shouting, the sentiment should be categorized as angry rather than neutral.

Similarly, phrases such as "Very good! Job well done!" may seem positive when analyzed purely as text. However, if delivered with a high, sarcastic intonation, the sentiment should be recognized as negative. Additionally, non-verbal nuances like sighs, groans, and background noises can indicate negative sentiment, which are often overlooked by existing solutions.

Our project addresses this issue by integrating an analysis of waveforms, frequencies, and other audio characteristics into semantic analysis, providing a more accurate understanding of the speaker's intent and sentiment.

## This repository consists of the following notebooks:

1) Text Semantic Analysis - This notebook deals with the text context analysis part of the proposed solution. 

2) Audio Signals Analysis - processes audio signals that accompanied the text of the dataset and performs audio signals analysis techniques. This notebook should output audio signals readings for the dataset. 

3) Accent_Classifier - classifies which agent should be assigned to caller based on accent detected from analysing audio signals

4) Age_Gender_Classifier - classifies which agent should be assigned to caller based on age and Gender

5) Neural Network - Contains the Dataset Preparation, Design, Training & Evaluation of the proposed Neural Network model that analyzes both text context and audio signals 

6) Classify - combines solutions from 1-5 to classify sentiment of customer and to which agent should the customer be assigned to

7) Evaluation - Notebook that contains evaluations done on new unseen recordings, including Ablation study