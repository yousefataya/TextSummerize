#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will build an abstractive based text summarizer using deep learning from the scratch in python using keras
# 
# I recommend you to go through the Kersa , NLTK , Tensorflow DeepLearning

# #Understanding the Problem Statement
# 
# Customer reviews can often be long and descriptive. Analyzing these reviews manually, as you can imagine, is really time-consuming. This is where the brilliance of Natural Language Processing can be applied to generate a summary for long reviews.
# 
# We will be working on a really cool dataset. Our objective here is to generate a summary for the Trump Deal About Palestine reviews using the abstraction-based approach we learned about above. 
# 
# It’s time to fire up our Jupyter notebooks! Let’s dive into the implementation details right away.
# 
# #Custom Attention Layer
# 
# Keras does not officially support attention layer. So, we can either implement our own attention layer or use a third-party implementation. We will go with the latter option for this article.
# 
# Let’s import it into our environment:

# In[ ]:





# #Import the Libraries

# In[ ]:

from keras_self_attention import SeqSelfAttention
import argparse
import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import tensorflow as tf
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from keras.models import Model , Sequential
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
import warnings
from PyPDF2 import PdfFileReader
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import sent_tokenize
stemmer = WordNetLemmatizer()
import six
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence
# #Read the dataset
# 
# You can Provide Csv and also you can provide Pdfs
# 
# We’ll take a sample of 100,000 reviews to reduce the training time of our model. Feel free to use the entire dataset for training your model if your machine has that kind of computational power.

# In[ ]:

# Here is more more complex I read from pdf PyPdf2 Lib pip install PyPDF2 ad read the pdf. I test it on trump pdf. also i test on csv sample about palestine dataset Palestine Authority National University in Palestine
#data=pd.read_csv("sample.csv",nrows=100000) 
data = ''
text = ''
text__ = ''
startPage = 0
cleanText = ''
stop_words = set(stopwords.words('english')) 
documents = []


import csv
total____________ =  ' '
iseral___________ = 0
palestine__________ = 0
with open("samplenlp.pdf", 'rb') as f:
        pdf = PdfFileReader(f)
        information = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()
        count = pdf.numPages
        for i in range(count):
            pageObj = pdf.getPage(i)
            text = pageObj.extractText()

            #from sklearn.feature_extraction.text import CountVectorizer
            #vectorizer = CountVectorizer(max_features=1500, min_df=0.1, max_df=0.95, stop_words=stopwords.words('english'))
            #X = vectorizer.fit_transform(documents).toarray()
            #from sklearn.feature_extraction.text import TfidfTransformer
            #tfidfconverter = TfidfTransformer()
            #arrys____ = tfidfconverter.fit_transform(X).toarray()
            text________________ = ''
            split__________ = text.split()
            for i___________ in  split__________:
                       text________________ +=  ' ' +str(i___________)

            text________________ = text________________.split()
            tokens = [w for w in text________________ if not w in stop_words]


            statementg_____________ = ' '
            for k_________ in  tokens:
               
                k_________.replace('.','')
                
                #statementg_____________.replace('!','')
                #statementg_____________.replace('?','')
                k_________.replace(':','')
                k_________.replace(')','')
                k_________.replace('(','')
                k_________.replace('\n','')
                k_________.replace('\r','')
                k_________.replace('\"','')
                k_________.replace('\'','')
                statementg_____________ += ' '+str(k_________)
            print('..........')  
            newRow = pd.DataFrame( {'text' : statementg_____________},index = [0] ,columns = ['text']   )
            newRow['text'].replace('.','')
            newRow['text'].replace('!','')
            newRow['text'].replace('?','')
            newRow['text'].replace(':','')
            newRow['text'].replace(')','')
            newRow['text'].replace('(','')
            newRow['text'].replace('\n','')
            newRow['text'].replace('\r','')
            newRow.drop_duplicates()
            newRow.dropna() 
            all__________ = ' '
            for i in range(len(newRow)) : 
               all__________  += newRow.loc[0, "text"]
            cc________________  = ' ' 
            sentences = sent_tokenize(all__________)  
            for o__________ in sentences:
                cc________________  +=  o__________   + '\n'

            total____________ += cc________________ 
df = pd.DataFrame( {'classifier' : total____________},index = [0] ,columns = ['classifier']   )
df.drop_duplicates()
df.dropna()
df.to_csv("statement_____.csv", sep=',', encoding='utf-8')
print('palestine__________' + str(total____________.count('Palestinian')))
print('iseral___________' + str(total____________.count('Israeli')))
# # Drop Duplicates and NA values

# In[ ]:


#data.drop_duplicates( subset=None, keep="first", inplace=False)#dropping duplicates
#data.dropna(axis=0,inplace=True)#dropping na


# # Information about dataset
# 
# Let us look at datatypes and shape of the dataset

# In[ ]:


# #Preprocessing
# 
# Performing basic preprocessing steps is very important before we get to the model building part. Using messy and uncleaned text data is a potentially disastrous move. So in this step, we will drop all the unwanted symbols, characters, etc. from the text that do not affect the objective of our problem.
# 
# Here is the dictionary that we will use for expanding the contractions:

# In[ ]:


# We will perform the below preprocessing tasks for our data:
# 
# 1.Convert everything to lowercase
# 
# 2.Remove HTML tags
# 
# 3.Contraction mapping
# 
# 4.Remove (‘s)
# 
# 5.Remove any text inside the parenthesis ( )
# 
# 6.Eliminate punctuations and special characters
# 
# 7.Remove stopwords
# 
# 8.Remove short words
# 
# Let’s define the function:

# In[ ]:



newString = ''
def text_cleaner(text,num):
    str = " "
   # newString = BeautifulSoup(newString, "lxml").text
   # newString = re.sub(r'\([^)]*\)', '', newString)
   # newString = re.sub('"','', newString)
   # newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
   # newString = re.sub(r"'s\b","",newString)
   # newString = re.sub("[^a-zA-Z]", " ", newString) 
   # newString = re.sub('[m]{2,}', 'mm', newString)
    str___ = text.lower()
    str =" "+ str___
    token______ = str.split()
    tokens = [w for w in token______ if not w in stop_words]
    tokens____ = [w for w in tokens if not w in contraction_mapping]
    str______ = ''
    for c___ in tokens____:
        str______ += ' ' + c___
    newRow = pd.DataFrame(index = [0] ,columns = ["Rows"]   , data = [str______])
    newRow.drop_duplicates()
    newRow.dropna()
    all___ = ' '
    select_text_______ = ''
    for i in range(len(newRow)) : 
               all___ = newRow.loc[0, "Rows"] 


     
    
    return (" ".join(all___)).strip()


# In[ ]:


#call the function
cleaned_text = []
cleaned_text.append(text_cleaner(data,0)) 


# Let us look at the first five preprocessed reviews

# In[ ]:


cleaned_text[:5]  


# In[ ]:


#call the function
cleaned_summary = []
cleaned_summary.append(text_cleaner(data,1))


# Let us look at the first 10 preprocessed summaries

# In[ ]:


cleaned_summary[:10]


# In[ ]:


datacleantext=cleaned_text
datacleaned_summary=cleaned_summary


# #Drop empty rows

# In[ ]:


#data.replace('', np.nan, inplace=True)
#data.dropna(axis=0,inplace=True)


# #Understanding the distribution of the sequences
# 
# Here, we will analyze the length of the reviews and the summary to get an overall idea about the distribution of length of the text. This will help us fix the maximum length of the sequence:

# In[ ]:


import matplotlib.pyplot as plt

text_word_count = []
summary_word_count = []

# populate the lists with sentence lengths
cltext = ' '
clsummary = ''
counter__summary = 0
counter__text = 0
for x__ in datacleantext :
    counter__text +=1
    cltext +=x__
for y__ in datacleaned_summary:
    counter__summary +=1
    clsummary +=y__


text_word_count.append(counter__text)
summary_word_count.append(counter__summary)

length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})

length_df.hist(bins = 30)
plt.show()


# Interesting. We can fix the maximum length of the summary to 8 since that seems to be the majority summary length.
# 
# Let us understand the proportion of the length of summaries below 8

# In[ ]:


cnt=0
for i in datacleaned_summary:
    if(counter__summary<=8):
        cnt=cnt+1
print(cnt/counter__summary)


# We observe that 94% of the summaries have length below 8. So, we can fix maximum length of summary to 8.
# 
# Let us fix the maximum length of review to 30

# In[ ]:


max_text_len=30
max_summary_len=8


# Let us select the reviews and summaries whose length falls below or equal to **max_text_len** and **max_summary_len**

# In[ ]:

count____txt = 0
count____summary = 0
txt_____ = ''
summary____ = ''
cleaned_text =np.array(cltext)
cleaned_summary=np.array(clsummary)

short_text=[]
short_summary=[]
for i________ in cltext:
    count____txt +=1
    short_text.append(i________)

for y________ in clsummary:
    count____summary +=1
    short_summary.append(y________)

print(count____txt) 
print(count____summary) 

df=pd.DataFrame( {'text':cleaned_text,'summary':cleaned_summary} , columns= ['text','summary'] , index=[0])
df.drop_duplicates()
df.dropna()
# Remember to add the **START** and **END** special tokens at the beginning and end of the summary. Here, I have chosen **sostok** and **eostok** as START and END tokens
# 
# **Note:** Be sure that the chosen special tokens never appear in the summary

# In[ ]:

str_________ = ' ' + 'iseral'
df['summary'] = df['summary'].apply(lambda x : str_________ + str(x))

# We are getting closer to the model building part. Before that, we need to split our dataset into a training and validation set. We’ll use 90% of the dataset as the training data and evaluate the performance on the remaining 10% (holdout set):

# In[ ]:


from sklearn.model_selection import train_test_split
x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['text']),np.array(df['summary']),test_size=0.25,random_state=0,shuffle=True ) 

print(len(x_tr))
print(len(y_tr))
# #Preparing the Tokenizer
# 
# A tokenizer builds the vocabulary and converts a word sequence to an integer sequence. Go ahead and build tokenizers for text and summary:
# 
# #Text Tokenizer

# In[ ]:


from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_tr))


# #Rarewords and its Coverage
# 
# Let us look at the proportion rare words and its total coverage in the entire text
# 
# Here, I am defining the threshold to be 4 which means word whose count is below 4 is considered as a rare word

# In[ ]:


thresh=4

cnt=0
tot_cnt=0
freq=0
tot_freq=0
print(len(x_tokenizer.word_counts.items()))
for key,value in x_tokenizer.word_counts.items():
    tot_cnt=tot_cnt+1
    tot_freq=tot_freq+value
    cnt=cnt+1
    freq=freq+value
    
#print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
#print("Total Coverage of rare words:",(freq/tot_freq)*100)


# **Remember**:
# 
# 
# * **tot_cnt** gives the size of vocabulary (which means every unique words in the text)
#  
# *   **cnt** gives me the no. of rare words whose count falls below threshold
# 
# *  **tot_cnt - cnt** gives me the top most common words 
# 
# Let us define the tokenizer with top most common words for reviews.

# In[ ]:


#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
x_tokenizer.fit_on_texts(list(x_tr))

#convert text sequences into integer sequences
x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)

#padding zero upto maximum length
x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
x_val   =   pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

#size of vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1


# In[ ]:


x_voc


# #Summary Tokenizer

# In[ ]:


#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_tr))


# #Rarewords and its Coverage
# 
# Let us look at the proportion rare words and its total coverage in the entire summary
# 
# Here, I am defining the threshold to be 6 which means word whose count is below 6 is considered as a rare word

# In[ ]:


thresh=6

cnt=0
tot_cnt=0
freq=0
tot_freq=0

print(len(y_tokenizer.word_counts.items()))

for key,value in y_tokenizer.word_counts.items():
    tot_cnt=tot_cnt+1
    tot_freq=tot_freq+value
    print(value)
    cnt=cnt+1
    freq=freq+value
    
#print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
#print("Total Coverage of rare words:",(freq/tot_freq)*100)


# Let us define the tokenizer with top most common words for summary.

# In[ ]:


#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
y_tokenizer.fit_on_texts(list(y_tr))

#convert text sequences into integer sequences
y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr) 
y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) 

#padding zero upto maximum length
y_tr    =   pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val   =   pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

#size of vocabulary
y_voc  =   y_tokenizer.num_words +1


# Let us check whether word count of start token is equal to length of the training data

# In[ ]:


#y_tokenizer.word_counts['deal'],len(y_tr)   


# Here, I am deleting the rows that contain only **START** and **END** tokens

# In[ ]:


ind=[]
for i in range(len(y_tr)):
    cnt=0
    for j in y_tr[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_tr=np.delete(y_tr,ind, axis=0)
x_tr=np.delete(x_tr,ind, axis=0)


# In[ ]:


ind=[]
for i in range(len(y_val)):
    cnt=0
    for j in y_val[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_val=np.delete(y_val,ind, axis=0)
x_val=np.delete(x_val,ind, axis=0)


# # Model building
# 
# We are finally at the model building part. But before we do that, we need to familiarize ourselves with a few terms which are required prior to building the model.
# 
# **Return Sequences = True**: When the return sequences parameter is set to True, LSTM produces the hidden state and cell state for every timestep
# 
# **Return State = True**: When return state = True, LSTM produces the hidden state and cell state of the last timestep only
# 
# **Initial State**: This is used to initialize the internal states of the LSTM for the first timestep
# 
# **Stacked LSTM**: Stacked LSTM has multiple layers of LSTM stacked on top of each other. 
# This leads to a better representation of the sequence. I encourage you to experiment with the multiple layers of the LSTM stacked on top of each other (it’s a great way to learn this)
# 
# Here, we are building a 3 stacked LSTM for the encoder:

# In[ ]:


from keras import backend as K 
K.clear_session()

latent_dim = 300
embedding_dim=100

# Encoder
encoder_inputs = Input(shape=(max_text_len,))

#embedding layer
enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)

#encoder lstm 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

#encoder lstm 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#encoder lstm 3
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

#embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

# Attention layer
model = Sequential()
model.add(Embedding(input_dim=10000,
                                 output_dim=300,
                                 mask_zero=True))
model.add(Bidirectional(LSTM(units=128,
                                                       return_sequences=True)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(Dense(units=5))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
)
model.summary()

# I am using sparse categorical cross-entropy as the loss function since it converts the integer sequence to a one-hot vector on the fly. This overcomes any memory issues.

# In[ ]:


model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')


# Remember the concept of early stopping? It is used to stop training the neural network at the right time by monitoring a user-specified metric. Here, I am monitoring the validation loss (val_loss). Our model will stop training once the validation loss increases:
# 

# In[ ]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)


# We’ll train the model on a batch size of 128 and validate it on the holdout set (which is 10% of our dataset):

# In[ ]:




# #Understanding the Diagnostic plot
# 
# Now, we will plot a few diagnostic plots to understand the behavior of the model over time:

# In[ ]:





def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['iseral']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='iseral'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'iseral'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


# Let us define the functions to convert an integer sequence to a word sequence for summary as well as the reviews:

# In[ ]:




