import os
import pickle
import numpy as np
from keras.models import Sequential
import gensim
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn.model_selection import train_test_split
import json
import nltk
from gensim import corpora, models, similarities
from keras.models import load_model

os.chdir("C:\\Users\\Mazhar\\Desktop\\TextsummarizationChat")
model=gensim.models.Doc2Vec.load('doc2vec.bin');
path2="corpus";
with open(path2+'/cnn_dataset.pkl','rb')as fp:
    storie= pickle.load(fp)
head=(storie[i]['highlights'])
desc=(storie[i]['story'])
x=[]
y=[]
for i in range(8):
    x.append(str(storie[i]['story']));
    y.append(str(storie[i]['highlights']));
tok_x=[]
tok_y=[]
for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))
sentend=np.ones((300,),dtype=np.float32)

vec_x=[]
for sent in tok_x:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_x.append(sentvec)
nofw=49
vec_y=[]
for sent in tok_y:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_y.append(sentvec)
for tok_sent in vec_x:
    tok_sent[49:]=[]
    tok_sent.append(sentend)
    
for tok_sent in vec_x:
    if len(tok_sent)<50:
        for i in range(50-len(tok_sent)):
            tok_sent.append(sentend)
                      
for tok_sent in vec_y:
    tok_sent[49:]=[]
    tok_sent.append(sentend)
      
for tok_sent in vec_y:
    if len(tok_sent)<50:
        for i in range(50-len(tok_sent)):
            tok_sent.append(sentend)
vec_x=np.array(vec_x,dtype=np.float32)
vec_y=np.array(vec_y,dtype=np.float32) 
x_train,x_test, y_train,y_test = train_test_split(vec_x, vec_y, test_size=0.2, random_state=1)
 
model=Sequential()
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

from keras.models import load_model
model.fit(x_train, x_train, nb_epoch=20,validation_data=(x_test, x_test))
model.save('LSTM100.h5')


