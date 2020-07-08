import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import Embedding
from keras import regularizers
import keras as k
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import andrews_curves

train_data=pd.read_csv('./public/train.csv')
test_data=pd.read_csv('./public/test.csv')

print(train_data.shape,test_data.shape)
print(train_data.isnull().sum())
print(test_data.isnull().sum())

diags=train_data.diag.unique()

dic={}
for i,diag in enumerate(diags):
    dic[diag]=i
labels=train_data.diag.apply(lambda x:dic[x])

val_data=train_data.sample(frac=0.24,random_state=200)

train_data=train_data.drop(val_data.index)

texts=train_data.comment_text
NUM_WORDS=20000
tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                       lower=True)
tokenizer.fit_on_texts(texts)
sequences_train = tokenizer.texts_to_sequences(texts)
sequences_valid=tokenizer.texts_to_sequences(val_data.comment_text)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_train = pad_sequences(sequences_train)
X_val = pad_sequences(sequences_valid,maxlen=X_train.shape[1])
y_train = to_categorical(np.asarray(labels[train_data.index]),num_classes=13)
y_val = to_categorical(np.asarray(labels[val_data.index]),num_classes=13)
print('Shape of X train and X validation tensor:', X_train.shape,X_val.shape)
print('Shape of label train and validation tensor:', y_train.shape,y_val.shape)

word_vectors = KeyedVectors.load_word2vec_format('./public/aimed.bin', binary=True)

EMBEDDING_DIM=200
vocabulary_size=min(len(word_index)+1,NUM_WORDS)
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if i>=NUM_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
del(word_vectors)
embedding_layer = Embedding(vocabulary_size,EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)
sequence_length = X_train.shape[1]
filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5
inputs = Input(shape=(sequence_length,))
embedding = embedding_layer(inputs)
reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)
conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)
merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
flatten = Flatten()(merged_tensor)
reshape = Reshape((3*num_filters,))(flatten)
dropout = Dropout(drop)(flatten)
output = Dense(units=13, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)
print('Output is:',output.shape)
model = Model(inputs, output)
adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
print(model.summary())
callbacks = [EarlyStopping(monitor='val_loss')]
model.fit(X_train, y_train, batch_size=10, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks)
sequences_test=tokenizer.texts_to_sequences(test_data.comment_text)
X_test = pad_sequences(sequences_test,maxlen=X_train.shape[1])
y_pred=model.predict(X_test)
to_submit=pd.DataFrame(index=test_data.id,data={'infective_endocarditis':y_pred[:,dic['infective_endocarditis']],'lymphoma':y_pred[:,dic['lymphoma']],'miliary_tuberculosis':y_pred[:,dic['miliary_tuberculosis']],'polyarteritis_nodosa':y_pred[:,dic['polyarteritis_nodosa']],'relapsing_fever':y_pred[:,dic['relapsing_fever']],'sarcoidosis':y_pred[:,dic['sarcoidosis']],'streptococcal_infection':y_pred[:,dic['streptococcal_infection']],'tuberculosis':y_pred[:,dic['tuberculosis']],'tuberculosis_of_abdomen':y_pred[:,dic['tuberculosis_of_abdomen']],'typhoid_and_paratyphoid_fever':y_pred[:,dic['typhoid_and_paratyphoid_fever']],'vasculitis':y_pred[:,dic['vasculitis']],'':y_pred[:,dic['']],'':y_pred[:,dic['']]})
to_submit.to_csv('submit.csv')
rl_data=pd.read_csv('submit1.csv')
cnn_data=pd.read_csv('submit.csv')
cnn_data.index[-1]+1
cnn_sum = cnn_data.loc[cnn_data.index[-1] + 1] = cnn_data.sum()
cnn_sum.to_csv('cnn_sum.csv')
cnn_sumdata=pd.read_csv('cnn_sum.csv')
cnn_sumdata.columns.values[[1]] = ['cnn_prob']
cnn_sumdata.to_csv('cnn_sum1.csv')
rl_cnn=pd.concat([rl_data, cnn_sumdata], axis=1)
rl_cnn.to_csv('cnn_sum2.csv')
rl_cnndata = pd.read_csv('cnn_sum2.csv',usecols=[1,2,4])
rl_cnndata.to_csv('submit2.csv')
data = read_csv('submit2.csv',usecols=[1,2,3])
data.plot(kind='bar',x='id',y='rlprob')
plt.xticks(size = 10)
plt.margins(0.2)
plt.subplots_adjust(bottom=0.4)
plt.savefig('submit_rlprob.png')
data.plot(kind='bar',x='id',y='cnn_prob')
plt.xticks(size = 10)
plt.margins(0.2)
plt.subplots_adjust(bottom=0.4)
plt.savefig('submit_cnnprob.png')
plt.show()
