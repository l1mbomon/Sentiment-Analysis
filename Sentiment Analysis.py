import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import pandas as pd

train, test, _ = imdb.load_data(path='imdb.pkl',
                                n_words=10000,
                                valid_portion=0.1) # 10% of data as "validation set"
trainX, trainY = train
testX,  testY  = test

pd.Series(trainX).tail()
print( list(pd.Series(trainX).iloc[5555]) )
pd.Series(trainX).shape

pd.Series(trainY).tail()
pd.Series(trainY).shape
pd.Series(trainY).value_counts()
pd.Series(trainY).value_counts().index.tolist()

len(pd.Series(trainY).value_counts().index.tolist())


# # Data Preprocessing

# ### Sequence Padding
# 
# Pad each sequence to the same length: the length of the longest sequence.
# If maxlen is provided, any sequence longer than maxlen is truncated to
# maxlen. Truncation happens off either the beginning (default) or the
# end of the sequence. Supports post-padding and pre-padding (default).

trainX = pad_sequences(trainX, maxlen=100, value=0.0)
testX  = pad_sequences(testX,  maxlen=100, value=0.0)

trainX.shape
pd.DataFrame(trainX).tail()
pd.DataFrame(testX).tail()

trainY = to_categorical(trainY, nb_classes=2)
testY  = to_categorical(testY,  nb_classes=2)
pd.DataFrame(trainY).tail()


# # Network Building

# The first element is the "batch size" which we set to "None"
# The second element is set to "100" coz we set the max sequence length to "100"
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128) # input_dim: Vocabulary size (number of ids)
net = tflearn.lstm(net, 128, dropout=0.8) # Long Short Term Memory Recurrent Layer
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, 
                         optimizer='adam', 
                         learning_rate=1e-4,
                         loss='categorical_crossentropy')


# # Training

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32)
