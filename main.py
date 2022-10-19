import gdown

url = 'https://drive.google.com/uc?id=15UqmiIm0xwh9mt0IYq2z3jHaauxQSTQT'
output = 'irish-lyrics-eof.txt'
# gdown.download(url, output, quiet=False)

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

tokenizer = Tokenizer()

data = open('./irish-lyrics-eof.txt').read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# print('word_index', tokenizer.word_index)
print('total words', total_words)
print('corpus len', len(corpus), (len(corpus) - 1) / 2)

input_sequences = []
for line in range(0, int((len(corpus)-1)/2)):
    token_list = tokenizer.texts_to_sequences([corpus[line]])[0]
    # print(token_list)
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        # print(n_gram_sequence)
        input_sequences.append(n_gram_sequence)

print('input sequences',len(input_sequences))

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
# print('input sequences', input_sequences)

# create predictors and label
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
print(xs,labels)
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
print(ys)
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# print(model.summary())
# print(model)
# print('word_index', tokenizer.word_index)

import os

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# history = model.fit(xs, ys, epochs=20, verbose=1, callbacks=cp_callback)


# import matplotlib.pyplot as plt
#
#
# def plot_graphs(history, string):
#     plt.plot(history.history[string])
#     plt.xlabel("Epochs")
#     plt.ylabel(string)
#     plt.show()
#
#
# plot_graphs(history, 'accuracy')
