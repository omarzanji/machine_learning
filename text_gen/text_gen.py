

import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM


# from official Keras docs
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# from official Keras docs
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - length_seq - 1)
    generated = ''
    sentence = text[start_index: start_index + length_seq]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, length_seq, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_ndx[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                                 temperature)
        next_character = ndx_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

article = 'news_0006410'

with open("./database/%s.json" % (article), 'r') as f:
    data = json.load(f)

with open("./texts/%s.txt" % (article), 'w') as w:
    w.write(data['text'])

text = open("./texts/%s.txt" % (article), 'rb').read().decode(encoding='utf-8').lower()
characters = sorted(set(text))

char_to_ndx = dict((c, i) for i, c in enumerate(characters))
ndx_to_char = dict((i, c) for i, c in enumerate(characters))

length_seq = 40
step = 3
sentences = []
next_char = []

for i in range(0, len(text) - length_seq, step):
    sentences.append(text[i: i+length_seq])
    next_char.append(text[i+length_seq])

x = np.zeros((len(sentences), length_seq, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

for i, sent in enumerate(sentences):
    for t, char in enumerate(sent):
        x[i, t, char_to_ndx[char]] = 1
    y[i, char_to_ndx[next_char[i]]] = 1


# building RNN
model = Sequential()
model.add(LSTM(128, input_shape=(length_seq, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=4)


if __name__ == "__main__":
    print(generate_text(1000, 0.9))