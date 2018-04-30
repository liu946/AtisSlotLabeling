# -- encoding: utf8
import keras
import sys
from keras.layers import *
import codecs, os
import re

def read_labels(filename):
    '''return label dictionary from a file.'''
    d = {'<PAD>': 0, 'O': 1}
    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            d['B-' + line.strip()] = len(d)
            d['I-' + line.strip()] = len(d)
    return d

def word_preprocess(word):
    if re.match(r'(DIGIT)+', word):
        word = 'DIGIT'
    return word

def read_from_file(filename, word_dict, label_dict, padding_len = None):
    ''' read data from file and update word_dict automatically return matrix of input and target. '''
    x, y = [], []
    with codecs.open(filename, encoding='utf8') as f:
        sentence, label = [], []
        for line in f:
            line = line.strip()
            if line == '':
                x.append(sentence)
                y.append(label)
                sentence, label = [], []
            else:
                line = line.split('\t')
                word = word_preprocess(line[0])
                if word not in word_dict:
                    word_dict[word] = len(word_dict)
                sentence.append(word_dict[word])
                label.append(label_dict[line[1]])
        if line != '':
            x.append(sentence)
            y.append(label)
    return (keras.preprocessing.sequence.pad_sequences(pad, maxlen=padding_len, dtype='int32', value=0) for pad in (x, y))

def get_model(word_dict, label_dict, stn_len, word_dim = None, rnn_hidden = None):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(len(word_dict), word_dim, mask_zero=True))
    model.add(Bidirectional(LSTM(rnn_hidden, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(label_dict), activation='softmax'), input_shape=(stn_len, 2 * rnn_hidden)))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def save_pred(pred_mat, label_dict, pred_file, output_file):
    '''save evaluatable format predict file from prediction matrix. '''
    reverse_map = [0] * len(label_dict)
    for i in label_dict: reverse_map[label_dict[i]] = i
    outter_counter, inner_counter = 0, 0
    stn_hold, label_hold = [], []
    with codecs.open(pred_file, encoding='utf8') as fi:
        with codecs.open(output_file, mode='w', encoding='utf8') as fo:
            for line in fi:
                line = line.strip()
                if line == '':
                    for i in range(len(label_hold)):
                        fo.write(stn_hold[i] + '\t' + label_hold[-i - 1] + '\n')
                    fo.write(line + '\n')
                    outter_counter += 1
                    inner_counter = 0
                    stn_hold, label_hold = [], []
                else:
                    stn_hold.append(line)
                    inner_counter -= 1
                    label_hold.append(reverse_map[pred_mat[outter_counter][inner_counter]])

if __name__ == '__main__':
    word_dict = {'<PAD>': 0}
    label_dict = read_labels(os.path.join('data', 'atis_slot_names.txt'))

    train_X, train_Y = read_from_file(os.path.join('data', 'atis.train.txt'), word_dict, label_dict)
    train_Y = np.expand_dims(train_Y, -1)

    test_X, test_Y = read_from_file(os.path.join('data', 'atis.test.txt'), word_dict, label_dict, len(train_X[0]))
    test_Y = np.expand_dims(test_Y, -1)

    model = get_model(word_dict, label_dict, len(train_X[0]), int(sys.argv[2]), int(sys.argv[3]))
    if len(sys.argv) != 4:
        print('usage:\n\t./main.py <train|pred> word_dim hidden_dim\n\n')
        exit(1)
    if sys.argv[1] == 'train':
        model.fit(train_X, train_Y, validation_split=0.05, batch_size=32, epochs=20, verbose=0)
        loss, acc = model.evaluate(test_X, test_Y, batch_size=32)
        model.save('atis-' + '-'.join(sys.argv[2:]) +'.model')
        print('loss=%f, acc=%f' % (loss, acc))
    elif sys.argv[1] == 'pred':
        model.load_weights('atis-' + '-'.join(sys.argv[2:]) +'.model')

    test_pred = model.predict(test_X, batch_size=32)
    test_pred = np.argmax(test_pred, axis=-1)
    save_pred(test_pred, label_dict, os.path.join('data', 'atis.test.txt'), os.path.join('pred-' + '-'.join(sys.argv[2:]) + '.out'))
