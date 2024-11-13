from __future__ import absolute_import, division, print_function  # Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import re
import unicodedata
import re
import numpy as np
import time


# print(tf.reduce_sum(tf.random.normal([1000, 1000])))
# print(tf.__version__)




script_path = os.path.dirname(__file__)
the_dataset = os.path.join(script_path + '/eng_igbo_dataset.csv')


# Converts the unicode file to ascii
def unicode_to_ascii(the_file):
    return ''.join(c for c in unicodedata.normalize('NFD', the_file)
                   if unicodedata.category(c) != 'MN')
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    w = re.sub(r"([?.!,])", r" \1 ", w)

    # adding a start and an end token to the sentence
    # so that the model knows when to start and stop predicting
    w = '<start> ' + w + ' <end>'
    return w

# remove the accents
# clean the sentences
# return word pairs in the format: [ENGLISH, IGBO]

def create_dataset(path,num_examples):
    lines = open(path,encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return word_pairs

# This class creates a word -> index mapping (e.g dad -> 5 and vice versa)

class LanguageIndex():
    def __init__(self,lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word =  {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] =0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word

def max_length(tensor):
    return max(len(t) for t in tensor)

def load_dataset(path, num_examples):
    # creating cleaned input, output pairs
    pairs = create_dataset(path,num_examples)

# index language using the class defined above
    inp_lang = LanguageIndex(ig for ig in pairs)
    targ_lang = LanguageIndex(en for en in pairs)

    # Vectorize the input and target languages

    # Igbo sentences
    input_tensor = [[inp_lang.word2idx[s] for s in ig.split(' ')] for  ig in pairs]

    # English sentences
    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')]for en in pairs]

    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp = max_length(input_tensor)
    max_length_tar = max_length(target_tensor)

    # padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, mxlen=max_length_inp, paddin='post')

    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen=max_length_tar,padding='post')

    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp,max_length_tar


num_examples = 3000
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(the_dataset,
                                                                                                 num_examples)

input_tensor_train, input_tensor_value, target_tensor_train, target_tensor_value = train_test_split(input_tensor, target_tensor, test_size=0.2)

# PreDefining some Values

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# The Encoder and Decoder Model

def gru(units):
    # If you have a GPU, it is recommended to use CuDNNGRU( provides a 3x speedup than GRU)
    # the code automatically does that
    return tf.keras.layers.GRU(units,return_sequences=True, return_state=True, recurrent_activation='sigmoid',recurrent_initializer="glorot_uniform")

class Encoder(tf.keras.Model):
    def __init__(self,vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)
    
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
    
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)


        def call(self, x, hidden, enc_output):
            # enc_output shape == (batch_size, max_length, hiddden_size)

            # hidden shape == (batch_size, hidden size)
            # hidden_with_time_axis shape == (batch_size, 1, hidden size)

            # we are doing this to perform addition to calculate the score 
            hidden_with_time_axis = tf.expand_dims(hidden, 1)

            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
            score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

            # attention weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)

            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * enc_output
            context_vector = tf.reduce_sum(context_vector, axis=1)

            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

            # passing the concatenated vector to the GRU
            output, state = self.gru(x)

            # output shape == (batch_size * 1, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))

            # output shape == (batch_size * 1, vocab)
            x = self.fc(output)

            return x, state, attention_weights
        def initialize_hidden_state(self):
            return tf.zeros((self.batch_sz, self.dec_units))
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)


optimizer = tf.train.AdamOptimizer()
def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


EPOCHS = 10
for epoch in range(EPOCHS):
    start = time.time()

    hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(the_dataset):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input 
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, the_underscore = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing 
                dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            total_loss += batch_loss

            variables = encoder.variables + decoder.variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradient(zip(gradients,variables))

