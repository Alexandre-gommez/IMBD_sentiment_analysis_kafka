from kafka import KafkaProducer, KafkaConsumer
import time
import requests
import keras.backend as K
from keras.layers import Layer as Layer
from keras import initializers, regularizers, constraints
import json
import numpy as np
from keras.layers import Bidirectional, GRU
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, TimeDistributed, Dense
import tensorflow as tf
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

def dot_product(x, kernel):
    """
    https://github.com/richliao/textClassifier/issues/13#issuecomment-377323318
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
    
    def __init__(self, return_coefficients=False,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                  initializer=self.init,
                                  name='{}_W'.format(self.name),
                                  regularizer=self.W_regularizer,
                                  constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                      initializer='zero',
                                      name='{}_b'.format(self.name),
                                      regularizer=self.b_regularizer,
                                      constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                  initializer=self.init,
                                  name='{}_u'.format(self.name),
                                  regularizer=self.u_regularizer,
                                  constraint=self.u_constraint)
        
        super(AttentionWithContext, self).build(input_shape)
    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        
        if self.bias:
            uit += self.b
        
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        
        a = K.exp(ait)
        
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        weighted_input = K.sum(a * x, axis=1, keepdims=True) ### fill the gap ### # compute the attentional vector
                
        if self.return_coefficients:
            return [weighted_input, a] ### fill the gap - [attentional vector, coefficients] ###
        else:
            return weighted_input ### fill the gap - attentional vector only ###
    
    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]

def bidir_gru(my_seq, n_units):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    '''
    return Bidirectional(GRU(n_units, return_sequences=True))(my_seq)

path_root = './'
path_to_data = path_root + 'data/'

d = 30 # dimensionality of word embeddings
n_units = 50 # RNN layer dimensionality
drop_rate = 0.5 # dropout
mfw_idx = 2 # index of the most frequent words in the dictionary 
            # 0 is for the special padding token
            # 1 is for the special out-of-vocabulary token

padding_idx = 0
oov_idx = 1
batch_size = 32
nb_epochs = 2
my_optimizer = 'adam'
my_patience = 2 # for early stopping strategy

my_docs_array_train = np.load(path_to_data + 'docs_train.npy')
my_docs_array_test = np.load(path_to_data + 'docs_test.npy')

my_labels_array_train = np.load(path_to_data + 'labels_train.npy')
my_labels_array_test = np.load(path_to_data + 'labels_test.npy')

# load dictionary of word indexes (sorted by decreasing frequency across the corpus)
with open(path_to_data + 'word_to_index.json', 'r') as my_file:
    word_to_index = json.load(my_file)

# invert mapping
index_to_word = dict((v,k) for k,v in word_to_index.items()) ### fill the gap (use a dict comprehension) ###

sent_ints = Input(shape=(my_docs_array_train.shape[2],)) # vec of ints of variable size
sent_wv = Embedding(input_dim=len(index_to_word)+2, # vocab size
                    output_dim=d, # dimensionality of embedding space
                    input_length=my_docs_array_train.shape[2],
                    trainable=True
                    )(sent_ints)

sent_wv_dr = Dropout(drop_rate)(sent_wv)
sent_wa = bidir_gru(sent_wv_dr, n_units) ### fill the gap ### # get the annotations for each word in the sent
sent_att_vec, word_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_wa)### fill the gap ### # get the attentional vector for the sentence
sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)                      
sent_encoder = Model(sent_ints, sent_att_vec_dr)

doc_ints = Input(shape=(my_docs_array_train.shape[1],my_docs_array_train.shape[2],))
sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)### fill the gap ### # apply the sentence encoder model to each sentence in the document. Search for 'TimeDistributed' in https://keras.io/layers/wrappers/
doc_sa = bidir_gru(sent_att_vecs_dr,n_units) ### fill the gap ### # get annotations for each sent in the doc
doc_att_vec, sent_att_coeffs = AttentionWithContext(return_coefficients=True)(doc_sa)### fill the gap ### # get attentional vector for the doc
doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)
                
preds = Dense(units=1,
              activation='sigmoid')(doc_att_vec_dr)

model = Model(doc_ints,preds)

model.compile(loss='binary_crossentropy',
              optimizer = my_optimizer,
              metrics = ['accuracy'])

print('model compiled')
model.summary()

model.load_weights('moviemodel')

nltk.download('punkt')

def pad_sent(sent): 
    if len(sent)>= 30:
        return sent[:30]
    else:
        return sent + [0]*(30 - len(sent))


consumer = KafkaConsumer('imbd-review', bootstrap_servers='localhost:9092')
producer = KafkaProducer(bootstrap_servers='localhost:9092')

while True:
    for review_json in consumer:
        review = json.loads(review_json.value.decode('utf-8'))
        title = review['title']
        txt = review['content']
        content = review['content']
        txt = [word_tokenize(sent) for sent in sent_tokenize(txt)]
        if len(txt) >= 7 :
            txt = txt[:7]
        else:
            txt = txt + [[0]*30]*(7 - len(txt))
        f = open("reviews.txt", "a")        
        txt_input = [[word_to_index[word] if word in word_to_index else 0 for word in sentence] for sentence in txt]
        txt_input = [pad_sent(sent) for sent in txt_input]
        print(f'Review: {content}')
        f.write(f'Review: {content}\n')
        print(f'Review for the movie: {title}\nFrom: {review["username"]}')
        f.write(f'Review for the movie: {title}\nFrom: {review["username"]}\n')
        grade = model.predict(np.array([txt_input]))[0][0]
        if model.predict(np.array([txt_input]))[0][0] > 0.5:
            print(f'Good review\nGrade from the model: {grade}')
            f.write(f'Good review\nGrade from the model: {grade}\n')
        else:
            print(f'Bad review\nGrade from the model: {grade}')
            f.write(f'Bad review\nGrade from the model: {grade}\n')
        f.close()
    time.sleep(5)