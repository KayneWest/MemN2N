from __future__ import division
import argparse
import glob
import numpy as np
import sys
from collections import OrderedDict
from sklearn import metrics
from sklearn.feature_extraction.text import *
from sklearn.preprocessing import *
from theano.ifelse import ifelse
import theano
import theano.tensor as T

if theano.config.floatX!='float32':
    theano.config.floatX=='float32'

# from keras 
def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def uniform(shape, scale=0.01):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))

def glorot_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    s = T.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s)

def glorot_numpy(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    s = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(low=-s, high=s, size=shape)

def categorical_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    c = T.nnet.categorical_crossentropy(y_pred, y_true)
    return c

def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates

def accuracy(y, t):
    if t.ndim == 2:
        t = np.argmax(t, axis=1)
    predictions = np.argmax(y, axis=1)
    return np.mean(predictions == t)


from keras.initializations import glorot_uniform


class TextLayer(object):
    def __init__(self, test_file, train_file, d = 30):
        # houses embedding B. 
        # deals with text
        # d = 30 to limit size of vectors

        # potentially add word2vec vectors
        self.d = d
        self.train_lines, self.test_lines = self.get_lines(train_file), self.get_lines(test_file)
        lines = np.concatenate([self.train_lines, self.test_lines], axis=0)
        self.vectorizer = CountVectorizer(lowercase=False)
        self.vectorizer.fit([x['text'] + ' ' + x['answer'] if 'answer' in x else x['text'] for x in lines])
        self.analyze = self.vectorizer.build_analyzer()

        self.V = len(self.vectorizer.vocabulary_)

        #self.B = theano.shared((np.random.uniform(-0.1, 0.1,(self.d, self.V))).astype(np.float32))
        self.B = glorot_uniform((self.d, self.V)) 

        self.params = [ self.B ]
        self.updates = []

    def embed(self, sentence):
        val = [self.B.T[x] for x in sentence]
        return sum(val)

    def embed_all(self, sentences):
        output = []
        for sentence in sentences:
            xi = self.embed(sentence)
            output.append(xi)
        return T.concatenate(output,axis=0)

    def positional_encoding(self, xi):
        # ensure L has same dimension as B.
        # xi is a list of lists. 
        """We therefore propose a second representation that encodes the position of words within the
        sentence. This takes the form: mi =SUMjlj * Axij , where * is an element-wise multiplication. lj is a
        column vector with the structure lkj = (1 − j/J) − (k/d)(1 − 2j/J) (assuming 1-based indexing),
        with J being the number of words in the sentence, and d is the dimension of the embedding. This
        sentence representation, which we call position encoding (PE), means that the order of the words
        now affects mi. The same representation is used for questions, memory inputs and memory outputs.  """
        L = glorot_uniform((len(xi), self.d))
        J = [len(sentence) for sentence in xi]
        lj = (-L.T / J ).T - ( L / self.d ) * (-2 * L.T).T
        return lj

    def return_train(self, positional_encoding = False):
        train_qs = []
        self.train_as = []
        self.train_xi = []
        for line in self.train_lines:
            if line['type'] == 'q':
                text = [self.vectorizer.vocabulary_[i] for i in self.analyze(line['text'])]
                answer = [self.vectorizer.vocabulary_[i] for i in self.analyze(line['answer'])]
                self.train_as.append(answer)
                train_qs.append(text)
            else:
                text = [self.vectorizer.vocabulary_[i] for i in self.analyze(line['text'])]
                self.train_xi.append(text)
        
        # embed xi
        if positional_encoding:
            lj = self.positional_encoding(train_qs)
            self.train_qs = self.embed_all(train_qs) * lj
            return self.train_qs, self.train_as, self.train_xi
        else:
            self.train_qs = self.embed_all(train_qs)
            #self.train_xi = self.embed_all(self.train_xi)
            return self.train_qs, self.train_as, self.train_xi            

    def return_test(self):
        test_qs = []
        self.test_as = []
        self.test_xi = []
        for line in self.test_lines:
            if line['type'] == 'q':
                text = [self.vectorizer.vocabulary_[i] for i in self.analyze(line['text'])]
                answer = [self.vectorizer.vocabulary_[i] for i in self.analyze(line['answer'])]
                self.test_as.append(answer)
                test_qs.append(text)
            else:
                text = [self.vectorizer.vocabulary_[i] for i in self.analyze(line['text'])]
                self.test_xi.append(text)
        
        # embed xi
        lj = self.positional_encoding(test_qs)
        self.test_qs = self.embed_all(test_qs) * lj
        return self.test_qs, self.test_as, self.test_xi
                 
    def get_lines(self, fname):
        lines = []
        for i,line in enumerate(open(fname)):
            id = int(line[0:line.find(' ')])
            line = line.strip()
            line = line[line.find(' ')+1:]        
            if line.find('?') == -1:
                lines.append({'type':'s', 'text': line})
            else:
                idx = line.find('?')
                tmp = line[idx+1:].split('\t')
                lines.append({'id':id, 'type':'q', 'text': line[:idx], 'answer': tmp[1].strip(), 'refs': [int(x) for x in tmp[2:][0].split(' ')]})
            if False and i > 1000:
                break
        return np.array(lines)


# step 1, learn language first
class MemLayer(object):

    def __init__(self, d, V):
        # d + V are inherited from textlayer
        # Defined embeddings
        self.d = d
        self.V = V
        self.A = glorot_uniform((d, V))
        self.C = glorot_uniform((d, V))
        self.params = [self.A, self.C]
        self.updates = []


    def init_xi_ci(self, xi, positional_encoding = False):
        if positional_encoding:
            self.lj = self.positional_encoding(xi)
            self.Embedded_A = self.embed_all(xi, embedding=self.A)
            self.Embedded_C = self.embed_all(xi, embedding=self.C)
            # B is reserved for when questions are asked
            self.mi = self.Embedded_A * self.lj
            self.ci = self.Embedded_C * self.lj
            self.embed_params = [self.Embedded_A, self.Embedded_C]
        else:
            self.Embedded_A = self.embed_all(xi, embedding=self.A)
            self.Embedded_C = self.embed_all(xi, embedding=self.C)
            # B is reserved for when questions are asked
            self.mi = self.Embedded_A
            self.ci = self.Embedded_C
            self.embed_params = [self.Embedded_A, self.Embedded_C]


    def embed_all(self, sentences, embedding):
        output = []
        for sentence in sentences:
            xi = self.embed(sentence, embedding)
            output.append(xi)
        return T.concatenate(output,axis=0)

    # standard embedding structure.
    def embed(self, sentence, embedding):
        """The first is the bag-of-words (BoW) representation that takes the sentence
        xi = {xi1, xi2, ..., xin}, embeds each word and sums the resulting vectors: e.g mi =
        SUMj Axij. The input vector u representing the question is also embedded as a bag of words."""
        val = [embedding.T[x] for x in sentence]
        return sum(val)

    # positional endoding structure
    def positional_encoding(self, xi):
        """We therefore propose a second representation that encodes the position of words within the
        sentence. This takes the form: mi =SUMjlj * Axij , where * is an element-wise multiplication. lj is a
        column vector with the structure lkj = (1 − j/J) − (k/d)(1 − 2j/J) (assuming 1-based indexing),
        with J being the number of words in the sentence, and d is the dimension of the embedding. This
        sentence representation, which we call position encoding (PE), means that the order of the words
        now affects mi. The same representation is used for questions, memory inputs and memory outputs.  """
        if type(xi)==np.ndarray or type(xi)==list: # if it's 
            L = glorot_uniform((len(xi), self.d))
            J = [len(sentence) for sentence in xi]
            lj = (-L.T / J ).T - ( L / self.d ) * (-2 * L.T).T
            return lj
        else:
            L = glorot_uniform((xi.shape[0], self.d))
            J = [sentence.shape[0] for sentence in xi]
            lj = (-L.T / J ).T - ( L / self.d ) * (-2 * L.T).T
            return lj            


    def get_output(self, u):
        pi = T.nnet.softmax(T.dot(u,self.mi.T))
        o = T.dot(self.ci.T, pi)
        return o

    def ask_question(self, question, positional_encoding=False):
        if positional_encoding:
            question = self.positional_encoding(question)
            response = self.get_output(question)
            return response
        else:
            response = self.get_output(question)
            return response            


# final relative to the model from Sukhbaatar et al. 
class FinalWeightLayer(object):
    """ Final Weight Layer of Mem Network """
    def __init__(self, d, V):
        self.W = glorot_uniform((V, d))
        self.params = [ self.W ]

    def get_output(self, o, u):
        answer = T.nnet.softmax( T.dot( self.W , (u.T + o) ) )
        return answer



def model_init(test_file,train_file):
    x_layer = TextLayer(test_file, train_file)
    mem_layer = MemLayer(d=30, V=x_layer.V)
    final_layer = FinalWeightLayer(d=30, V=x_layer.V)
    return x_layer, mem_layer, final_layer

train_file ='/Users/Krzus/data/tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_train.txt'
test_file = '/Users/Krzus/data/tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_test.txt'


x_layer, mem_layer, final_layer = model_init(test_file, train_file)
x_layer.return_train()
mem_layer.init_xi_ci(x_layer.train_xi)


def forward(x,mem_layer=mem_layer,final_layer=final_layer):
    response = mem_layer.get_output(x)
    result = final_layer.get_output(x,response)
    return result



params = []
params.extend(x_layer.params) 
params.extend(mem_layer.params)
params.extend(final_layer.params)
assert(test_file)


t_x = T.ivector('t_x')
t_y = T.ivector('t_y')

t_x = T.matrix('t_x')

t_y_softmax = forward(t_x)
# change to categorical cross from keras library.
cross_entropy = T.mean(T.nnet.categorical_crossentropy(t_y_softmax, t_y))
gradients = T.grad(cost=cross_entropy, wrt=params)
gradient_norms = [gradient.norm(L=2) for gradient in gradients]
updates = rmsprop(cross_entropy, params)

compile_start_time = time.time()

train_fn = theano.function(inputs=[t_x, t_y],
                           outputs=[cross_entropy],
                           updates=updates)

predict_fn = theano.function(inputs=[t_x],
                            outputs=[t_y_softmax])


gradient_norms_fn = theano.function(inputs=[t_x, t_y],
                                    outputs=gradient_norms)

print "compilation took %0.3f s" % (time.time()-compile_start_time)


def train(self, n_epochs):
    # init model
    model_init()
    train_qs, train_as, train_xi = x_layer.return_train()
    x_layer.init_xi_ci(train_xi)
    start()
    costs = []
    prev_err = None
    for epoch in range(n_epochs):
        total_err = 0
        print "*" * 80
        print "epoch: ", epoch
        n_wrong = 0
        for i, question, answer in enumerate(zip(train_qs, train_as)):
            cost,  = train_fn( question, answer )
            costs.append(float(cost))
            print "i: ", i, " cost: ", np.mean(costs)
        print "epoch: ", epoch, " err: ", (total_err/len(train_lines))
        # use last example trained as an example of L2-norms of gradients
        sample_gradient_l2_norms = OrderedDict()
        for param, norm in zip(params, gradient_norms_fn(x, y)):
            sample_gradient_l2_norms[param.name] = float(norm)

def test(self):
    test_qs, test_as, test_xi = x_layer.return_test()
    x_layer.init_xi_ci(test_xi)

    accuracy = []
    for i, question, answer in enumerate(zip(test_qs, test_as)):
        y_softmaxs  = predict_fn( question, answer )
        # costs.append(float(cost))
        # print "i: ", i, " cost: ", np.mean(costs)
        acc = accuracy(y_softmaxs, answer)
        accuracy.append(acc)
        print "    acc:\t%.2f%%" % (acc * 100)





