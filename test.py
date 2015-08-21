
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


class SentenceLayer(object):
    def __init__(self, test_file, train_file):

        # potentially add word2vec vectors
        self.train_lines, self.test_lines = self.get_lines(train_file), self.get_lines(test_file)
        lines = np.concatenate([self.train_lines, self.test_lines], axis=0)

        # samples = [x['text'] + ' ' + x['answer'] if 'answer' in x else x['text'] for x in lines]

        self.vectorizer = CountVectorizer(lowercase=False)
        self.vectorizer.fit([x['text'] + ' ' + x['answer'] if 'answer' in x else x['text'] for x in lines])

        analyze = vectorizer.build_analyzer()
        # sentences has items of varying length
        self.sentences = [[vectorizer.vocabulary_[i] for i in analyze(x['text'] )] for x in lines]
        #self.L = self.vectorizer.transform([x['text'] for x in lines]).toarray().astype(np.float32)
        #self.L_train, self.L_test = self.L[:len(self.train_lines)], self.L[len(self.train_lines):]
        self.L_train, self.L_test = self.sentences[:len(self.train_lines)], self.sentences[len(self.train_lines):]
        #L_train, L_test = sentences[:len(train_lines)], sentences[len(train_lines):]


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

    def get_V(self):
        return self.L

    def get_output(self, train=True):
        if train==True:
            return self.L_train
        else:
            return self.L_test

# step 1, learn language first

class MemLayer(object):

    def __init__(self, sentences):
        # todo add pretrained wordvectors
        # add Temporal Encoding
        # add adjacent training
        # add Linear Start
        # add train
        # add test
        # add theano functions

        # Defined embeddings
        self.A = theano.shared((np.random.uniform(-0.1, 0.1,(d, V))).astype(np.float32))
        self.B = theano.shared((np.random.uniform(-0.1, 0.1,(d, V))).astype(np.float32))
        self.C = theano.shared((np.random.uniform(-0.1, 0.1,(d, V))).astype(np.float32))


        # for Positional Encoding
        self.l = theano.shared((np.random.uniform(-0.1, 0.1,(d, V))).astype(np.float32))
        # self.W = self.init((self.input_dim, self.output_dim))

        self.Embedded_A = self.embed_all(sentences, ABC = 'A')
        self.Embedded_C = self.embed_all(sentences, ABC = 'C')
        # B is reserved for when questions are asked

    # standard embedding structure.
    def embed(self, sentence, ABC = 'A'):

        """The first is the bag-of-words (BoW) representation that takes the sentence
            xi = {xi1, xi2, ..., xin}, embeds each word and sums the resulting vectors: e.g mi =
            SUMj Axij. The input vector u representing the question is also embedded as a bag of words."""

        if ABC == 'A'
            return np.sum(self.A[sentence])
        elif ABC == 'B'
            return np.sum(self.B[sentence])                     
        elif ABC == 'C'
            return np.sum(self.C[sentence])
        else:
            raise('ABC embedding error')

    # positional endoding structure
    def positional_encoder(sentence, ABC = 'A'):
        # TODO add self.
        """We therefore propose a second representation that encodes the position of words within the
        sentence. This takes the form: mi =SUMjlj * Axij , where * is an element-wise multiplication. lj is a
        column vector with the structure lkj = (1 − j/J) − (k/d)(1 − 2j/J) (assuming 1-based indexing),
        with J being the number of words in the sentence, and d is the dimension of the embedding. This
        sentence representation, which we call position encoding (PE), means that the order of the words
        now affects mi. The same representation is used for questions, memory inputs and memory outputs.  """

        l_ = []
        for i in range(len(l)):
            # row
            k = l[i]
            for j in k:
                lkj = (1 - j/len(sentence)) - (k/l.shape[1])*(1 - (2*j)/len(sentence))
                l_.append(lkj)
        if ABC == 'A'
            xi = embed(sentence,ABC)
            return xi * sum(l_)
        elif ABC == 'B'
            xi = embed(sentence,ABC)
            return xi * sum(l_)                     
        elif ABC == 'C'
            xi = embed(sentence,ABC)
            return xi * sum(l_)
        else:
            raise('ABC embedding error')
            xi = embed(sentence,ABC)

    def embed_all(self, sentences, ABC = 'A', PE=True):
        # change numpy to Theano tensors
        # input memory representation
        if ABC == 'A'
            mi = []
            for sentence in sentences:
                if PE:
                    xi = positional_encoder(sentence, ABC)
                    mi.append(xi)
                else:
                    xi = embed(sentence, ABC)
                    mi.append(xi)
            return np.vstack(mi)

        # query representation
        elif ABC == 'B'
            # changing the name to query for identification only
            query = sentences
            u = []
            if PE:
                ui = positional_encoder(query, ABC)
                u.append(ui)
            else:
                ui = embed(query, ABC)
                u.append(ui)
            return np.vstack(u)

        # Output memory representation
        elif ABC == 'C'
            ci = []
            for sentence in sentences:
                if PE:
                    xi = positional_encoder(sentence, ABC)
                    ci.append(xi)
                else:
                    xi = embed(sentence, ABC)
                    ci.append(xi)
            return np.vstack(ci)

    def ask_question(self, query, PE=True):
        u = self.embed_all(query,ABC='B',PE=PE)
        pi = T.nnet.softmax(T.dot(u.T,self.Embedded_A))
        # feel like this is wrong
        o = pi + self.Embedded_C
        o_plus_u = o + u 
        return o_plus_u



class WeightSumLayer(object):
    def __init__(self,d,V):

        self.W = theano.shared((np.random.uniform(-0.1, 0.1,(d, V))).astype(np.float32))

    def get_output(self, nput):
        a = T.nnet.softmax( T.dot( self.W , nput ) )
        return a







class MemNN(object):
    def __init__(self):


        # add some shit here

