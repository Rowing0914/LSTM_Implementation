# python = 2.7

import numpy as np
import csv
import itertools
import operator
import nltk
import sys
from datetime import datetime
from utils import *
import matplotlib.pyplot as plt


class LSTM:
    def __init__(self, word_dim, nepoch, learning_rate, gradient_check_threshold, hidden_dim, bptt_truncate):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.learning_rate = learning_rate
        self.nepoch = nepoch
        self.h = 0.01
        self.error_threshold = gradient_check_threshold

        # Randomly initialize the network parameters: input to memory cell
        self.W_hx = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, word_dim))
        self.W_outx = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, word_dim))
        self.W_inx = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, word_dim))

        # Randomly initialize the network parameters: memory cell to memory cell
        self.W_hh = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        self.W_outh = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.W_inh = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

        # Randomly initialize the network parameters: memory cell to prediction
        self.W_yh = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))

        # maybe not needed
        self.S = np.zeros((hidden_dim, hidden_dim))
        self.cell_state = np.zeros((hidden_dim, hidden_dim))

    def one_hot(self, x):
        return np.eye(self.word_dim)[x]

    def forward_propagation(self, x):
        # The total number of time steps in a sentence
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        self.cell_output = np.zeros((T + 1, self.hidden_dim))
        self.cell_output[-1] = np.zeros(self.hidden_dim)
        self.cell_input = np.zeros((T + 1, self.hidden_dim))
        self.cell_input[-1] = np.zeros(self.hidden_dim)

        # memory cell's gates
        self.y_in = np.zeros((T + 1, hidden_dim))
        self.y_out = np.zeros((T + 1, hidden_dim))

        # The outputs at each time step. Again, we save them for later.
        prediction = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # initialise the lenght for time to store the CEC_state
            if t == 0: self.CEC_state = np.zeros((T, hidden_dim))
            # combine current input and previous hidden layer as an input for a memory cell
            self.cell_input[t] = np.tanh(self.W_hx[:, x[t]] + self.W_hh.dot(self.cell_output[t-1]))
            # passing current input and previous hidden layer to memory cell
            self.lstm_memory_cell(x, t)
            # prediction output
            prediction[t] = softmax(self.W_yh.dot(self.cell_output[t]))
        return prediction

    def lstm_memory_cell(self, x, t):
        # input gate
        self.y_in[t] = sigmoid( self.W_inx[:, x[t]] + np.dot(self.W_inh, self.cell_output[t-1]) )
        # output gate
        self.y_out[t] = sigmoid( self.W_outx[:, x[t]] + np.dot(self.W_outh, self.cell_output[t-1]) )
        # internal state of cell
        if t == 0:
            self.CEC_state[t] = (self.y_in[t] * self.cell_input[t])
        else:
            self.CEC_state[t] = self.CEC_state[t-1] + (self.y_in[t] * np.tanh(self.cell_input[t]))
        # cell output
        self.cell_output[t] = np.tanh(self.CEC_state[t]) * self.y_out[t]
        return "Done"
        
    def bptt(self, x, y):
        T = len(x)
        # Perform forward propagation
        prediction = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdW_yh = np.zeros(self.W_yh.shape)
        dLdW_hh = np.zeros(self.W_hh.shape)
        dLdW_hx = np.zeros(self.W_hx.shape)
        dLdW_outh = np.zeros(self.W_outh.shape)
        dLdW_inh = np.zeros(self.W_inh.shape)
        dLdW_outx = np.zeros(self.W_outx.shape)
        dLdW_inx = np.zeros(self.W_inx.shape)
        error = prediction
        # derivative of Cross entropy is (target - predict), element-wise, (1 - outcome of softmax) 
        error[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            # backprop to the weight between memory cell and prediction: error * activated hidden layer
            delta_L = np.outer(error[t], self.cell_output[t].T)
            dLdW_yh += delta_L
            # declare the internal state for truncate BPTT
            dLdS = np.zeros((self.bptt_truncate + 1, self.hidden_dim))
            # index for truncate bppt
            i = 0
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # condense the backpropagated error to hidden layer size
                delta_e = np.dot(self.W_yh.T, error[bptt_step])
                # backprop to the weight of output gate
                dLdW_outx += np.outer((delta_e * np.tanh(self.CEC_state[bptt_step])), self.one_hot(x[bptt_step]))
                dLdW_outh += np.outer(delta_e, np.tanh(self.CEC_state[bptt_step]) * self.cell_output[bptt_step-1])
                # preparation of backprop for memory cell state update purpose
                dLdS[i] = delta_e * self.y_out[bptt_step] * (1 - np.tanh(self.CEC_state[bptt_step])**2)
                # backprop to the weight of input gate
                dLdW_inx += np.outer(dLdS[i] * np.tanh(self.cell_input[bptt_step]), self.one_hot(x[bptt_step]))
                dLdW_inh += np.outer(self.cell_output[bptt_step-1], dLdS[i] * np.tanh(self.cell_input[bptt_step]))
                # backprop to the weight between input to memory cell
                dLdW_hx += np.outer(dLdS[i] * self.y_in[bptt_step] * (1 - np.tanh(self.cell_input[bptt_step])**2), self.one_hot(x[bptt_step]))
                dLdW_hh += np.outer(self.cell_output[bptt_step-1], dLdS[i] * self.y_in[bptt_step] * (1 - np.tanh(self.cell_input[bptt_step])**2))
                # index for truncate bppt
                i += 1

        return [dLdW_yh, dLdW_hh, dLdW_hx, dLdW_outh, dLdW_inh, dLdW_outx, dLdW_inx]

    def gradient_check(self, x, y):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = model.bptt(x, y)
        # change this according to the parameter
        model_parameters = ['W', 'V', 'U']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + self.h
                gradplus = model.calculate_total_loss([x],[y])
                parameter[ix] = original_value - self.h
                gradminus = model.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*self.h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > self.error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return 
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        prediction = self.forward_propagation(x)
        return np.argmax(prediction, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            prediction = self.forward_propagation(x[i])
            # We only care about our prediction of the "target" words
            correct_word_predictions = prediction[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N

    def from_index_to_word(index_words):
        res = []
        for i in index_words:
            res.append(index_to_word[i])
        return res

    # Performs one step of SGD.
    def sdg_step(self, x, y):
        # Calculate the gradients
        dLdW_yh, dLdW_hh, dLdW_hx, dLdW_outh, dLdW_inh, dLdW_outx, dLdW_inx = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.W_yh -= self.learning_rate * dLdW_yh
        self.W_hh -= self.learning_rate * dLdW_hh
        self.W_hx -= self.learning_rate * dLdW_hx
        self.W_outh -= self.learning_rate * dLdW_outh
        self.W_inh -= self.learning_rate * dLdW_inh
        self.W_outx -= self.learning_rate * dLdW_outx
        self.W_inx -= self.learning_rate * dLdW_inx

    # Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
    def train_with_sgd(self, model, X_train, y_train, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(self.nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = model.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    self.learning_rate = self.learning_rate * 0.5  
                    print "Setting learning rate to %f" % self.learning_rate
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                model.sdg_step(X_train[i], y_train[i])
                num_examples_seen += 1

    def generate_sentence(self, model, word_to_index, index_to_word):
        # We start the sentence with the start token
        new_sentence = [word_to_index[sentence_start_token]]
        # Repeat until we get an end token
        while not new_sentence[-1] == word_to_index[sentence_end_token]:
            prediction = model.forward_propagation(new_sentence)
            sampled_word = word_to_index[unknown_token]
            # We don't want to sample unknown words
            while sampled_word == word_to_index[unknown_token]:
                samples = np.random.multinomial(1, prediction[-1])

                sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
        return sentence_str

class DataPreparation:
    def __init__(self, word_dim, sentence_start_token, sentence_end_token):
        self.word_dim=word_dim
        self.sentence_start_token=sentence_start_token
        self.sentence_end_token=sentence_end_token

    def data_preprocessing(self):
        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        print "Reading CSV file..."
        with open('data/data.csv', 'rb') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            # Split full comments into sentences
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
            # Append SENTENCE_START and SENTENCE_END
            sentences = ["%s %s %s" % (self.sentence_start_token, x, self.sentence_end_token) for x in sentences]
        print "Parsed %d sentences." % (len(sentences))
        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print "Found %d unique words tokens." % len(word_freq.items())
        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(word_dim-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
        print "Using vocabulary size %d." % word_dim
        print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
        print "\nExample sentence: '%s'" % sentences[0]
        print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
        # Create the training data
        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
        return X_train, y_train, word_to_index, index_to_word
        

if __name__ == '__main__':
    word_dim                 = 400
    learning_rate            = 0.005
    nepoch                   = 100
    hidden_dim               = 100
    bptt_truncate            = 1
    gradient_check_threshold = 0.001
    unknown_token            = "UNKNOWN_TOKEN"
    sentence_start_token     = "SENTENCE_START"
    sentence_end_token       = "SENTENCE_END"
    np.random.seed(10)
    
    data = DataPreparation(word_dim, sentence_start_token, sentence_end_token)
    X_train, y_train, word_to_index, index_to_word = data.data_preprocessing()
    
    # Train on a small subset of the data to see what happens
    print "Initialise Model"
    model = LSTM(word_dim, nepoch, learning_rate, gradient_check_threshold, hidden_dim, bptt_truncate)

    print "Train Model"
    model.train_with_sgd(model, X_train, y_train, evaluate_loss_after=1)

    print "Training done!"
    print "Generate random sentence based on the model!"
    print "Let's see how it goes!"
    num_sentences = 10
    senten_min_length = 7
    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = model.generate_sentence(model, word_to_index, index_to_word)
        print " ".join(sent)
