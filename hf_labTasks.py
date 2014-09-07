# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 17:47:32 2014

@author: Selmaan
"""

from rnn import MetaRNN
from hf import SequenceDataset, hf_optimizer
import numpy as np
import matplotlib.pyplot as plt
import logging

def test_dCue(n_updates=1000):
    """ Test RNN with softmax outputs. """
    n_hidden = 50
    n_in = 2
    n_steps = 60
    n_seq = 40
    n_out = 2
    cueOn = 0.2 
    cueOff = 0.25
    decideOn = 0.25    
    
    
    np.random.seed(np.random.randint(1e3))    
    cueOn = round(cueOn*n_steps)
    cueOff = round(cueOff*n_steps)
    decideOn = round(decideOn*n_steps)
    trialConditions = np.random.randint(0,2,n_seq)
    seq = np.zeros((n_seq, n_steps, n_in))
    targets = np.zeros((n_seq, n_steps, n_in), dtype='int32')
    for trial in xrange(n_seq):
        seq[trial,cueOn:cueOff,trialConditions[trial]] = 1
        targets[trial,decideOn:,trialConditions[trial]] = 1
    
    

    # SequenceDataset wants a list of sequences
    # this allows them to be different lengths, but here they're not
    seq = [i for i in seq]
    targets = [i for i in targets]

    gradient_dataset = SequenceDataset([seq, targets], batch_size=None,
                                       number_batches=2)
    cg_dataset = SequenceDataset([seq, targets], batch_size=None,
                                 number_batches=2)

    model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    activation='tanh', output_type='binary')

    # optimizes negative log likelihood
    # but also reports zero-one error
    opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
                       s=model.rnn.y_pred,
                       costs=[model.rnn.loss(model.y),
                              model.rnn.errors(model.y)], h=model.rnn.h)

    # using settings of initial_lambda and mu given in Nicolas' RNN example
    # seem to do a little worse than the default
    opt.train(gradient_dataset, cg_dataset, num_updates=n_updates)

#    seqs = xrange(10)
#
#    plt.close('all')
#    for seq_num in seqs:
#        fig = plt.figure()
#        ax1 = plt.subplot(211)
#        plt.plot(seq[seq_num])
#        ax1.set_title('input')
#
#        ax2 = plt.subplot(212)
#        # blue line will represent true classes
#        true_targets = plt.step(xrange(n_steps), targets[seq_num], marker='o')
#
#        # show probabilities (in b/w) output by model
#        guess = model.predict_proba(seq[seq_num])
#        guessed_probs = plt.imshow(guess.T, interpolation='nearest',
#                                   cmap='gray')
#        ax2.set_title('blue: true class, grayscale: probs assigned by model')
            
    return model

if __name__ == "__main__":
    modelName = 'dCue001_'
    logging.basicConfig(level=logging.INFO)
    model = test_dCue()
    np.savetxt(modelName + 'W_hh.txt',model.rnn.params[0].get_value())
    np.savetxt(modelName + 'W_in.txt',model.rnn.params[1].get_value())
    np.savetxt(modelName + 'W_out.txt',model.rnn.params[2].get_value())
    np.savetxt(modelName + 'h0.txt',model.rnn.params[3].get_value())
    np.savetxt(modelName + 'b_h.txt',model.rnn.params[4].get_value())
    np.savetxt(modelName + 'b_y.txt',model.rnn.params[5].get_value())