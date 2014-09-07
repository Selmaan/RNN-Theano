import os
os.chdir("/Users/Selmaan/GitHub/RNN-Theano/trainingRNNs-master")
f = numpy.load(state['name']+'_final_state.npz')

os.chdir("/Users/Selmaan/Dropbox/Lab/Data/Trained_RNN")
numpy.savetxt(state['name']+'_b_hh.txt',f['b_hh'])
numpy.savetxt(state['name']+'_b_hy.txt',f['b_hy'])
numpy.savetxt(state['name']+'_W_hh.txt',f['W_hh'])
numpy.savetxt(state['name']+'_W_hy.txt',f['W_hy'])
numpy.savetxt(state['name']+'_W_uh.txt',f['W_uh'])