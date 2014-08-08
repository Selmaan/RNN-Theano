f = numpy.load(state['name']+'_final_state.npz')

numpy.savetxt(state['name']+'_b_hh.txt',f['b_hh'])
numpy.savetxt(state['name']+'_b_hy.txt',f['b_hy'])
numpy.savetxt(state['name']+'_W_hh.txt',f['W_hh'])
numpy.savetxt(state['name']+'_W_hy.txt',f['W_hy'])
numpy.savetxt(state['name']+'_W_uh.txt',f['W_uh'])