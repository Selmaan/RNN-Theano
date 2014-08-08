
"""
Generate input and test sequence for the Delayed Match-To-Sample test

Description
-----------
The input has 2 channels.
"""

import numpy

class dMTSTask(object):
     def __init__(self, rng, floatX):
        self.rng = rng
        self.floatX = floatX
        self.nin = 1
        self.nout = 2
        self.classifType='lastSoftmax'

     def generate(self, batchsize, length):
        l = length
        p0 = self.rng.randint(int(l*.15), size=(batchsize,)) + int(l*.25)
        p0 = p0*0 + int(l*.33)
        v0 = self.rng.randint(2, size=(batchsize,))
        p1 = self.rng.randint(int(l*.15), size=(batchsize,)) + int(l*.60)
        p1 = p1*0 + int(l*.66)
        v1 = self.rng.randint(2, size=(batchsize,))
        targ_vals = abs(v1-v0) #v0+v1*2
        vals  = numpy.zeros((l, batchsize))
        for trialNum in range(batchsize):
            vals[range(p0[trialNum]-int(l*.1),p0[trialNum]), trialNum] = 2*v0[trialNum]-1
            vals[range(p1[trialNum],p1[trialNum]+int(l*.1)), trialNum] = 2*v1[trialNum]-1
        #vals[p0, numpy.arange(batchsize)] = 10*(v0-.5)
        #vals[p1, numpy.arange(batchsize)] = 10*(v1-.5)
        data = numpy.zeros((l, batchsize, 1), dtype=self.floatX)
        targ = numpy.zeros((batchsize, 2), dtype=self.floatX)
        data[:,:,0] = vals
        #data.reshape((l*batchsize, 1))
        targ[numpy.arange(batchsize), targ_vals] = 1.
        return data, targ


if __name__ == '__main__':
    print 'Testing temp Order task generator ..'
    task = dMTSTask(numpy.random.RandomState(numpy.random.randint(100)), 'float32')
    seq, targ = task.generate(5, 20)
    assert seq.dtype == 'float32'
    assert targ.dtype == 'float32'
    print 'Sequence 0'
    print '----------'
    print seq[:,0,:]
    print 'Target:', targ[0]
    print
    print 'Sequence 1'
    print '----------'
    print seq[:,1,:]
    print 'Target:', targ[1]
    print
    print 'Sequence 2'
    print '----------'
    print seq[:,2,:]
    print 'Target', targ[2]
    print
    print 'Sequence 3'
    print '----------'
    print seq[:,3,:]
    print 'Target', targ[3]
    print
    print 'Sequence 4'
    print '----------'
    print seq[:,4,:]
    print 'Target', targ[4]