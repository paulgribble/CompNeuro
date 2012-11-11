# Figure 20.25 from AIMA by Russell and Norvig. Back-propagation Neural Net

import math

def g(x):                                        # tanh faster than the standard 1/(1+e^-x)
    return math.tanh(x)

def gp(y):                                       # derivative of g
    return 1.0-y*y

def BACK_PROP_LEARNING(examples, network) :
    alpha = 0.2
    (Wih, Who) = network
    for epoch in range(4000) :
        for (x,y) in examples :
            ai=x[:]                                           # inputs/outputs
            deltao = [0.0]*len(y)
            deltah = [0.0]*len(Who[0])
            ah = [0.0]*len(Who[0])
            ao = [0.0]*len(y)

            for j in range(len(ah)) :                         # activate hidden layer
                ini = 0.0
                for k in range(len(ai)) :
                    ini = ini+Wih[j][k]*ai[k]
                    ah[j] = g(ini)                            # hidden activation
                    
            for i in range(len(ao)) :                         # activate output layer
                ini = 0.0
                for j in range(len(ah)) :           
                    ini = ini+Who[i][j]*ah[j]
                    ao[i] = g(ini)                            # output activation
                    deltao[i] = gp(ao[i])*(y[i]-ao[i])        # output error gradient
                    
            for k in range(len(ah)) :
                error = 0.0
                for j in range(len(y)) :                      # back propagate to hidden
                    error = error + Who[j][k]*deltao[j]
                    deltah[k] = gp(ah[k]) * error             # hidden error gradient

            for j in range(len(ah)) :
                for i in range(len(ao)) :                     # update output weights
                    Who[i][j] = Who[i][j] + (alpha * ah[j] * deltao[i])

            for k in range(len(ai)) :
                for j in range(len(ah)) :                     # update hidden weights
                    Wih[j][k] = Wih[j][k] + (alpha * ai[k] * deltah[j])

    return network

def BACK_PROP_TEST(examples, network) :
    result = []
    (Wih, Who) = network
    for (x,y) in examples :
        eresult=[]
        ai=x[:]                                          # inputs/outputs
        ah = [0.0]*len(Who[0])
        
        for j in range(len(ah)) :                        # activate hidden layer
            ini = 0.0
            for i in range(len(ai)) :           
                ini = ini+Wih[j][i]*ai[i] 
                ah[j] = g(ini)                           # hidden activation
                
        for k in range(len(y)) :                         # activate output layer
            ini = 0.0
            for j in range(len(ah)) :           
                ini = ini+Who[k][j]*ah[j]

            eresult.append(g(ini))                       # output activation
            result.append(eresult)
            return result
            
XOR = [ ([0,0], [0]),                                    # Training examples
        ([0,1], [1]), 
        ([1,0], [1]), 
        ([1,1], [0])] 

NN = [[[0.1, -0.2],                                   # input to hidden weights
       [-0.3, 0.4]],                                  # 2 input and 2 hidden
      [[0.5, -0.6]]                                   # hidden to output weights
]                                                     # 2 hidden and 1 output

print 'Learning results: ', BACK_PROP_LEARNING(XOR, NN) 
print 'Training and test set: ', XOR 
print 'Test results: ', BACK_PROP_TEST(XOR, NN)
