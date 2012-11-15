# assignment 6
# due Friday Nov 23, 11:59pm EST

import pickle
fid = open('traindata.pickle','r')
traindata = pickle.load(fid)
fid.close()

# traindata has 100 training examples
# traindata['inputs'] are inputs (2 per training example)
# traindata['outputs'] are outputs (1 per training example)

i1 = where(traindata['outputs']==1)[0]
i2 = where(traindata['outputs']==2)[0]
i3 = where(traindata['outputs']==3)[0]
i4 = where(traindata['outputs']==4)[0]
figure()
plot(traindata['inputs'][i1,0],traindata['inputs'][i1,1],'bs')
plot(traindata['inputs'][i2,0],traindata['inputs'][i2,1],'rs')
plot(traindata['inputs'][i3,0],traindata['inputs'][i3,1],'gs')
plot(traindata['inputs'][i4,0],traindata['inputs'][i4,1],'ms')
axis('equal')
xlabel('INPUT 1')
ylabel('INPUT 2')

# Train a neural network to learn the mapping between inputs and
# output categories

# hint: your network will have an input layer with 2 input neurons
#       a hidden layer with N neurons (you choose N)
#       you should make your output layer have 4 neurons, one for
#       each category. You will have to recode your outputs:
# 1 = [1.0, 0.0, 0.0, 0.0]
# 2 = [0.0, 1.0, 0.0, 0.0]
# 3 = [0.0, 0.0, 1.0, 0.0]
# 4 = [0.0, 0.0, 0.0, 1.0]
#       The hidden layer transfer function should be tansig
#       The output layer transfer function should be logsig


