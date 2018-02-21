import numpy as np
import ailib
import random

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST", one_hot=True, validation_size=10000)

random.seed(0)
net = ailib.neuralnetwork()
net.push_variable("INPUT1", 28, 28)

net.push_convolution("CONV1", "INPUT1", 4, 4, 10, 1, 1)
net.push_normalization("NORM1", "CONV1", 0.5)
net.push_relu("REL1", "NORM1")
net.push_maxpooling("MAX1", "REL1")
#net.push_dropout("DROP1", "MAX1", 0.1)
net.push_convolution("CONV2", "MAX1", 3, 3, 8, 1, 1)
net.push_normalization("NORM2", "CONV2", 0.5)
net.push_relu("REL2", "NORM2")
net.push_averagepooling("AVG", "REL2")
#net.push_dropout("DROP2", "AVG", 0.1)
net.push_convolution("CONV3", "AVG", 3, 3, 8, 1, 1)
net.push_normalization("NORM3", "CONV3", 0.5)
net.push_relu("REL3", "NORM3")
net.push_linear("FC1", "REL3", 128)
net.push_normalization("NORM4", "FC1", 0.5)
net.push_relu("REL4", "NORM4")
#net.push_dropout("DROP3", "REL4", 0.4)
net.push_linear("FC2", "REL4", 10)
net.push_softmax("OUTPUT", "FC2")

"""
random.seed(0)
net.push_linear("FC1", "INPUT1", 100)
net.push_normalization("NORM4", "FC1", 0.5)
net.push_relu("REL4", "NORM4")
net.push_linear("FC2", "REL4", 10)
net.push_softmax("OUTPUT", "FC2", 1.0)
a = np.array([ float(i)/32.0 for i in range(32)]).astype("f")
print(a)
net.run(a)
print(net.getoutput("OUTPUT"))

net.push_convolution("CONV1", "INPUT1", 4, 4, 4, 1, 1)
net.push_linear("FC1", "CONV1", 128)
net.push_normalization("NORM4", "FC1")
net.push_relu("REL4", "NORM4")
net.push_linear("FC2", "REL4", 10)
net.push_softmax("OUTPUT", "FC2", 1.0)
"""

net.printstack()

baselr = 0.01
opt = ailib.optimizer_sdg(10, baselr, 0.5, ailib.CostFunction.CROSSENTROPY)

"""
a = np.array([ float(i)/(28.0 * 28.0) for i in range(28 * 28)]).astype("f")
b = np.array([ float(i == 0) for i in range(10)]).astype("f")
print(b)
err = net.optimize(a, b, opt)
print(net.getoutput("OUTPUT"))
print(err)
"""

loss = 0.0
for i in range(100000):
    
    #Update learningrate
    opt.set_learningrate(baselr * (1.0 - float(i) / 100000.0))
    
    #Select random sample and optimize network
    rand_sample = random.randint(0, len(mnist.train.images)-1) 
    loss += net.optimize(mnist.train.images[rand_sample], mnist.train.labels[rand_sample].astype(np.float32), opt)
    #print(net.getoutput("OUTPUT"))
    if (loss != loss): print("Error: {}".format(i))
        

    #Log
    if i % 1000 == 0 and i != 0:
        print("Cicle: {} MediumLoss: {} learningrate: {}".format(i, loss / 1000.0, opt.get_learningrate()))
        loss = 0.0
    
    #Test on validationset to see generalization
    if i % 25000 == 0 and i != 0:
        errors = 0
        for k in range(len(mnist.test.images)):
            net.run(mnist.test.images[k])
            prediction = np.argmax(net.getoutput("OUTPUT"))
            correct_solution = np.argmax(mnist.test.labels[k])
            if (prediction != correct_solution): errors += 1
        print("Testing errors: {}".format(errors))
