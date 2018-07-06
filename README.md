# Sentiment Analysis With Deep Learning

#**Command Line**

python3 sentiment.py positive.txt negative.txt vectors.txt 75


The purpose of this project is primarily to work in the TensorFlow library and gain experience with the library.   

**What is Tensorflow ?**

TensorFlow™ is an open source software library for high performance numerical computation. Its flexible architecture allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices. Originally developed by researchers and engineers from the Google Brain team within Google’s AI organization, it comes with strong support for machine learning and deep learning and the flexible numerical computation core is used across many other scientific domains.

**Creating Data Structure**

There are 3 different text files to work on. These files contain positive sentences, negative sentences, and vectors.
75% of the positive and negative sentences constitute the training spindle. The remaining 25% constitutes our test set. The list is used as the data structure.
Before the constructions are formed, the two components are mixed homogeneously positive and negative phrases.
Positive censuses [1,0] and negative censuses are marked with [0,1].

The vectors are in the form shown below.

**duymak:0.00039987528  ...   0.1955209**

The dictionary structure is used to store the vectors, the part before  ':' is key, and the part after ':' is value.
At the same time, to create Lexicon, a new list structure was created with the values before ':'.

#**Creating Neural Network**


**What is Neural Network**    

Neural Network is a structure established as layers. The first layer is called the input, the last layer is called the output. Layers in the middle are called 'Hidden Layers'. Each layer contains a certain number of 'Neurons'. These neurons are linked to each other by 'Synapse'. Synapselar contains a coefficient.
The value of a neuron is the result of multiplying the inputs of those neurons by the coefficients. This result is incorporated into an activation function. According to the outcome of the function, it is decided that the neuron will not be ignited and ignited.

**Layers**   
Two hidden layers and one output layer were created when creating the model. Each hidden layer contains 100 neurons.

**Batch Size**   
Total number of training examples present in a single batch.
Batch size is defined as 10.

**Epoch**   
One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
Epoch is defined as 10.

**Learning Rate**   
Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient.
learning rate is defined as 0.001.

**Activation function**   
The activation ops provide different types of nonlinearities for use in neural networks. These include smooth nonlinearities (sigmoid, tanh, elu, selu, softplus, and softsign), continuous but not everywhere differentiable functions (relu, relu6, crelu and relu_x), and random regularization (dropout).   
All activation ops apply componentwise, and produce a tensor of the same shape as the input tensor.   
ReLU activation is preferred for use in this assignment.   
The ReLU is the most used activation function in the world right now.Since, it is used in almost all the convolutional neural networks or deep learning.
The function and its derivative both are monotonic.

**Optimezers**
The Optimizer base class provides methods to compute gradients for a loss and apply gradients to variables. A collection of subclasses implement classic optimization algorithms such as GradientDescent and Adagrad.   
You never instantiate the Optimizer class itself, but instead instantiate one of the subclasses.   
Gradient Descent optimezer method is preferred for use in this assignment.

**Accurracy** 

When the above parameters are used, the program output is as follows.

```
Epoch 1 completed out of 10 loss: 1760.3988037109375
Epoch 2 completed out of 10 loss: 1216.342875957489
Epoch 3 completed out of 10 loss: 780.2468023300171
Epoch 4 completed out of 10 loss: 653.8694211430848
Epoch 5 completed out of 10 loss: 464.1355482339518
Epoch 6 completed out of 10 loss: 367.6751266717911
Epoch 7 completed out of 10 loss: 210.90972113609314
Epoch 8 completed out of 10 loss: 172.84423232078552
Epoch 9 completed out of 10 loss: 133.68374681472778
Epoch 10 completed out of 10 loss: 118.21700771152973
Accuracy: 0.5135135
````

#**Studies to Improve Accuracy**

Several studies can be done to raise Accuracy. Some examples will be given from these studies.
In all studies, the dataset is fixed, 75% of the data and 25% of the training are used for the test.
It should be known that the data set studied is so small that the results may not always be consistent!


**1-)Learning Rate Increase**

If the Learning Rate is 0.01 instead of 0.001, the program output is as follows (Other parameter values are in the initial state)


```
Epoch 1 completed out of 10 loss: 3931.6829872131348
Epoch 2 completed out of 10 loss: 660.2926139831543
Epoch 3 completed out of 10 loss: 140.66496564372792
Epoch 4 completed out of 10 loss: 22.761163228936493
Epoch 5 completed out of 10 loss: 3.9362246155738774
Epoch 6 completed out of 10 loss: 0.0025798297002861403
Epoch 7 completed out of 10 loss: 0.00021605361425258707
Epoch 8 completed out of 10 loss: 0.00015600405308191512
Epoch 9 completed out of 10 loss: 0.00012468664371567684
Epoch 10 completed out of 10 loss: 0.00010501051892219948
Accuracy: 0.6756757
```

I am very impressed by the fact that I keep the learning speed high. The high speed of learning will cause the swing. If it is small, but it will cause it to take a long time because it will progress in small steps.
From the output it can be seen that the accuracy value increases as the learning rate is increased.

**2-) Increasing Number of Hidden Layer Neurons**

The number of neurons in the hidden layers that were originally 100 was changed to 500, and the program output was as follows (other parameter values are in the initial state)

``````
Epoch 1 completed out of 10 loss: 13076.121643066406
Epoch 2 completed out of 10 loss: 6691.089462280273
Epoch 3 completed out of 10 loss: 3720.3374938964844
Epoch 4 completed out of 10 loss: 6959.547369003296
Epoch 5 completed out of 10 loss: 2114.202033996582
Epoch 6 completed out of 10 loss: 439.5688076019287
Epoch 7 completed out of 10 loss: 175.9564208984375
Epoch 8 completed out of 10 loss: 8.839767217636108
Epoch 9 completed out of 10 loss: 0.0
Epoch 10 completed out of 10 loss: 0.0
Accuracy: 0.7567568
``````

As you can see, the increase in the number of neurons increases learning. The number of neurons indicates the number of information retained in the memory, thus increasing the memory requirement and increasing the computation time as the number of neurons increases. This should be taken care of if you are working in a non-GPU environment.
A small number of neurons causes underfitting. If we want to achieve an ideal performance as learning, memory, and computation, the number of neurons can be reduced as layers progress, provided there is more in the first layer.
As can be seen, the more the number of neurons is increased, the higher the accuracy.

**3-)Increasing Epoch**

At the beginning, the epoch value of 10 is raised to 20 and the program output is as follows (Other parameter values are in the initial state)

 ```
Epoch 1 completed out of 20 loss: 1398.9751472473145
Epoch 2 completed out of 20 loss: 1256.160596549511
Epoch 3 completed out of 20 loss: 678.5792555809021
Epoch 4 completed out of 20 loss: 491.5374345779419
Epoch 5 completed out of 20 loss: 413.6645915135741
Epoch 6 completed out of 20 loss: 332.42966196760534
Epoch 7 completed out of 20 loss: 267.7186470031738
Epoch 8 completed out of 20 loss: 207.65561473369598
Epoch 9 completed out of 20 loss: 170.9161548325792
Epoch 10 completed out of 20 loss: 245.12543314695358
Epoch 11 completed out of 20 loss: 121.83309148717672
Epoch 12 completed out of 20 loss: 66.78864712053519
Epoch 13 completed out of 20 loss: 51.31663509715979
Epoch 14 completed out of 20 loss: 41.07846698231242
Epoch 15 completed out of 20 loss: 29.140323545303545
Epoch 16 completed out of 20 loss: 27.273552540368108
Epoch 17 completed out of 20 loss: 23.97162863191261
Epoch 18 completed out of 20 loss: 18.554949614961515
Epoch 19 completed out of 20 loss: 15.593880893226014
Epoch 20 completed out of 20 loss: 14.288819721092295
Accuracy: 0.6486486
 ```
 
As the number of epoch increases, success will increase as the number of epochs increases. However, after a certain step, our model's learning situation will be greatly reduced.
Although we are working on small data, it is often a long time to train models that are working with higher data sizes; There are days of training, months of service. This is what happens when you learn deeply. For this reason, it is tried to shorten the training process as much as possible with other hyper parameters.
As you can see, increasing the number of epochs increases the accuracy.
 
 **4-) Reducing Batch Size**
 
Initially the batch size value of 10 was changed to 5 and the program output was as follows (other parameter values are in the initial state)
 
 ```
Epoch 1 completed out of 10 loss: 2339.421614241562
Epoch 2 completed out of 10 loss: 959.4171237945557
Epoch 3 completed out of 10 loss: 397.78268472175114
Epoch 4 completed out of 10 loss: 209.77629041671753
Epoch 5 completed out of 10 loss: 156.46796821395438
Epoch 6 completed out of 10 loss: 235.91201602220536
Epoch 7 completed out of 10 loss: 85.01098102381002
Epoch 8 completed out of 10 loss: 40.099752407963365
Epoch 9 completed out of 10 loss: 24.506239606434974
Epoch 10 completed out of 10 loss: 12.283557556083625
Accuracy: 0.6216216
```

The small size of batch creates a 'reguralization' effect. When modeled data is given in large groups, memorization is more.
As can be seen, decreasing the batch size value increases the accuracy value.


**5-) Modification of Various Hyperparameters**

learning-rate = 0.01   
hidden layer neuron sizes = 500
(Other parameter values are in the initial state.)

```
Epoch 1 completed out of 10 loss: 39054.108627319336
Epoch 2 completed out of 10 loss: 4221.537200927734
Epoch 3 completed out of 10 loss: 1185.9473781585693
Epoch 4 completed out of 10 loss: 616.0330344438553
Epoch 5 completed out of 10 loss: 115.26939392089844
Epoch 6 completed out of 10 loss: 0.0
Epoch 7 completed out of 10 loss: 0.0
Epoch 8 completed out of 10 loss: 0.0
Epoch 9 completed out of 10 loss: 0.0
Epoch 10 completed out of 10 loss: 0.0
Accuracy: 0.7297297
```

learning-rate = 0.001  
hidden layer neuron sizes = 100
batch size =1  
epochs = 10

```
Epoch 1 completed out of 10 loss: 21058.632644620906
Epoch 2 completed out of 10 loss: 4237.221376246773
Epoch 3 completed out of 10 loss: 1866.8031574487684
Epoch 4 completed out of 10 loss: 1353.6395415184088
Epoch 5 completed out of 10 loss: 703.4235920198262
Epoch 6 completed out of 10 loss: 356.8508113666903
Epoch 7 completed out of 10 loss: 195.97468948364258
Epoch 8 completed out of 10 loss: 0.0
Epoch 9 completed out of 10 loss: 0.0
Epoch 10 completed out of 10 loss: 0.0
Accuracy: 0.8108108
```

