import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

"""
Expresses Links between some nodes in network, and defines which activation function to use
"""

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain,self).__init__()
        with self.init_scope():
            self.l1=L.Linear(2,3) # input layer node=3, middle layer node=2
            self.l2=L.Linear(3,2) # middle layer=3, output layer=2
    def __call__(self,x):
        h1=F.relu(self.l1(x)) #Calculate output of l1 with ReLU and weight of link l1
        y=self.l2(h1) #calculate output with weight of link l2
        return y

epoch=10
batchsize=4 #minibatch learning method

#creating data
trainx=np.array(([0,0],[0,1],[1,0],[1,1]),dtype=np.float32)
trainy=np.array([0,1,1,1],dtype=np.int32)
train=chainer.datasets.TupleDataset(trainx,trainy)
test=chainer.datasets.TupleDataset(trainx,trainy)

#admistrating neural network
model=L.Classifier(MyChain(),lossfun=F.softmax_cross_entropy)
#chainer.serializers.load_npz("result/out_model",model)
#choosing optimizing function. (calculating error in trained data output)
optimizer=chainer.optimizers.Adam()
optimizer.setup(model)

#Defining iterators (generating learn and test data)
train_iter=chainer.iterators.SerialIterator(train,batchsize) #learning
test_iter=chainer.iterators.SerialIterator(test,batchsize,repeat=False,shuffle=False) #evaluating

#administrating updater (Tying together train data and optimizer)
updater=training.StandardUpdater(train_iter,optimizer)

#administrating trainer (Making an environment for the learning, and setting it's epoch)
trainer=training.Trainer(updater,(epoch,"epoch"))

#Display and save of Learning situation
trainer.extend(extensions.LogReport()) #log
trainer.extend(extensions.Evaluator(test_iter,model)) #cur epoch
trainer.extend(extensions.PrintReport(["epoch","main/loss","validation/main/loss","main/accuracy","validation/main/accuracy","elapsed_time"])) #cur calculations status

trainer.extend(extensions.dump_graph("main/loss")) #structure of NN
trainer.extend(extensions.PlotReport(["main/loss","validation/main/loss"],"epoch",file_name="loss.png")) #error graph
trainer.extend(extensions.PlotReport(["main/accuracy","validation/main/accuracy"],"epoch",file_name="accuracy.png")) #accuracy graph
#trainer.extend(extensions.snapshot(),trigger=(100,"epoch")) #snapshort output power to resume learning
#chainer.serializers.load_npz("result/snapshot_iter_100",trainer) #for restart
#chainer.serializers.save_npz("result/out_model",model)

#start learning
trainer.run()


"""
Results in accuraccy of 1 eventually
"""