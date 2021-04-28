import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

"""
Classifying hand-written numbers.
Each input is 8x8 pixel (provided by sklearn), and grey-scale value raning in [0,16]
img is sliced into rows of pixels, which are sequentially placed in as inputs
"""

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain,self).__init__()
        with self.init_scope():
            self.l1=L.Linear(64,100)
            self.l2=L.Linear(100,100)
            self.l3=L.Linear(100,10)
    def __call__(self,x):
        h1=F.relu(self.l1(x))
        h2=F.relu(self.l2(h1))
        y=self.l3(h2)
        return y

epoch=20
batchsize=100

#data generation
digits=load_digits()
data_train,data_test,label_train,label_test=train_test_split(digits.data,digits.target,test_size=0.2)
data_train=(data_train).astype(np.float32)
data_test=(data_test).astype(np.float32)
train=chainer.datasets.TupleDataset(data_train,label_train)
test=chainer.datasets.TupleDataset(data_test,label_test)

"""below is identical to OR.py"""

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

trainer.extend(extensions.dump_graph("main/loss")) #structure of NN as dot file
trainer.extend(extensions.PlotReport(["main/loss","validation/main/loss"],"epoch",file_name="loss.png")) #error/epoch graph as png
trainer.extend(extensions.PlotReport(["main/accuracy","validation/main/accuracy"],"epoch",file_name="accuracy.png")) #accuracy/epoch graph as png
#trainer.extend(extensions.snapshot(),trigger=(100,"epoch")) #snapshort output power to resume learning
#chainer.serializers.load_npz("result/snapshot_iter_100",trainer) #for restart
#chainer.serializers.save_npz("result/out_model",model) #save/output current model

#start learning
trainer.run()