## lstmDL
The third project in the [Deep Learning Foundations Nanodegree program](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101), designed to introduce the concepts of Recurrent Neural Networks (and Layers!):

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

The goal here was to **generate Simpsons scripts using a dataset of scripts**, and the details of our methodology is available in [`tv_script_generation.ipynb`](https://github.com/Vvkmnn/introDL/blob/master/image_classification.ipynb).

### Setup

This project requires Python 3 (Probably as distributed by [Anaconda](https://www.continuum.io/downloads)) and [Tensorflow](https://www.tensorflow.org/):

```python
appnope==0.1.0
autopep8==1.3.2
backports.weakref==1.0rc1
bleach==1.5.0
decorator==4.1.2
html5lib==0.9999999
ipykernel==4.6.1
ipython==6.1.0
ipython-genutils==0.2.0
jedi==0.10.2
jupyter-client==5.1.0
jupyter-core==4.3.0
Markdown==2.6.8
numpy==1.13.1
pexpect==4.2.1
pickleshare==0.7.4
prompt-toolkit==1.0.15
protobuf==3.3.0
ptyprocess==0.5.2
pycodestyle==2.3.1
Pygments==2.2.0
python-dateutil==2.6.1
pyzmq==16.0.2
simplegeneric==0.8.1
six==1.10.0
tensorflow==1.0.0
tornado==4.5.1
traitlets==4.3.2
wcwidth==0.1.7
Werkzeug==0.12.2
```


### Results

In this project, we implemented the steps necessary to build a simple text-generation, by implementing a **Multi-Cell RNN** trained on batches of script sequences. The **Layers** involved include:
* **Word Encoding**: Conversion of a script corpus into a dictionary of words, such that we can create vector encodings of string sequences. 
* **Input Layer**: Data placeholders for input and target sequences, hyperparameters like the learning rate (etc.)
* **RNN Layers**: A series (256) of RNN cells lined up and zeroed so that they can be trained to yield weight matrices (with fully-connected/linear activation functions between every cell)
* **Word Embedding**: Embed these new string tensors into our proposed RNN structure, by embedding a vocabulary of words as a train/target pair across our 256 cells.
* **Word Batching**: Our script corpus is definitely longer than 27; let's slice each train/target pair such that we are iteratively going through to the end.
* **Training**: Plug in some valid hyperparameters, and start training the graph. Pipe through a GPU or a CPU until loss is minimized (`train_loss = 0.6101` was our best.)
* **Generation**: Using the handy `get_tensor_by_name()` from [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name), lets pull out our weight matrices. 

The **final model** yielded a final **Training Loss of 0.61** using *100 epochs*, a *batch size of 128*, *256 rnn cells* and a *learning rate of 0.01%*. This yielded the following completely comptuer generated script (!!):

```text
moe_szyslak: heck no-- that's true. that's the man who is from the forget-me-shot.
moe_szyslak:(too voice) oh yeah, uh... what all you know who i made that?
bart_simpson: yeah? look, we gotta celebrate! throw a guy who fixes things?
homer_simpson: wait on!
homer_simpson: lisa's mad at me and marge is mad.(really does you,(flips page) from no, i have kids?
voice: excuse me, the plaster's flaking again!
crowd:(chanting) barney! i believe my wife was madonna.
krusty_the_clown: a chance of backgammon.
bart_simpson:(singing) hello...
dr. _zander: with you poor soul thing i see. health in the car.
moe_szyslak: better set the morning? how 'bout lenny.(stands up coaster) this coaster's fine.
moe_szyslak: just ask anyone in this bar.
moe_szyslak: power off, einstein.
moe_szyslak: yeah! ooh, you've been taking ventriloquism lessons.(nervous laugh)
lenny_leonard:(singing,
```

You'll notice that even at this level, the script could be more logical. This could be achieved with an even larger training site, better tuned hyperparameters, and more computational power. In my opinion, this experiment still proves the validity of using this methodology of artifically-generated text sequence generation; it just needs to optimized before production use.  