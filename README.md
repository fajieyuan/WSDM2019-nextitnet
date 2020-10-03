strongly suggest running the code on GPU, which is at least 20 times faster than on CPU. 
You can email me if you have any questions about the performance and efficiency of NextItNet.


If you want to use this model in product with extremely large items (e.g.,> 100 million items). You may have memory problem during training and prediction by using nextitrec.py. In this case, you just use our negative sampling setting for training, and recalled items for predicting instead of outputing a huge softmax, as shown in nextitrec__recall.py.

In general, NextItNet is able to handle several millions of items with billions of interaction feedback, and performs much faster and better than LSTM/GRU using the same training setttings.

There are many tricks to solve memory and efficiency problems and you can email me if you donot know how to do it. It is also very easy to add various features by concating (tf.concat) operation before convolution or after convolution.



Please cite this paper if you find our code is useful

@inproceedings{yuan2018simple,
  title={A Simple Convolutional Generative Network for Next Item Recommendation },
  author={Yuan, Fajie and Karatzoglou, Alexandros and Arapakis, Ioannis and Jose, Joemon M and He, Xiangnan},
  booktitle={Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  year={2019},
  organization={ACM}
}


1 )you can run nextitrec.py (python nextitrec.py or nextitnet_topk.py (faster for evaluation)) directly, which includes training and testing

2 )you can also run nextitrec_generate.py, which is only for predicting/generating. But you need make sure whether your model 
variables have been saved when you run nextitrec.py. 

3 )MostPop.py is a baseline based on item popularity


Your training data should be some sequences of the same length, if the sequences are not the same length, padding 0 in the beggining, e.g.,

1,44,67,78,1000

0,0,88,98,1

0,88,981,13,17

Your testing data can be any length, but suggest you first using the same length for evalution, once you are familar with the model, you can change your data or slightly change the code to meet your requirements.


note that the attached dataset is very small, you can use the user-filter-20000items-session5.csv.zip or using the below datasets
Movielencs (http://files.grouplens.org/datasets/movielens/):

Link:  https://drive.google.com/file/d/1zy5CnpxmOgdQBoOMt0-zJUqIBeXMdLKc/view?usp=sharing (it allows at least 20 layers for NextItNet, e.g., using dilation 1,4,1,4,1,4,1,4,1,4,)

download a large sequential dataset of tiktok: https://pan.baidu.com/s/1sdvA-prlW8cBVtFuuiqF3Q  
code (提取码):wncu





## References

- [Tensorflow GatedCNN][1] code
- [Tensorflow PixelCNN][2] code
- [Tensorflow Wavenet][3] code
- [Tensorflow Bytenet][4] code
- [Tensorflow Convolutional Seq2Seq][5] code
- [Sugar Tensor Source Code][6] code
- [Tensorflow Convolutional Neural Networks for Sentence Classification][7] code
- [Tensorflow RNN ][7] code


[1]:https://github.com/anantzoid/Language-Modeling-GatedCNN

[2]:https://github.com/openai/pixel-cnn

[3]:https://github.com/ibab/tensorflow-wavenet

[4]:https://github.com/paarthneekhara/byteNet-tensorflow

[5]:https://github.com/tobyyouup/conv_seq2seq

[6]:https://github.com/buriburisuri/sugartensor

[7]:https://github.com/dennybritz/cnn-text-classification-tf

[8]:https://github.com/tensorflow/models/tree/master/tutorials/rnn

# nextitnet
A Simple Convolutional Generative Network for Next Item Recommendation


We noticed that there is a Recsys paper "Performance comparison of neural and non-neural approaches to session-based recommendation" arguing that NextItNet produces worse results than GRU4Rec and Caser, and run slowly during training. We have consulted with the author regarding this issue. The author replied us this is because of the session length problem. For example, if you use session length (i.e., window size) 20 for Caser and GRU4Rec but using 500 for NextItNet (the authors performed experiments exactly in this way), then there will be 480 additional zeros padded for NextItNet, which is of course very slow during training. Besides, using relatively smaller sessions also means you have more training sequences and thus will get better results if your dataset is not large enough. In other words, the evaluation in this paper is unfair. Please refer to our paper regarding the data preprocessing. With a fair comparison, you definitely will find NextItNet performs better than GRU4Rec and Caser on most datasets。

