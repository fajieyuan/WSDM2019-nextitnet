strongly suggest running the code on GPU, which is at least 20 times faster than on CPU. 
You can email me if you have any questions about the performance and efficiency of NextItNet.


If you want to use this model in product with extremely large items (e.g.,> 100 million items). You may have memory problem during training and prediction by using nextitrec.py. In this case, you just use our negative sampling setting for training, and recalled items for predicting instead of outputing a huge softmax, as shown in nextitrec__recall.py. Also, using  tf.estimator.Estimator class is much better for real production environments.

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

# Notice


We noticed that there is a Recsys paper "Performance comparison of neural and non-neural approaches to session-based recommendation" arguing that NextItNet yields much worse results than GRU4Rec and other trivial baselines, and run extremely slow during training. We have consulted with the author regarding this issue. The author replied us this is because of the session length problem (see the attached image). 

To be specific, they use a very short session length (by a sliding window) e.g., 10, for GRU4Rec and other baselines, but use 500 (assume 500 is the longest user session) for NextItNet (the authors performed experiments exactly in this way). However, please note that in our paper (see Section 4.1.1) we have splitted the long-range user sessions into several sub-sessions and compare all methods under the same session length for fair comparison. This pre-processing has a very large impact on the final results, regarding both accuracy and training time.
For example, if users have 50 positive interactions in average, the preprocessing method for baselines will create 5 training sequences (ignoring data augmentation, see Section 3.1) for each user while for NextItNet there is only one training sequence per user. In other words, the number of all training examples for baselines are 5 times larger than NextItNet. This is very unfair since in Table 2 we have clearly demonstrated that models with different session length are not comparable at all, and show completely different results. 

In terms of training time, it is even much worse since padding around 450 zeros for each user session will of course make the training of NextItNet extremely slow.  In other words, the evaluation of this paper is unfair and conclusions are misleading. Following our paper with a fair comparison, you definitely will find NextItNet performs better than GRU4Rec/LSTM4Rec and Caser on most datasets. Note we are not sure whether there are other inappropriate experimental settings when comparing non-neural and neural methods.
<p align="center">
    <br>
    <img src="https://github.com/fajieyuan/nextitnet/blob/master/Data/author_reply.png" width="1200"/>
    <br>
<p>

