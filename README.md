PyTorch version https://github.com/mlpotter/NextItNet/blob/main/model.py (not sure whether it is correct or not)


Please cite this paper if you find our code is useful

@inproceedings{yuan2018simple,
  title={A Simple Convolutional Generative Network for Next Item Recommendation },
  author={Yuan, Fajie and Karatzoglou, Alexandros and Arapakis, Ioannis and Jose, Joemon M and He, Xiangnan},
  booktitle={Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  year={2019},
  organization={ACM}
}

## Demo:
1 )you can run nextitnet_topk.py 
   python nextitnet_topk.py 


Your training data should be some sequences of the same length, if the sequences are not the same length, padding 0 in the beggining, e.g.,

1,44,67,78,1000

0,0,88,98,1

0,88,981,13,17

If a user has 1000 interactions but most users have around 100 interactions，then i would suggest you use sequence length of 100 rather than 1000, since it is much slower by padding so many zeros. You can also simply split the longer sequences into many short subsequences.

## Dataset:

Video-6M: https://drive.google.com/file/d/1wd3xzF9VnZ6r35nMb3-H4E31vWK87vjW/view?usp=sharing

We construct a large-scale session-based recommendation dataset (denoted as Video-6M) by collecting the interactiton behaviors of nearly 6 million users in a week  from a commercial recommender system. The dataset can be used to evaluate very deep recommendation models (up to 100 layers), such as NextItNet (as shown in our paper StackRec（SIGIR2021)). 
If you use this dataset in your paper, you should cite our NextItNet and StackRec for publish permission. 


```
@article{wang2020stackrec,
  title={StackRec: Efficient Training of Very Deep Sequential Recommender Models by Iterative Stacking},
  author={Wang, Jiachun and Yuan, Fajie and Chen, Jian and Wu, Qingyao and Li, Chengmin and Yang, Min and Sun, Yang and Zhang, Guoxiao},
  journal={Proceedings of the 44th International ACM SIGIR conference on Research and Development in Information Retrieval},
  year={2021}
}
```

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

To be specific, they use a very short session length (by a sliding window) e.g., 10, for GRU4Rec and other baselines, but use 500 (assume 500 is the longest user session) for NextItNet. However, please note that in our paper (see Section 4.1.1) we have splitted the long-range user sessions into several sub-sessions and compare all methods under the same session length for fair comparison. This pre-processing has a very large impact on the final results, regarding both accuracy and training time.
For example, if users have 50 positive interactions in average, the preprocessing method for baselines will create 5 training sequences (ignoring data augmentation, see Section 3.1) for each user while for NextItNet there is only one training sequence per user. In other words, the number of all training examples for baselines are 5 times larger than NextItNet. This is very unfair since in Table 2 we have clearly demonstrated that models with different session length are not comparable at all, and show completely different results. 

In terms of training time, it is even much worse since padding around 450 zeros for each user session will of course make the training of NextItNet extremely slow.  In other words, the evaluations in this paper are performed on unfair settings and conclusions do not hold. Following our paper with a fair comparison, you definitely will find NextItNet performs better than GRU4Rec/LSTM4Rec and Caser on most datasets. Note we are not sure whether there are other inappropriate experimental settings in their paper when comparing other neural models with non-neural models, which they claimed much better.
<p align="center">
    <br>
    <img src="https://github.com/fajieyuan/nextitnet/blob/master/Data/auth_reply.png" width="1200"/>
    <br>
<p>

# Hiring
If you want to work with Fajie https://fajieyuan.github.io/, Please contact him by email yuanfajie@westlake.edu.cn. His lab is now recruiting visiting students, interns, research assistants, posdocs (Chinese yuan: 450,000-550,000 per year), and research scientists. You can also contact him if you want to pursue a Phd degree at Westlake University. Please feel free to talk to him (by weichat: wuxiangwangyuan) if you have ideas or papers for collaboration. He is open to various collaborations.
 西湖大学原发杰团队长期招聘：推荐系统和生物信息（尤其蛋白质相关）方向 ，科研助理，博士生，博后，访问学者，研究员系列。
