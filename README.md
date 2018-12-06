
strongly suggest running the code on GPU, which is at least 20 times faster than on CPU



Please cite this paper if you find our code is useful

@inproceedings{yuan2018simple,
  title={A Simple Convolutional Generative Network for Next Item Recommendation },
  author={Yuan, Fajie and Karatzoglou, Alexandros and Arapakis, Ioannis and Jose, Joemon M and He, Xiangnan},
  booktitle={Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  year={2019},
  organization={ACM}
}


1 )you can run nextitrec.py (python nextitrec.py) directly, which includes training and testing

2 )you can also run nextitrec_generate.py, which is only for predicting/generating. But you need check whether your model 
variables have been saved when you run nextitrec.py. 

3 )MostPop.py is a baseline based on item popularity


Your training data should be some sequences of the same length, if the sequences are not the same length, padding 0 in the beggining, e.g.,

1,44,67,78,1000

0,0,88,98,1

0,88,981,13,17

Your testing data can be any length, but suggest you first using the same length for evalution, once you are familar with the model, you can change your data or slightly change the code to meet your requirements.


note that the attached dataset is very small, you can use the user-filter-20000items-session5.csv.zip 
We are trying to release a sequential dataset, which has very good sequence properties.




## References

- [Tensorflow GatedCNN][1] code
- [Tensorflow PixelCNN][2] code
- [Tensorflow Wavenet][3] code
- [Tensorflow Bytenet][4] code
- [Tensorflow Convolutional Seq2Seq][5] code
- [Sugar Tensor Source Code][6] code
- [Tensorflow RNN][7] code





[1]:https://github.com/anantzoid/Language-Modeling-GatedCNN

[2]:https://github.com/openai/pixel-cnn

[3]:https://github.com/ibab/tensorflow-wavenet

[4]:https://github.com/paarthneekhara/byteNet-tensorflow

[5]:https://github.com/tobyyouup/conv_seq2seq

[6]:https://github.com/buriburisuri/sugartensor

[7]:https://github.com/dennybritz/cnn-text-classification-tf

[7]:https://github.com/tensorflow/models/tree/master/tutorials/rnn

# nextitnet
A Simple Convolutional Generative Network for Next Item Recommendation

