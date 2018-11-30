strongly suggest running this job on GPU, which should be more 20 times faster than on CPU
e.g.,


Please cite this paper if you find our code is useful
@article{yuan2018simple,
  title={A Simple but Hard-to-Beat Baseline for Session-based Recommendations},
  author={Yuan, Fajie and Karatzoglou, Alexandros and Arapakis, Ioannis and Jose, Joemon M and He, Xiangnan},
  journal={arXiv preprint arXiv:1808.05163},
  year={2018}
}


1 )you can run nextitrec.py directly, which includes training and testing

2 )you can aslo run nextitrec_generate.py, which is only for predicting/generating. But you need check whether your model 
variables have been saved when you run nextitrec.py. 

3 )MostPop.py is a baseline based on item popularity


Your training data should be some sequences of the same length, if the sequences are not the same length, padding 0 in the beggining, e.g.,
1,44,67,78,1000
0,0,88,98,1
0,88,981,13,17

Your testing data can be any length, but suggest you first using the same length for evalution, once you are familar with the model, you can change your data or slightly change the code to meet your requirements.





## References

- [Tensorflow GatedCNN][1] code
- [Tensorflow PixelCNN][2] code
- [Tensorflow Wavenet][3] code
- [Tensorflow Bytenet][4] code
- [Tensorflow Convolutional Seq2Seq][5] code
- [Sugar Tensor Source Code][6] For implementing some ops





[1]:https://github.com/anantzoid/Language-Modeling-GatedCNN

[2]:https://github.com/openai/pixel-cnn

[3]:https://github.com/ibab/tensorflow-wavenet

[4]:https://github.com/paarthneekhara/byteNet-tensorflow

[5]:https://github.com/tobyyouup/conv_seq2seq

[6]:https://github.com/buriburisuri/sugartensor
