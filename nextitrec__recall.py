#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import data_loader_recsys
import generator_recsys_20190302_lastindex as generator_recsys
import utils
import shutil
import time
import math
import eval
import numpy as np
import argparse


import shutil
import os.path
from tensorflow.python.framework import graph_util

# You can run it directly, first training and then evaluating
# nextitrec_generate.py can only be run when the model parameters are saved, i.e.,
#  save_path = saver.save(sess,
#                       "Data/Models/generation_model/model_nextitnet.ckpt".format(iter, numIters))

# In the prediction phrase, only consider the last index to reduce the time compplexity


#Strongly suggest running codes on GPU with more than 10G memory!!!
#if your session data is very long e.g, >50, and you find it may not have very strong internal sequence properties, you can consider generate subsequences

# based on the recalled items
def generatesubsequence(train_set):
    # create subsession only for training
    subseqtrain = []
    for i in range(len(train_set)):
        # print x_train[i]
        seq = train_set[i]
        lenseq = len(seq)
        # session lens=100 shortest subsession=5 realvalue+95 0
        for j in range(lenseq - 2):
            subseqend = seq[:len(seq) - j]
            subseqbeg = [0] * j
            subseq = np.append(subseqbeg, subseqend)
            # beginseq=padzero+subseq
            # newsubseq=pad+subseq
            subseqtrain.append(subseq)
    x_train = np.array(subseqtrain)  # list to ndarray
    del subseqtrain
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_train = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_train]
    print "generating subsessions is done!"
    return x_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions, used for generating')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    #history_sequences_20181014_fajie_smalltest.csv    /data/weishi_ai_ceph/fajieyuan/nextitnet-master-2/Data/Session/history_sequences_20181014_fajie.csv
    # /data/weishi_ai_ceph/fajieyuan/nextitnet-master-2/Data/Session/history_sequences_20181014_fajie.index
    # /data/weishi_ai_ceph/fajieyuan/nextitnet-master-2/Data/Models/generation_model/model_nextitnet.pb
    parser.add_argument('--datapath', type=str, default='Data/Session/user-filter-20000items-session5_index.csv',
                        help='data path')
    parser.add_argument('--datapath_index', type=str, default='Data/Session/',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=2000,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=2000,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.5,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--is_generatesubsession', type=bool, default=False,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
    args = parser.parse_args()



    dl = data_loader_recsys.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath, 'dir_name_index': args.datapath_index})
    all_samples = dl.item
    items = dl.item_dict
    all_items=items.values()
    print "len(items)",len(items)
    # print all_items


    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]


    # Split train/test set
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    if args.is_generatesubsession:
        train_set = generatesubsequence(train_set)

    model_para = {
        #if you changed the parameters here, also do not forget to change paramters in nextitrec_generate.py
        'item_size': len(items),
        'dilated_channels': 100,#200 is usually better
        # if you use nextitnet_residual_block, you can use [1, 4, ],
        # if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
        # when you change it do not forget to change it in nextitrec_generate.py
        # if you find removing residual network, the performance does not obviously decrease, then I think your data does not have strong seqeunce. Change a dataset and try again.
        'dilations': [1, 2,],
        'kernel_size': 3,
        'learning_rate':0.001,
        'batch_size':32,#128 is usually better
        'iterations':100,
        'top_k': args.top_k,
        'is_negsample':True #False denotes no negative sampling. You have to use True if you want to do it based on recalled items
    }

    itemrec = generator_recsys.NextItNet_Decoder(model_para)
    itemrec.train_graph(model_para['is_negsample'])
    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1).minimize(itemrec.loss)
    itemrec.predict_graph_onrecall(model_para['is_negsample'],reuse=True)

    sess= tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()




    numIters = 1
    for iter in range(model_para['iterations']):
        batch_no = 0
        batch_size = model_para['batch_size']
        while (batch_no + 1) * batch_size < train_set.shape[0]:

            start = time.clock()

            item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]
            _, loss, results = sess.run(
                [optimizer, itemrec.loss,
                 itemrec.arg_max_prediction],
                feed_dict={
                    itemrec.itemseq_input: item_batch
                })
            end = time.clock()
            if numIters % args.eval_iter == 0:
                print "-------------------------------------------------------train1"
                print "LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, iter, batch_no, numIters, train_set.shape[0] / batch_size)
                print "TIME FOR BATCH", end - start
                print "TIME FOR ITER (mins)", (end - start) * (train_set.shape[0] / batch_size) / 60.0

            # if numIters % args.eval_iter == 0:
            #     print "-------------------------------------------------------test1"
            #     if (batch_no + 1) * batch_size < valid_set.shape[0]:
            #         item_batch = valid_set[(batch_no) * batch_size: (batch_no + 1) * batch_size, :]
            #     loss = sess.run(
            #         [itemrec.loss_test],
            #         feed_dict={
            #             itemrec.input_predict: item_batch
            #         })
            #     print "LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
            #         loss, iter, batch_no, numIters, valid_set.shape[0] / batch_size)

            batch_no += 1


            if numIters % args.eval_iter == 0:
                batch_no_test = 0
                batch_size_test = batch_size*1
                curr_preds_5=[]
                rec_preds_5=[] #1
                ndcg_preds_5=[] #1
                curr_preds_20 = []
                rec_preds_20 = []  # 1
                ndcg_preds_20  = []  # 1

                while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
                    if (numIters / (args.eval_iter) < 10):
                        if (batch_no_test > 20):
                            break
                    else:
                        if (batch_no_test > 500):
                            break
                    item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                    allitem_list=[]
                    # allitem_batch=[allitem_list[all_items] for i in xrange(batch_size_test)]
                    for i in range(batch_size_test):
                        allitem_list.append(all_items)


                    [top_k] = sess.run(
                        [itemrec.top_k],
                        feed_dict={
                            itemrec.input_predict: item_batch,
                            itemrec.input_recall: allitem_list
                        })
                    batch_top_n=[]

                    for bi in range(len(allitem_list)):
                        recall_batch_num=allitem_list[bi]
                        top_n = [recall_batch_num[x] for x in top_k[1][bi]]
                        batch_top_n.append(top_n)

                    for bi in range(top_k[1].shape[0]):
                        top_n=batch_top_n[bi]
                        true_item = item_batch[bi][-1]

                        top_n={ch: i for i, ch in enumerate(top_n)}
                        rank_n = top_n.get(true_item)
                        if rank_n == None:
                            curr_preds_5.append(0.0)
                            rec_preds_5.append(0.0)  # 2
                            ndcg_preds_5.append(0.0)  # 2
                        else:
                            MRR_5 = 1.0 / (rank_n + 1)
                            Rec_5 = 1.0  # 3
                            ndcg_5 = 1.0 / math.log(rank_n + 2, 2)  # 3
                            curr_preds_5.append(MRR_5)
                            rec_preds_5.append(Rec_5)  # 4
                            ndcg_preds_5.append(ndcg_5)

                    batch_no_test += 1
                    print "BATCH_NO: {}".format(batch_no_test)
                    print "Accuracy mrr_5:",sum(curr_preds_5) / float(len(curr_preds_5))#5
                    # print "Accuracy mrr_20:", sum(curr_preds_20) / float(len(curr_preds_20))  # 5
                    print "Accuracy hit_5:", sum(rec_preds_5) / float(len(rec_preds_5))#5
                    # print "Accuracy hit_20:", sum(rec_preds_20) / float(len(rec_preds_20))  # 5
                    print "Accuracy ndcg_5:", sum(ndcg_preds_5) / float(len(ndcg_preds_5))  # 5
                    # print "Accuracy ndcg_20:", sum(ndcg_preds_20) / float(len(ndcg_preds_20))  #
                    #print "curr_preds",curr_preds
                # print "---------------------------Test Accuray----------------------------"
            numIters += 1
            if numIters % args.save_para_every == 0:

                save_path = saver.save(sess,
                                       "Data/Models/generation_model/model_nextitrec_20190302_lastindex.ckpt".format(
                                           iter, numIters))


                # save_path = saver.save(sess,
                #                        "/data/weishi_ai_ceph/fajieyuan/nextitnet-master-2/Data/Models/generation_model/model_nextitnet.ckpt".format(iter, numIters))

                # print("%d ops in the final graph." % len(tf.get_default_graph().as_graph_def().node))  # 得到当前图有几个操作节点
                # for op in tf.get_default_graph().get_operations():  # 打印模型节点信息
                #     print (op.name, op.values())



                graph_def = tf.get_default_graph().as_graph_def()  # 得到当前的图的 GraphDef 部分，通过这个部分就可以完成重输入层到输出层的计算过程
                output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
                    sess,
                    graph_def,
                    ["input_predict","top-k"]  # 需要保存节点的名字
                )

                with tf.gfile.GFile("Data/Models/generation_model/model_nextitrec_20190302_lastindex.pb", "wb") as f:  # 保存模型
                    f.write(output_graph_def.SerializeToString())  # 序列化输出

                print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    main()
