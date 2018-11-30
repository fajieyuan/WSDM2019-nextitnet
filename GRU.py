import tensorflow as tf
import numpy as np
import argparse
import data_loader_recsys as data_loader
import utils
import shutil
import time
import os
import sys
import math
from text_cnn_hv import TextCNN_hv
from rnn import PTBModel

'''
reimplementation of
Session-based Recommendations with Recurrent Neural Networks
screen print has been changed a bit so that to print the output not that ofen
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Learning Rate')
    parser.add_argument('--sample_every', type=int, default=2000,
                        help='Sample generator output evry x steps')
    parser.add_argument('--summary_every', type=int, default=50,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_model_every', type=int, default=1500,
                        help='Save model every')
    parser.add_argument('--sample_size', type=int, default=300,
                        help='Sampled output size')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='Max Epochs')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Momentum for Adam Update')
    parser.add_argument('--resume_model', type=str, default=None,
                        help='Pre-Trained Model Path, to resume from')
    # parser.add_argument('--text_dir', type=str, default='Data/generator_training_data',
    #                     help='Directory containing text files')
    parser.add_argument('--text_dir', type=str, default='Data/Session/user-filter-200000items-session10.csv-map-5to100.csv',
                        help='Directory containing text files')
    parser.add_argument('--data_dir', type=str, default='Data',
                        help='Data Directory')
    parser.add_argument('--seed', type=str, default='f78c95a8-9256-4757-9a9f-213df5c6854e,1151b040-8022-4965-96d2-8a4605ce456c',
                        help='Seed for text generation')
    parser.add_argument('--sample_percentage', type=float, default=0.5,
                        help='sample_percentage from whole data, e.g.0.2= 80% training 20% testing')


    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--l2_reg_lambda', type=float, default=0,
                        help='L2 regularization lambda (default: 0.0)')

    parser.add_argument("--allow_soft_placement", default=True, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False, help="Log placement of ops on devices")

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout keep probability (default: 0.5)')


    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='embedding size')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='num_layers')

    parser.add_argument('--rnn_model', type=str,
                        default='gru',
                        help='gru, listm')

    args = parser.parse_args()



    dl = data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.text_dir})
    # text_samples=16390600  vocab=947255  session100
    all_samples = dl.item
    items = dl.item_dict
    # dl = data_loader.Data_Loader({'model_type' : 'generator', 'dir_name' : args.text_dir})
    # text_samples, vocab = dl.load_generator_data(config['sample_size'])

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    text_samples = all_samples[shuffle_indices]


    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(args.sample_percentage * float(len(text_samples)))
    x_train, x_dev = text_samples[:dev_sample_index], text_samples[dev_sample_index:]

    print "shape", x_train.shape[1]

    # create subsession only for training
    subseqtrain = []
    for i in range(len(x_train)):
        # print x_train[i]
        seq = x_train[i]
        lenseq = len(seq)
        # session lens=100 shortest subsession=5 realvalue+95 0
        for j in range(lenseq - 4):
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
    print "shape", x_train.shape[0]
    print "dataset", args.text_dir



    rnn=PTBModel(args,num_steps=x_train.shape[1], vocab_size=len(items))

    session_conf = tf.ConfigProto(
        # allow to distribute device automatically if your assigned device is not found
        allow_soft_placement=args.allow_soft_placement,
        # whether print or not
        log_device_placement=args.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Define Training procedure
        # global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimizer = tf.train.AdamOptimizer(1e-3)
        # grads_and_vars = optimizer.compute_gradients(rnn.cost)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        sess.run(tf.global_variables_initializer())


    step = 1
    for epoch in range(args.max_epochs):
        batch_no = 0
        batch_size = args.batch_size
        while (batch_no + 1) * batch_size < x_train.shape[0]:

            start = time.clock()
            # do not need to evaluate all, only after several 10 sample_every, then output final results


            text_batch = x_train[batch_no * batch_size: (batch_no + 1) * batch_size, :]

            _, loss = sess.run(
                [rnn.optim, rnn.loss],
                feed_dict={
                    rnn.wholesession: text_batch,
                    rnn.dropout_keep_prob: args.dropout
                })
            end = time.clock()
            if step % args.sample_every == 0:
                print "-------------------------------------------------------train1"
                print "LOSS: {}\tEPOCH: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, epoch, batch_no, step, x_train.shape[0] / args.batch_size)
                print "TIME FOR BATCH", end - start
                print "TIME FOR EPOCH (mins)", (end - start) * (x_train.shape[0] / args.batch_size) / 60.0


            if step % args.sample_every == 0:
                print "-------------------------------------------------------test1"
                if (batch_no + 1) * batch_size < x_dev.shape[0]:
                    text_batch = x_dev[(batch_no) * batch_size: (batch_no + 1) * batch_size, :]
                loss = sess.run(
                    [rnn.loss],
                    feed_dict={
                        rnn.wholesession: text_batch,
                        rnn.dropout_keep_prob: 1.0
                    })
                print "LOSS: {}\tEPOCH: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, epoch, batch_no, step, x_dev.shape[0] / args.batch_size)
            batch_no += 1

            if step % args.sample_every == 0:
                print "********************************************************accuracy"
                batch_no_test = 0
                batch_size_test = batch_size*2
                curr_preds_5 = []
                rec_preds_5 = []  # 1
                ndcg_preds_5 = []  # 1
                curr_preds_20 = []
                rec_preds_20 = []  # 1
                ndcg_preds_20 = []  # 1

                while (batch_no_test + 1) * batch_size_test < x_dev.shape[0]:
                    if (step / (args.sample_every) < 10):
                        if (batch_no_test > 2):
                            break
                    else:
                        if (batch_no_test > 500):
                            break
                    text_batch = x_dev[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                    [probs] = sess.run(
                        [rnn.probs_flat],
                        feed_dict={
                            rnn.wholesession: text_batch,
                            rnn.dropout_keep_prob: 1.0
                        })
                    for bi in range(probs.shape[0]):
                        pred_words_5 = utils.sample_top_k(probs[bi], top_k=args.top_k)  # top_k=5
                        pred_words_20 = utils.sample_top_k(probs[bi], top_k=args.top_k + 15)

                        true_word = text_batch[bi][-1]
                        predictmap_5 = {ch: i for i, ch in enumerate(pred_words_5)}
                        pred_words_20 = {ch: i for i, ch in enumerate(pred_words_20)}

                        rank_5 = predictmap_5.get(true_word)
                        rank_20 = pred_words_20.get(true_word)
                        if rank_5 == None:
                            curr_preds_5.append(0.0)
                            rec_preds_5.append(0.0)  # 2
                            ndcg_preds_5.append(0.0)  # 2
                        else:
                            MRR_5 = 1.0 / (rank_5 + 1)
                            Rec_5 = 1.0  # 3
                            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
                            curr_preds_5.append(MRR_5)
                            rec_preds_5.append(Rec_5)  # 4
                            ndcg_preds_5.append(ndcg_5)  # 4
                        if rank_20 == None:
                            curr_preds_20.append(0.0)
                            rec_preds_20.append(0.0)  # 2
                            ndcg_preds_20.append(0.0)  # 2
                        else:
                            MRR_20 = 1.0 / (rank_20 + 1)
                            Rec_20 = 1.0  # 3
                            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
                            curr_preds_20.append(MRR_20)
                            rec_preds_20.append(Rec_20)  # 4
                            ndcg_preds_20.append(ndcg_20)  # 4

                    batch_no_test += 1
                    print "BATCH_NO: {}".format(batch_no_test)
                    print "Accuracy mrr_5:", sum(curr_preds_5) / float(len(curr_preds_5))  # 5
                    print "Accuracy mrr_20:", sum(curr_preds_20) / float(len(curr_preds_20))  # 5
                    print "Accuracy hit_5:", sum(rec_preds_5) / float(len(rec_preds_5))  # 5
                    print "Accuracy hit_20:", sum(rec_preds_20) / float(len(rec_preds_20))  # 5
                    print "Accuracy ndcg_5:", sum(ndcg_preds_5) / float(len(ndcg_preds_5))  # 5
                    print "Accuracy ndcg_20:", sum(ndcg_preds_20) / float(len(ndcg_preds_20))  #

            step += 1



if __name__ == '__main__':
    main()
