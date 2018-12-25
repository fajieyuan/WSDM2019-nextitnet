import tensorflow as tf
import data_loader_recsys
import generator_recsys
import utils
import shutil
import time
import math
import eval
import numpy as np
import argparse

#check whether the files exists or not,   "Data/Models/generation_model/model_nextitnet.ckpt"
#  if yes run this file directly, if not run nextitrec.py first, which is the training file.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')

    parser.add_argument('--datapath', type=str, default='Data/Session/user-filter-20000items-session5.csv',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=10,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=10,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.5,
                        help='default=0.2 means 80% training 20% testing')
    parser.add_argument('--is_generatesubsession', type=bool, default=False,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
    args = parser.parse_args()



    dl = data_loader_recsys.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath})
    all_samples = dl.item
    items = dl.item_dict
    print "len(items)",len(items)


    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]


    # Split train/test set
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]


    model_para = {
        #all parameters shuold be consist with those in nextitred.py!!!!
        'item_size': len(items),
        'dilated_channels': 100,
        'dilations': [1, 2,],
        'kernel_size': 3,
        'learning_rate':0.001,
        'batch_size':32,
        'iterations':2,#useless, can be removed
        'is_negsample':False #False denotes no negative sampling
    }



    itemrec = generator_recsys.NextItNet_Decoder(model_para)
    itemrec.train_graph(model_para['is_negsample'])
    itemrec.predict_graph(model_para['is_negsample'],reuse=True)

    sess= tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    saver.restore(sess,"Data/Models/generation_model/model_nextitnet.ckpt")




    batch_no_test = 0
    batch_size_test = model_para['batch_size']
    curr_preds_5 = []
    rec_preds_5 = []  # 1
    ndcg_preds_5 = []  # 1
    curr_preds_20 = []
    rec_preds_20 = []  # 1
    ndcg_preds_20 = []  # 1
    while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
        item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
        [probs] = sess.run(
            [itemrec.g_probs],
            feed_dict={
                itemrec.input_predict: item_batch
            })
        for bi in range(probs.shape[0]):
            pred_items_5 = utils.sample_top_k(probs[bi][-1], top_k=args.top_k)  # top_k=5
            pred_items_20 = utils.sample_top_k(probs[bi][-1], top_k=args.top_k + 15)

            true_item = item_batch[bi][-1]
            predictmap_5 = {ch: i for i, ch in enumerate(pred_items_5)}
            pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

            rank_5 = predictmap_5.get(true_item)
            rank_20 = pred_items_20.get(true_item)
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
        # print "curr_preds",curr_preds




if __name__ == '__main__':
    main()
