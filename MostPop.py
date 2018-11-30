import tensorflow as tf
import numpy as np
import argparse

import data_loader_recsys as data_loader
import utils
import shutil
import time
import eval
import math


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=320,
                        help='Learning Rate')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='Max Epochs')


    parser.add_argument('--text_dir', type=str, default='Data/Session/user-filter-200000items-session10.csv-map-5to100.csv',
                        help='Directory containing text files')
    parser.add_argument('--seed', type=str, default='f78c95a8-9256-4757-9a9f-213df5c6854e,1151b040-8022-4965-96d2-8a4605ce456c,4277434f-e3c2-41ae-9ce3-23fd157f9347,fb51d2c4-cc69-4128-92f5-77ec38d66859,4e78efc4-e545-47af-9617-05ff816d86e2',
                        help='Seed for text generation')
    parser.add_argument('--sample_percentage', type=float, default=0.5,
                        help='sample_percentage from whole data, e.g.0.2= 80% training 20% testing')

    args = parser.parse_args()



    dl = data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.text_dir})

    all_samples = dl.item
    items = dl.item_dict


    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    text_samples = all_samples[shuffle_indices]


    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(args.sample_percentage * float(len(text_samples)))
    x_train, x_dev = text_samples[:dev_sample_index], text_samples[dev_sample_index:]

    item_dic={}
    for bi in x_train:
        for item in bi:
            if item_dic.has_key(item):
                item_dic[item]=item_dic.get(item)+1
            else:
                item_dic[item]=1
    sorted_names_5 = sorted(item_dic.iteritems(), key=lambda (k, v): (-v, k))[:args.top_k]#top_k=5
    toplist_5=[tuple[0] for tuple in sorted_names_5] #the same order with sorted_names

    sorted_names_20 = sorted(item_dic.iteritems(), key=lambda (k, v): (-v, k))[:(args.top_k+15)]  # top_k=5
    toplist_20= [tuple[0] for tuple in sorted_names_20]  # the same order with sorted_names

    # predictmap=[tuple for tuple in sorted_names]
    predictmap_5=dict(sorted_names_5)
    predictmap_20 = dict(sorted_names_20)

    batch_no_test = 0
    batch_size_test = args.batch_size * 1
    curr_preds_5 = []
    rec_preds_5 = []  # 1
    ndcg_preds_5 = []  # 1
    curr_preds_20 = []
    rec_preds_20 = []  # 1
    ndcg_preds_20 = []  # 1

    while (batch_no_test + 1) * batch_size_test < x_dev.shape[0]:
        if (batch_no_test > 100):
            break
        text_batch = x_dev[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
        for bi in range(batch_size_test):
            # predictmap = sorted_names
            true_word = text_batch[bi][-1]

            rank_5 = predictmap_5.get(true_word)
            rank_20 = predictmap_20.get(true_word)
            if rank_5 == None:
                curr_preds_5.append(0.0)
                rec_preds_5.append(0.0)  # 2
                ndcg_preds_5.append(0.0)  # 2
            else:

                rank_5 = toplist_5.index(true_word)

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
                rank_20 = toplist_20.index(true_word)
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




if __name__ == '__main__':
    main()
