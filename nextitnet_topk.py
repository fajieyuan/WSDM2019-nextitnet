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

# You can run it directly, first training and then evaluating
# nextitrec_generate.py can only be run when the model parameters are saved, i.e.,
#  save_path = saver.save(sess,
#                       "Data/Models/generation_model/model_nextitnet.ckpt".format(iter, numIters))
# if you are dealing very huge industry dataset, e.g.,several hundred million items, you may have memory problem during training, but it 
# be easily solved by simply changing the last layer, you do not need to calculate the cross entropy loss
# based on the whole item vector. Similarly, you can also change the last layer (use tf.nn.embedding_lookup or gather) in the prediction phrase 
# if you want to just rank the recalled items instead of all items. The current code should be okay if the item size < 5 million.



#Strongly suggest running codes on GPU with more than 10G memory!!!
#if your session data is very long e.g, >50, and you find it may not have very strong internal sequence properties, you can consider generate subsequences
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
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    #this is a demo dataset, which just let you run this code, suggest dataset link: http://grouplens.org/datasets/.
    parser.add_argument('--datapath', type=str, default='Data/Session/user-filter-20000items-session5.csv',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=1000,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=1000,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
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

    if args.is_generatesubsession:
        x_train = generatesubsequence(train_set)

    model_para = {
        #if you changed the parameters here, also do not forget to change paramters in nextitrec_generate.py
        'item_size': len(items),
        'dilated_channels': 100,#larger is better until 512 or 1024
        # if you use nextitnet_residual_block, you can use [1, 4, 1, 4, 1,4,],
        # if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
        # when you change it do not forget to change it in nextitrec_generate.py
        'dilations': [1, 2, 1, 2, 1, 2,],#YOU should tune this hyper-parameter, refer to the paper.
        'kernel_size': 3,
        'learning_rate':0.001,
        'batch_size':128,#YOU should tune this hyper-parameter, options: 32, 64, 128, 256
        'iterations':400,# if your dataset is small, suggest adding regularization to prevent overfitting
        'is_negsample':True #False denotes no negative sampling
    }

    itemrec = generator_recsys.NextItNet_Decoder(model_para)
    itemrec.train_graph(model_para['is_negsample'])
    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1).minimize(itemrec.loss)
    itemrec.predict_graph(model_para['is_negsample'],reuse=True)

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

            if numIters % args.eval_iter == 0:
                print "-------------------------------------------------------test1"
                if (batch_no + 1) * batch_size < valid_set.shape[0]:
                    item_batch = valid_set[(batch_no) * batch_size: (batch_no + 1) * batch_size, :]
                loss = sess.run(
                    [itemrec.loss_test],
                    feed_dict={
                        itemrec.input_predict: item_batch
                    })
                print "LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, iter, batch_no, numIters, valid_set.shape[0] / batch_size)

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
                        if (batch_no_test > 10):
                            break
                    else:
                        if (batch_no_test > 50):
                            break
                    item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                    output_tokens_batch, maskedpositions_batch, maskedlabels_batch = create_masked_predictions_frombatch(
                        item_batch)

                    [top_k_batch] = sess.run(
                        [itemrec.top_k],
                        feed_dict={
                            itemrec.itemseq_input: output_tokens_batch,
                            itemrec.masked_position: maskedpositions_batch
                        })
                    top_k = np.squeeze(top_k_batch[1])
                    for bi in range(top_k.shape[0]):
                        # pred_items_5 = utils.sample_top_k(probs[bi], top_k=args.top_k)#top_k=5
                        # pred_items_20 = utils.sample_top_k(probs[bi], top_k=args.top_k+15)
                        pred_items_5 = top_k[bi][:5]
                        # pred_items_20 = top_k[bi]
                        true_item = item_batch[bi][-1]
                        predictmap_5 = {ch: i for i, ch in enumerate(pred_items_5)}
                        # pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

                        rank_5 = predictmap_5.get(true_item)
                        # rank_20 = pred_items_20.get(true_item)
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

                    batch_no_test += 1
                    if (numIters / (args.eval_iter) < 10):
                        if (batch_no_test == 10):
                            print "mrr_5:", sum(curr_preds_5) / float(len(curr_preds_5)), "hit_5:", sum(
                                rec_preds_5) / float(
                                len(rec_preds_5)), "ndcg_5:", sum(ndcg_preds_5) / float(
                                len(ndcg_preds_5))
                    else:
                        if (batch_no_test == 50):
                            print "mrr_5:", sum(curr_preds_5) / float(len(curr_preds_5)), "hit_5:", sum(
                                rec_preds_5) / float(
                                len(rec_preds_5)), "ndcg_5:", sum(ndcg_preds_5) / float(
                                len(ndcg_preds_5))

            numIters += 1
            if numIters % args.save_para_every == 0:
                save_path = saver.save(sess,
                                       "Data/Models/generation_model/model_nextitnet.ckpt".format(iter, numIters))


if __name__ == '__main__':
    main()