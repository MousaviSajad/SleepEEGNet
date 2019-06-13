import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from  sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
import random
import time
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from dataloader import SeqDataLoader
import argparse

def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
#     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size]
        start += batch_size
def flatten(name, input_var):
    dim = 1
    for d in input_var.get_shape()[1:].as_list():
        dim *= d
    output_var = tf.reshape(input_var,
                            shape=[-1, dim],
                            name=name)

    return output_var
def build_firstPart_model(input_var,keep_prob_=0.5):
        # List to store the output of each CNNs
        output_conns = []

        ######### CNNs with small filter size at the first layer #########

        # Convolution
        network = tf.layers.conv1d(inputs=input_var, filters=64, kernel_size=50, strides=6,
                                 padding='same', activation=tf.nn.relu)

        network = tf.layers.max_pooling1d(inputs=network, pool_size=8, strides=8, padding='same')

        # Dropout
        network = tf.nn.dropout(network, keep_prob_)


        # Convolution
        network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                                 padding='same', activation=tf.nn.relu)

        network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                                 padding='same', activation=tf.nn.relu)
        network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                                 padding='same', activation=tf.nn.relu)


        # Max pooling
        network = tf.layers.max_pooling1d(inputs=network, pool_size=4, strides=4, padding='same')


        # Flatten
        network = flatten(name="flat1", input_var=network)


        output_conns.append(network)

        ######### CNNs with large filter size at the first layer #########



        # Convolution
        network = tf.layers.conv1d(inputs=input_var, filters=64, kernel_size=400, strides=50,
                                   padding='same', activation=tf.nn.relu)

        network = tf.layers.max_pooling1d(inputs=network, pool_size=4, strides=4, padding='same')

        # Dropout
        network = tf.nn.dropout(network, keep_prob_)

        # Convolution
        network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=6, strides=1,
                                   padding='same', activation=tf.nn.relu)

        network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=6, strides=1,
                                   padding='same', activation=tf.nn.relu)
        network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=6, strides=1,
                                   padding='same', activation=tf.nn.relu)

        # Max pooling
        network = tf.layers.max_pooling1d(inputs=network, pool_size=2, strides=2, padding='same')

        # Flatten
        network = flatten(name="flat2", input_var=network)


        output_conns.append(network)

        # Concat
        network = tf.concat(output_conns,1, name="concat1")

        # Dropout
        network = tf.nn.dropout(network, keep_prob_)

        return network
def plot_attention(attention_map, input_tags = None, output_tags = None):
    attn_len = len(attention_map)

    # Plot the attention_map
    plt.clf()
    f = plt.figure(figsize=(15, 10))
    ax = f.add_subplot(1, 1, 1)

    # Add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='gray')

    # Add colorbar
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    # Add labels
    ax.set_yticks(range(attn_len))
    if output_tags != None:
      ax.set_yticklabels(output_tags[:attn_len])

    ax.set_xticks(range(attn_len))
    if input_tags != None:
      ax.set_xticklabels(input_tags[:attn_len], rotation=45)

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()
    HERE = os.path.realpath(os.path.join(os.path.realpath(__file__), '..'))
    dir_save = os.path.join(HERE, 'attention_maps')
    if (os.path.exists(dir_save) == False):
        os.mkdir(dir_save)
    f.savefig(os.path.join(dir_save, 'a_map_1.pdf'), bbox_inches='tight')
    # f.show()
    plt.show()
def build_network(hparams,char2numY,inputs,dec_inputs,keep_prob_=0.5,):

    if hparams.akara2017 is True:
        _inputs = tf.reshape(inputs, [-1, hparams.input_depth,1])
        network = build_firstPart_model(_inputs, keep_prob_)
        shape = network.get_shape().as_list()
        data_input_embed = tf.reshape(network, (-1, hparams.max_time_step, shape[1]))
    else:
        _inputs = tf.reshape(inputs, [-1, hparams.n_channels, hparams.input_depth / hparams.n_channels])

        conv1 = tf.layers.conv1d(inputs=_inputs, filters=32, kernel_size=2, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=2, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=128, kernel_size=2, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

        shape = max_pool_3.get_shape().as_list()
        data_input_embed = tf.reshape(max_pool_3, (-1, hparams.max_time_step, shape[1] * shape[2]))

    # timesteps = max_time
    # lstm_in = tf.unstack(data_input_embed, timesteps, 1)
    # lstm_size = 128
    # # Get lstm cell output
    # # Add LSTM layers
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # data_input_embed, states = tf.contrib.rnn.static_rnn(lstm_cell, lstm_in, dtype=tf.float32)
    # data_input_embed = tf.stack(data_input_embed, 1)
    # shape = data_input_embed.get_shape().as_list()
    # embed_size = 10 #128 lstm_size # shape[1]*shape[2]

    # Embedding layers
    with tf.variable_scope("embeddin") as embedding_scope:
        decoder_embedding = tf.Variable(tf.random_uniform((len(char2numY), hparams.embed_size), -1.0, 1.0), name='dec_embedding') # +1 to consider <EOD>
        decoder_emb_inputs = tf.nn.embedding_lookup(decoder_embedding, dec_inputs)


    with tf.variable_scope("encoding") as encoding_scope:
        if not hparams.bidirectional:

            # Regular approach with LSTM units
            # encoder_cell = tf.contrib.rnn.LSTMCell(hparams.num_units)
            # encoder_cell = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * hparams.lstm_layers)
            def lstm_cell():
                lstm = tf.contrib.rnn.LSTMCell(hparams.num_units)
                return lstm
            encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hparams.lstm_layers)])
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, inputs=data_input_embed, dtype=tf.float32)

        else:

            # Using a bidirectional LSTM architecture instead
            # enc_fw_cell = tf.contrib.rnn.LSTMCell(hparams.num_units)
            # enc_bw_cell = tf.contrib.rnn.LSTMCell(hparams.num_units)

            def lstm_cell():
                lstm = tf.contrib.rnn.LSTMCell(hparams.num_units)
                return lstm

            stacked_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hparams.lstm_layers)],state_is_tuple=True)
            stacked_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hparams.lstm_layers)],state_is_tuple=True)


            ((enc_fw_out, enc_bw_out), (enc_fw_final, enc_bw_final)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=stacked_cell_fw,
                cell_bw=stacked_cell_bw,
                inputs=data_input_embed,
                dtype=tf.float32)
            encoder_final_state = []
            for layer in range(hparams.lstm_layers):
                enc_fin_c = tf.concat((enc_fw_final[layer].c, enc_bw_final[layer].c), 1)
                enc_fin_h = tf.concat((enc_fw_final[layer].h, enc_bw_final[layer].h), 1)
                encoder_final_state.append(tf.contrib.rnn.LSTMStateTuple(c=enc_fin_c, h=enc_fin_h))

            encoder_state = tuple(encoder_final_state)
            encoder_outputs = tf.concat((enc_fw_out, enc_bw_out), 2)


    with tf.variable_scope("decoding") as decoding_scope:

        output_layer = Dense(
            len(char2numY), use_bias=False)
        decoder_lengths = np.ones((hparams.batch_size), dtype=np.int32) * (hparams.max_time_step+1)
        training_helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, decoder_lengths)

        if not hparams.bidirectional:
            # decoder_cell = tf.contrib.rnn.LSTMCell(hparams.num_units)
            def lstm_cell():
                lstm = tf.contrib.rnn.LSTMCell(hparams.num_units)
                return lstm
            decoder_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hparams.lstm_layers)])

        else:
            # decoder_cell = tf.contrib.rnn.LSTMCell(2 * hparams.num_units)
            def lstm_cell():
                lstm = tf.contrib.rnn.LSTMCell(2 * hparams.num_units)
                return lstm
            decoder_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hparams.lstm_layers)])

        if hparams.use_attention:
            # Create an attention mechanism
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                hparams.num_units * 2 if hparams.bidirectional else hparams.num_units , encoder_outputs,
                memory_sequence_length=None)

            decoder_cells = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cells, attention_mechanism,
                attention_layer_size=hparams.attention_size,alignment_history=True)

            encoder_state = decoder_cells.zero_state(hparams.batch_size, tf.float32).clone(cell_state=encoder_state)



        # Basic Decoder and decode
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cells, training_helper, encoder_state,
            output_layer=output_layer)

        dec_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True)

        # dec_outputs, _ = tf.nn.dynamic_rnn(decoder_cell, inputs=decoder_emb_inputs, initial_state=encoder_state)

    logits = dec_outputs.rnn_output

    # Inference
    start_tokens =  tf.fill([hparams.batch_size], char2numY['<SOD>'])
    end_token = char2numY['<EOD>']
    if not hparams.use_beamsearch_decode:

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            decoder_embedding,
            start_tokens,end_token)

        # Inference Decoder
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cells, inference_helper, encoder_state,
            output_layer=output_layer)
    else:

        encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=hparams.beam_width)
        decoder_initial_state = decoder_cells.zero_state(hparams.batch_size * hparams.beam_width, tf.float32).clone(cell_state=encoder_state)

        inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=decoder_cells,
                                                                  embedding=decoder_embedding,
                                                                  start_tokens=start_tokens,
                                                                  end_token=end_token,
                                                                  initial_state=decoder_initial_state,
                                                                  beam_width=hparams.beam_width,
                                                                  output_layer=output_layer)

    # Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder,impute_finished = False, maximum_iterations=hparams.output_max_length)
    pred_outputs = outputs.sample_id
    if  hparams.use_beamsearch_decode:
          # [batch_size, max_time_step, beam_width]
          pred_outputs = pred_outputs[0]


    return logits,pred_outputs,_final_state
def tf_confusion_metrics(y_true_,y_pred_,num_classes=5):

    tf_cm = tf.cast(tf.confusion_matrix(y_true_, y_pred_,num_classes=None),"float")
    FP =  tf.reduce_sum(tf_cm,axis=0) - tf.diag_part(tf_cm)
    FN =  tf.reduce_sum(tf_cm,axis=1) - tf.diag_part(tf_cm)
    TP = tf.diag_part(tf_cm)
    TN = tf.reduce_sum(tf_cm) - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    return FPR, FNR
def evaluate_metrics(cm,classes):

    print ("Confusion matrix:")
    print (cm)

    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    ACC_macro = np.mean(ACC) # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)

    F1 = (2 * PPV * TPR) / (PPV + TPR)
    F1_macro = np.mean(F1)

    print ("Sample: {}".format(int(np.sum(cm))))
    n_classes = len(classes)
    for index_ in range(n_classes):
        print ("{}: {}".format(classes[index_], int(TP[index_] + FN[index_])))


    return ACC_macro,ACC, F1_macro, F1, TPR, TNR, PPV
random.seed(654) # to make have the same training set and test set each time the code is run, we use a fixed random seed

def build_whole_model(hparams,char2numY,inputs, targets,dec_inputs, keep_prob_):
    # logits = build_network(inputs,dec_inputs=dec_inputs)
    logits, pred_outputs,dec_states = build_network(hparams,char2numY,inputs, dec_inputs, keep_prob_)
    decoder_prediction = tf.argmax(logits, 2)

    # optimization operation
    with tf.name_scope("optimization"):
        # Loss function
        vars = tf.trainable_variables()
        beta = 0.001
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                           if 'bias' not in v.name]) * beta

        # class_ratio = [0.1,0.4, 0.1, 0.1, 0.1, 0.1,0.1]
        # class_weight = tf.constant(class_ratio)
        # weighted_logits = tf.multiply(logits, class_weight)

        loss_is = []
        for i in range(logits.get_shape().as_list()[-1]):
            class_fill_targets = tf.fill(tf.shape(targets), i)
            weights_i = tf.cast(tf.equal(targets, class_fill_targets), "float")
            loss_is.append(tf.contrib.seq2seq.sequence_loss(logits, targets, weights_i,average_across_batch=False))

        loss = tf.reduce_sum(loss_is,axis=0)

        # loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([hparams.batch_size, hparams.max_time_step+1])) #+1 is because of the <EOD> token
        # Optimizer
        loss = tf.reduce_mean(loss)+lossL2
        optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

    return logits, pred_outputs, loss, optimizer,dec_states

def run_program(hparams,FLAGS):
    # load dataset
    num_folds = FLAGS.num_folds
    data_dir = FLAGS.data_dir
    if '13' in data_dir:
        data_version = 2013
    else:
        n_oversampling = 30000
        data_version = 2018

    output_dir = FLAGS.output_dir
    classes = FLAGS.classes
    n_classes = len(classes)

    path, channel_ename = os.path.split(data_dir)
    traindata_dir = os.path.join(os.path.abspath(os.path.join(data_dir, os.pardir)),'traindata/')
    print(str(datetime.now()))

    def evaluate_model(hparams, X_test, y_test, classes):
        acc_track = []
        n_classes = len(classes)
        y_true = []
        y_pred = []
        alignments_alphas_all = []  # (batch_num,B,max_time_step,max_time_step)
        for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_test, y_test, hparams.batch_size)):
            # if source_batch.shape[1] != hparams.max_time_step:
            #     print ("Num of steps is: ", source_batch.shape[1])
            # try:
            pred_outputs_ = sess.run(pred_outputs,
                                     feed_dict={inputs: source_batch, keep_prob_: 1.0})

            alignments_alphas = sess.run(dec_states.alignment_history.stack(),
                                         feed_dict={inputs: source_batch, dec_inputs: target_batch[:, :-1],
                                                    keep_prob_: 1.0})

            # acc_track.append(np.mean(dec_input == target_batch))
            pred_outputs_ = pred_outputs_[:, :hparams.max_time_step]  # remove the last prediction <EOD>
            target_batch_ = target_batch[:, 1:-1]  # remove the last <EOD> and the first <SOD>
            acc_track.append(pred_outputs_ == target_batch_)

            alignments_alphas = alignments_alphas.transpose((1, 0, 2))
            alignments_alphas = alignments_alphas[:, :hparams.max_time_step]
            alignments_alphas_all.append(alignments_alphas)

            _y_true = target_batch_.flatten()
            _y_pred = pred_outputs_.flatten()

            y_true.extend(_y_true)
            y_pred.extend(_y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
        ck_score = cohen_kappa_score(y_true, y_pred)
        acc_avg, acc, f1_macro, f1, sensitivity, specificity, PPV = evaluate_metrics(cm, classes)
        # print ("batch_i: {}").format(batch_i)
        print(
        'Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen\'s Kappa -> {:>6.4f} on test set'.format(acc_avg,
                                                                                                               f1_macro,
                                                                                                               ck_score))
        for index_ in range(n_classes):
            print(
            "\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision (PPV): {:1.4f}, F1 : {:1.4f} Accuracy: {:1.4f}".format(
                classes[index_],
                sensitivity[
                    index_],
                specificity[
                    index_], PPV[index_], f1[index_],
                acc[index_]))
        print(
        "\tAverage -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision (PPV): {:1.4f}, F1-score: {:1.4f}, Accuracy: {:1.4f}".format(
            np.mean(sensitivity), np.mean(specificity), np.mean(PPV), np.mean(f1), np.mean(acc)))

        return acc_avg, f1_macro, ck_score, y_true, y_pred, alignments_alphas_all

    def count_prameters():
        print ('# of Params: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    # folds = [4,5,6,7]
    # # folds = [8,9,10,11]
    # # folds = [12,13,14,15]
    # # folds = [16,17,18,19] 
    # folds = [8]
    # for fold_idx in folds:
    for fold_idx in range(num_folds):
        start_time_fold_i = time.time()
        data_loader = SeqDataLoader(data_dir, num_folds, fold_idx, classes=classes)
        X_train, y_train, X_test, y_test = data_loader.load_data(seq_len=hparams.max_time_step)

        # preprocessing
        char2numY = dict(zip(classes, range(len(classes))))        
        pre_f1_macro = 0

        # <SOD> is a token to show start of decoding  and <EOD> is a token to indicate end of decoding
        char2numY['<SOD>'] = len(char2numY)
        char2numY['<EOD>'] = len(char2numY)
        num2charY = dict(zip(char2numY.values(), char2numY.keys()))


        # over-sampling: SMOTE:
        X_train = np.reshape(X_train,[X_train.shape[0]*X_train.shape[1],-1])
        y_train= y_train.flatten()

        if data_version == 2018:
            # extract just undersamples For 2018
            under_sample_len = 35000#30000
            Ws = np.where(y_train == char2numY['W'])[0]
            len_W = len(np.where(y_train == char2numY['W'])[0])
            permute = np.random.permutation(len_W)
            len_r = len_W - under_sample_len if (len_W - under_sample_len) > 0 else 0
            permute = permute[:len_r]
            y_train = np.delete(y_train,Ws[permute],axis =0)
            X_train = np.delete(X_train,Ws[permute],axis =0)

            under_sample_len = 35000 #40000
            N2s = np.where(y_train == char2numY['N2'])[0]
            len_N2 = len(np.where(y_train == char2numY['N2'])[0])
            permute = np.random.permutation(len_N2)
            len_r = len_N2 - under_sample_len if (len_N2 - under_sample_len) > 0 else 0
            permute = permute[:len_r]
            y_train = np.delete(y_train, N2s[permute],axis =0)
            X_train = np.delete(X_train, N2s[permute],axis =0)

        nums = []
        for cl in classes:
            nums.append(len(np.where(y_train == char2numY[cl])[0]))

        if (os.path.exists(traindata_dir) == False):
            os.mkdir(traindata_dir)
        fname = os.path.join(traindata_dir,'trainData_'+channel_ename+'_SMOTE_all_10s_f'+str(fold_idx)+'.npz')

        if (os.path.isfile(fname)):
            X_train, y_train,_ = data_loader.load_npz_file(fname)

        else:
            if data_version == 2013:
                n_osamples = nums[2] - 7000
                ratio = {0: n_osamples if nums[0] < n_osamples else nums[0], 1: n_osamples if nums[1] < n_osamples else nums[1],
                         2: nums[2], 3: n_osamples if nums[3] < n_osamples else nums[3], 4: n_osamples if nums[4] < n_osamples else nums[4]}


            if data_version==2018:
                ratio = {0: n_oversampling if nums[0] < n_oversampling else nums[0], 1: n_oversampling if nums[1] < n_oversampling else nums[1], 2: nums[2],
                     3: n_oversampling if nums[3] < n_oversampling else nums[3], 4: n_oversampling if nums[4] < n_oversampling else nums[4]}

            # ratio = {0: 40000 if nums[0] < 40000 else nums[0], 1: 27000 if nums[1] < 27000 else nums[1], 2: nums[2],
            #          3: 30000 if nums[3] < 30000 else nums[3], 4: 27000 if nums[4] < 27000 else nums[4]}
            sm = SMOTE(random_state=12,ratio=ratio)
            # sm = SMOTE(random_state=12, ratio=ratio)
            # sm = RandomUnderSampler(random_state=12,ratio=ratio)
            X_train, y_train = sm.fit_sample(X_train, y_train)
            data_loader.save_to_npz_file(X_train, y_train,data_loader.sampling_rate,fname)

        X_train = X_train[:(X_train.shape[0] // hparams.max_time_step) * hparams.max_time_step, :]
        y_train = y_train[:(X_train.shape[0] // hparams.max_time_step) * hparams.max_time_step]

        X_train = np.reshape(X_train,[-1,X_test.shape[1],X_test.shape[2]])
        y_train = np.reshape(y_train,[-1,y_test.shape[1],])

        # shuffle training data_2013
        permute = np.random.permutation(len(y_train))
        X_train = np.asarray(X_train)
        X_train = X_train[permute]
        y_train = y_train[permute]


        # add '<SOD>' to the beginning of each label sequence, and '<EOD>' to the end of each label sequence (both for training and test sets)
        y_train= [[char2numY['<SOD>']] + [y_ for y_ in date] + [char2numY['<EOD>']] for date in y_train]
        y_train = np.array(y_train)


        y_test= [[char2numY['<SOD>']] + [y_ for y_ in date] + [char2numY['<EOD>']] for date in y_test]
        y_test = np.array(y_test)

        print ('The training set after oversampling: ', classes)
        for cl in classes:
            print (cl, len(np.where(y_train==char2numY[cl])[0]))

        # training and testing the model
        if (os.path.exists(FLAGS.checkpoint_dir) == False):
            os.mkdir(FLAGS.checkpoint_dir)

        if (os.path.exists(output_dir) == False):
            os.makedirs(output_dir)
        loss_track = []
        with tf.Graph().as_default(), tf.Session() as sess:

            # Placeholders
            inputs = tf.placeholder(tf.float32, [None, hparams.max_time_step, hparams.input_depth], name='inputs')
            targets = tf.placeholder(tf.int32, (None, None), 'targets')
            dec_inputs = tf.placeholder(tf.int32, (None, None), 'decoder_inputs')
            keep_prob_ = tf.placeholder(tf.float32, name='keep')

            # model
            logits, pred_outputs, loss, optimizer,dec_states = build_whole_model(hparams,char2numY,inputs,targets, dec_inputs, keep_prob_)
            count_prameters()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
            print(str(datetime.now()))
            # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            ckpt_name = "model_fold{:02d}.ckpt".format(fold_idx)
            ckpt_exist = False
            for file in os.listdir(FLAGS.checkpoint_dir):
                if file.startswith(ckpt_name):
                    ckpt_exist=True
            ckpt_name = os.path.join(FLAGS.checkpoint_dir, ckpt_name)

            # if ckpt and ckpt.model_checkpoint_path:
            # if os.path.isfile(ckpt_name):
            if ckpt_exist:
                # # Restore
                # ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                # saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
                # saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

                saver.restore(sess, ckpt_name)

                # or 'load meta graph' and restore weights
                # saver = tf.train.import_meta_graph(ckpt_name+".meta")
                # saver.restore(session,tf.train.latest_checkpoint(checkpoint_dir))
                evaluate_model(hparams,X_test, y_test, classes)
            else:

                for epoch_i in range(hparams.epochs):
                    start_time = time.time()
                    # train_acc = []
                    y_true = []
                    y_pred  =[]
                    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, hparams.batch_size)):

                        # _, batch_loss, batch_logits, alignments_alphas = sess.run([optimizer, loss, logits,dec_states.alignment_history.stack()],
                        #     feed_dict = {inputs: source_batch,
                        #                  dec_inputs: target_batch[:, :-1],
                        #                  targets: target_batch[:, 1:],keep_prob_: 0.5} #,
                        #                                        )

                        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                            feed_dict = {inputs: source_batch,
                                         dec_inputs: target_batch[:, :-1],
                                         targets: target_batch[:, 1:],keep_prob_: 0.5} #,
                                                               )
                        loss_track.append(batch_loss)
                        # alignments_alphas = alignments_alphas.transpose((1, 0, 2))
                        # alignments_alphas = alignments_alphas[:, :hparams.max_time_step]
                        # train_acc.append(batch_logits.argmax(axis=-1) == target_batch[:,1:])
                        y_pred_ = batch_logits[:, :hparams.max_time_step].argmax(axis=-1)
                        y_true_ = target_batch[:, 1:-1]

                        # input_tags - word representation of input sequence, use None to skip
                        # output_tags - word representation of output sequence, use None to skip
                        # i - index of input element in batch
                        # input_tags = [[num2charY[i] for i in seq] for seq in y_true_]
                        # output_tags = [[num2charY[i] for i in seq] for seq in y_pred_]
                        # plot_attention(alignments_alphas[1, :, :], input_tags[1], output_tags[1])

                        y_true.extend(y_true_)
                        y_pred.extend(y_pred_)
                    # accuracy = np.mean(train_acc)
                    y_true = np.asarray(y_true)
                    y_pred = np.asarray(y_pred)
                    y_true = y_true.flatten()
                    y_pred = y_pred.flatten()
                    n_examples = len(y_true)
                    cm = confusion_matrix(y_true, y_pred,labels=range(len(char2numY)-2))
                    accuracy = np.mean(y_true == y_pred)
                    mf1 = f1_score(y_true, y_pred, average="macro")
                    ck_score = cohen_kappa_score(y_true, y_pred)

                    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} F1-score: {:>6.4f} Cohen\'s Kappa: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, np.mean(batch_loss),
                                                                                      accuracy,mf1,ck_score, time.time() - start_time))
                    if (epoch_i+1)%hparams.test_step==0:
                        acc_avg, f1_macro,ck_score, y_true, y_pred,alignments_alphas_all = evaluate_model(hparams,X_test, y_test,classes)

                        if np.nan_to_num(f1_macro) > pre_f1_macro: # save the better model based on the f1 score
                            print('Loss {:.4f} after {} epochs (batch_size={})'.format(loss_track[-1], epoch_i + 1,
                                                                                       hparams.batch_size))
                            pre_f1_macro = f1_macro
                            ckpt_name = "model_fold{:02d}.ckpt".format(fold_idx)
                            save_path = os.path.join(FLAGS.checkpoint_dir, ckpt_name)
                            saver.save(sess, save_path)
                            print("The best model (till now) saved in path: %s" % save_path)

                            # Save
                            save_dict = {
                                "y_true": y_true,
                                "y_pred": y_pred,
                                "ck_score": ck_score,
                                "alignments_alphas_all":alignments_alphas_all[:200],# we save just the first 200 batch results because it is so huge
                                }
                            filename = "output_"+channel_ename+"_fold{:02d}.npz".format(fold_idx)
                            save_path = os.path.join(output_dir, filename)
                            np.savez(save_path, **save_dict)
                            print("The best results (till now) saved in path: %s" % save_path)





                # plt.plot(loss_track)
                # plt.show()
                # print 'Classes: ', classes

            print(str(datetime.now()))
            print ('Fold{} took: {:>6.3f}s'.format(fold_idx, time.time()-start_time_fold_i))

def main(args=None):

    FLAGS = tf.app.flags.FLAGS

    # outputs_eeg_fpz_cz
    tf.app.flags.DEFINE_string('data_dir', 'data_2013/eeg_fpz_cz',
                               """Directory where to load training data_2013.""")
    tf.app.flags.DEFINE_string('output_dir', 'outputs_2013/outputs_eeg_fpz_cz',
                               """Directory where to save trained models """
                               """and outputs.""")
    tf.app.flags.DEFINE_integer('num_folds', 20,
                                """Number of cross-validation folds.""")
    tf.app.flags.DEFINE_list('classes', ['W', 'N1', 'N2', 'N3', 'REM'],  """classes""")
    tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoints-seq2seq-sleep-EDF', """Directory to save checkpoints""")
    # tf.app.flags.DEFINE_string('ckpt_name', 'seq2seq_sleep.ckpt',"""Check point name""")

    # hyperparameters
    hparams = tf.contrib.training.HParams(
        epochs=120,  # 300
        batch_size=20,  # 10
        num_units=128,
        embed_size=10,
        input_depth=3000,
        n_channels=100,
        bidirectional=False,
        use_attention=True,
        lstm_layers=2,
        attention_size=64,
        beam_width=4,
        use_beamsearch_decode=False,
        max_time_step=10,  # 5 3 second best 10# 40 # 100
        output_max_length=10 + 2,  # max_time_step +1
        akara2017=True,
        test_step=5,  # each 10 epochs
    )
    # classes = ['W', 'N1', 'N2', 'N3', 'REM']
    run_program(hparams,FLAGS)
if __name__ == "__main__":
     tf.app.run()



