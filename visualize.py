#! /usr/bin/python
# -*- coding: utf8 -*-

import argparse
import os
import re
import scipy.io as sio
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
W=0
N1=1
N2=2
N3=3
REM=4
classes= ['W','N1', 'N2','N3','REM']
n_classes = len(classes)

def plot_attention(attention_map, input_tags = None, output_tags = None):
    attn_len = len(attention_map)

    # Plot the attention_map
    # plt.clf()
    f = plt.figure(figsize=(15, 10))
    ax = f.add_subplot(1, 1, 1)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    # Add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='gray')

    # Add colorbar
    cbaxes = f.add_axes([0.2, -0.02, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.xaxis.label.set_size(16)

    # Add labels
    ax.set_yticks(range(attn_len))
    if output_tags != None:
      ax.set_yticklabels(output_tags[:attn_len])

    ax.set_xticks(range(attn_len))
    if input_tags != None:
      ax.set_xticklabels(input_tags[:attn_len], rotation=45)


    ax.set_xlabel('Input: EEG Epochs (a sequence)')
    ax.xaxis.label.set_size(20)
    ax.set_ylabel('Output: Satge Scoring')
    ax.yaxis.label.set_size(20)

    # add grid and legend
    ax.grid()
    HERE = os.path.realpath(os.path.join(os.path.realpath(__file__), '..'))
    dir_save = os.path.join(HERE, 'attention_maps')
    if (os.path.exists(dir_save) == False):
        os.mkdir(dir_save)
    f.savefig(os.path.join(dir_save, 'a_map_6_5.pdf'), bbox_inches='tight')
    # f.show()
    plt.show()

def visualize(data_dir):

    # Remove non-output files, and perform ascending sort
    allfiles = os.listdir(data_dir)
    outputfiles = []
    for idx, f in enumerate(allfiles):
        if re.match("^output_.+\d+\.npz", f):
            outputfiles.append(os.path.join(data_dir, f))
    outputfiles.sort()


    with np.load(outputfiles[0]) as f:
        y_true = f["y_true"]
        y_pred = f["y_pred"]
        alignments_alphas_all = f["alignments_alphas_all"] # (batch_num,B,max_time_step,max_time_step)

    batch_len = len(alignments_alphas_all)
    char2numY = dict(zip(classes, range(len(classes))))
    num2charY = dict(zip(char2numY.values(), char2numY.keys()))
    shape = alignments_alphas_all.shape
    max_time_step = shape[2]
    batch_num = 6
    alignments_alphas = alignments_alphas_all[batch_num] # get results for the batch of batch_num
    y_true_ = np.reshape(y_true,[-1,shape[1],shape[2]])
    y_pred_ = np.reshape(y_pred, [-1, shape[1], shape[2]])

    input_tags = [[num2charY[i] for i in seq] for seq in y_true_[batch_num,:]]
    output_tags = [[num2charY[i] for i in seq] for seq in y_pred_[batch_num,:]]

    plot_attention(alignments_alphas[5, :, :], input_tags[5], output_tags[5])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="outputs_2013/outputs_eeg_fpz_cz",
                        help="Directory where to load prediction outputs")
    args = parser.parse_args()

    if args.data_dir is not None:
        visualize(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
