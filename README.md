# SleepEEGNet: Automated Sleep Stage Scoring with Sequence to Sequence Deep Learning Approach
In this study, we introduced a novel deep learning approach, called SleepEEGNet, for automated sleep stage scoring using a single-channel EEG.

# Paper
 Our paper can be downloaded from the [arxiv website](https://arxiv.org/pdf/1812.07421.pdf)
 * The Model architecture
  ![Alt text](/images/seq2seq_sleep.jpg)  
 * The CNN architecture
  ![Alt text](/images/seq2seq_cnn.jpg)
 
## Recruitments
* Python 2.7
* tensorflow/tensorflow-gpu
* numpy
* scipy
* scikit-learn
* matplotlib
* imblearn
## Dataset
We evaluated our model using [the Physionet Sleep-EDF datasets](https://physionet.org/physiobank/database/sleep-edfx/) published in 2013 and 2018
* To download SC subjects from the Sleep_EDF (2013) dataset, use the below script:
```
cd data_2013
chmod +x download_physionet.sh
./download_physionet.sh
```

* To download SC subjects from the Sleep_EDF (2013) dataset, use the below script:
```
cd data_2018
chmod +x download_physionet.sh
./download_physionet.sh
```

## Train

* Modify args settings in seq_seq_annot_aami.py for the intra-patient ECG heartbeat classification
* Modify args settings in seq_seq_annot_DS1DS2.py for the inter-patient ECG heartbeat classification

* Run each file to reproduce the model described in the paper, use:

```
python seq_seq_annot_aami.py --data_dir data/s2s_mitbih_aami --epochs 500
```
```
python seq_seq_annot_DS1DS2.py --data_dir data/s2s_mitbih_aami_DS1DS2 --epochs 500
```
## Results
  ![Alt text](/images/results.jpg)
## Citation
If you find it useful, please cite our paper as follows:

```
@article{mousavi2019sleepEEGnet,
  title={SleepEEGNet: Automated Sleep Stage Scoring with Sequence to Sequence Deep Learning Approach},
  author={Sajad Mousavi, Fatemeh Afghah and U. Rajendra Acharya},
  journal={arXiv preprint arXiv:1812.07421},
  year={2018}
}
```

## References
 [github:akaraspt](https://github.com/akaraspt/deepsleepnet)  
 [deepschool.io](https://github.com/sachinruk/deepschool.io/blob/master/DL-Keras_Tensorflow)
 
## Licence 
For academtic and non-commercial usage 
