# SleepEEGNet: Automated Sleep Stage Scoring with Sequence to Sequence Deep Learning Approach
In this study, we introduced a novel deep learning approach, called SleepEEGNet, for automated sleep stage scoring using a single-channel EEG.

# Paper
 Our paper can be downloaded from the [arxiv website](https://arxiv.org/pdf/1903.02108).
 * The Model architecture
  ![Alt text](/images/seq2seq_sleep.jpg)  
  
 * The CNN architecture  
 
  ![Alt text](/images/seq2seq_cnn.jpg)
 
## Requirements
* Python 2.7
* tensorflow/tensorflow-gpu
* numpy
* scipy
* matplotlib
* scikit-learn
* matplotlib
* imbalanced-learn(0.4.3)
* pandas
* mne
## Dataset and Data Preparation
We evaluated our model using [the Physionet Sleep-EDF datasets](https://physionet.org/physiobank/database/sleep-edfx/) published in 2013 and 2018.  
We have used the source code provided by [github:akaraspt](https://github.com/akaraspt/deepsleepnet) to prepare the dataset.

* To download SC subjects from the Sleep_EDF (2013) dataset, use the below script:

```
cd data_2013
chmod +x download_physionet.sh
./download_physionet.sh
```

* To download SC subjects from the Sleep_EDF (2018) dataset, use the below script:
```
cd data_2018
chmod +x download_physionet.sh
./download_physionet.sh
```

Use below scripts to extract sleep stages from the specific EEG channels of the Sleep_EDF (2013) dataset:

```
python prepare_physionet.py --data_dir data_2013 --output_dir data_2013/eeg_fpz_cz --select_ch 'EEG Fpz-Cz'
python prepare_physionet.py --data_dir data_2013 --output_dir data_2013/eeg_pz_oz --select_ch 'EEG Pz-Oz'
```

## Train

* Modify args settings in seq2seq_sleep_sleep-EDF.py for each dataset.

* For example, run the below script to train SleepEEGNET model with the 20-fold cross-validation using Fpz-Cz channel of the Sleep_EDF (2013) dataset:
```
python seq2seq_sleep_sleep-EDF.py --data_dir data_2013/eeg_fpz_cz --output_dir output_2013 --num_folds 20
```

## Results
* Run the below script to present the achieved results by SleepEEGNet model for Fpz-Cz channel.
```
python summary.py --data_dir output_2013/eeg_fpz_cz
```

![Alt text](/images/results.jpg)

## Visualization
* Run the below script to visualize attention maps of a sequence input (EEG epochs) for Fpz-Cz channel.
```
python visualize.py --data_dir output_2013/eeg_fpz_cz
```


## Citation

If you find it useful, please cite our paper as follows:

```
@article{mousavi2019sleepEEGnet,
  title={SleepEEGNet: Automated Sleep Stage Scoring with Sequence to Sequence Deep Learning Approach},
  author={Sajad Mousavi, Fatemeh Afghah and U. Rajendra Acharya},
  journal={arXiv preprint arXiv:1903.02108},
  year={2019}
}
```

## References
 [github:akaraspt](https://github.com/akaraspt/deepsleepnet)  
 [deepschool.io](https://github.com/sachinruk/deepschool.io/blob/master/DL-Keras_Tensorflow)
 
## Licence 
For academtic and non-commercial usage 
