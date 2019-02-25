
# this command downloads the whole dataset
# wget -r -np --cut-dirs=6 -R "index.html*" https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/
wget -nd -r -A edf,hyp https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/
