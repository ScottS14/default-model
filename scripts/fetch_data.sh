# fetch the dataset from Kaggle
# and unzip it into the data/raw directory
set -e
mkdir -p data/raw
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip data/raw/home-credit-default-risk.zip -d data/raw/home_credit
