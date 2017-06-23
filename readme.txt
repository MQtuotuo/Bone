
data folder: containing dataset
models: different networks
bootstrap.py: downloads the female and male bone age images
train.py: starts end-to-end training process


Step 1: bootstrap and create train, test, validation data sets
python bootstrap.py

Step 2: train the network, add model argument
python train.py --model=inception_v3

Step 3: predict your own image
python predict.py --path "/full/path/to/image" --model=inception_v3