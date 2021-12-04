
### Train and Val
```
python main.py
```
### Test
```
python main.py -test -snapshot="./snapshot/2021-11-13_19-56-26/best_steps_2600.pt" -field_dir "./snapshot/2021-11-13_19-56-26" -pred_path "./../results/T5_output_pred.csv"
```
### Data 
```
data
|- aave_clean_train.csv
|- aave_clean_val.csv
|- aave_clean_test.csv
|- sae_clean_train.csv
|- sae_clean_val.csv
`- sae_clean_test.csv

results
|- T5_output.csv
|- T5_output_pred.csv
|- T5_output_gt.csv

cnn-text-classification-pytorch
|- main.py
`- snapshot
  `- 2021-11-13_19-33-15
    |- text_field.Field
    |- label_field.Field
    `- bext_steps_2400.pt
```

### Experiments

T5_output_pred- loss: 0.676092  acc: 68.3168% (138/202)

aave_train- loss: 0.122812 acc: 95.2322% (1538/1615)

aave_val- loss: 0.470344  acc: 80.1980% (162/202)

aave_test- loss: 0.385787  acc: 83.6634% (169/202)

```
Batch[2600] - loss: 0.093615  acc: 96.8750%(62/64))
Evaluation - loss: 0.530538  acc: 79.9505%(323/404)

Batch[2700] - loss: 0.133574  acc: 96.8750%(62/64)
Evaluation - loss: 0.536757  acc: 79.9505%(323/404)

Batch[2800] - loss: 0.178408  acc: 92.1875%(59/64))
Evaluation - loss: 0.550763  acc: 79.9505%(323/404)

Batch[2900] - loss: 0.103764  acc: 92.1875%(59/64))
Evaluation - loss: 0.565227  acc: 79.7030%(322/404)

Batch[3000] - loss: 0.115233  acc: 96.8750%(62/64))
Evaluation - loss: 0.571869  acc: 79.4554%(321/404)

Batch[3100] - loss: 0.113508  acc: 93.7500%(60/64))
Evaluation - loss: 0.589411  acc: 78.9604%(319/404)

Batch[3200] - loss: 0.233915  acc: 87.5000%(56/64))
Evaluation - loss: 0.603081  acc: 79.2079%(320/404)

Batch[3264] - loss: 0.221127  acc: 90.0000%(27/30)%
```
---

## Introduction
This is the implementation of Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in PyTorch.

1. Kim's implementation of the model in Theano:
[https://github.com/yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence)
2. Denny Britz has an implementation in Tensorflow:
[https://github.com/dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
3. Alexander Rakhlin's implementation in Keras;
[https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras)

## Requirement
* python 3
* pytorch > 0.1
* torchtext > 0.1
* numpy

## Result
I just tried two dataset, MR and SST.

|Dataset|Class Size|Best Result|Kim's Paper Result|
|---|---|---|---|
|MR|2|77.5%(CNN-rand-static)|76.1%(CNN-rand-nostatic)|
|SST|5|37.2%(CNN-rand-static)|45.0%(CNN-rand-nostatic)|

I haven't adjusted the hyper-parameters for SST seriously.

## Usage
```
./main.py -h
```
or 

```
python3 main.py -h
```

You will get:

```
CNN text classificer

optional arguments:
  -h, --help            show this help message and exit
  -batch-size N         batch size for training [default: 50]
  -lr LR                initial learning rate [default: 0.01]
  -epochs N             number of epochs for train [default: 10]
  -dropout              the probability for dropout [default: 0.5]
  -max_norm MAX_NORM    l2 constraint of parameters
  -cpu                  disable the gpu
  -device DEVICE        device to use for iterate data
  -embed-dim EMBED_DIM
  -static               fix the embedding
  -kernel-sizes KERNEL_SIZES
                        Comma-separated kernel size to use for convolution
  -kernel-num KERNEL_NUM
                        number of each kind of kernel
  -class-num CLASS_NUM  number of class
  -shuffle              shuffle the data every epoch
  -num-workers NUM_WORKERS
                        how many subprocesses to use for data loading
                        [default: 0]
  -log-interval LOG_INTERVAL
                        how many batches to wait before logging training
                        status
  -test-interval TEST_INTERVAL
                        how many epochs to wait before testing
  -save-interval SAVE_INTERVAL
                        how many epochs to wait before saving
  -predict PREDICT      predict the sentence given
  -snapshot SNAPSHOT    filename of model snapshot [default: None]
  -save-dir SAVE_DIR    where to save the checkpoint
```

## Train
```
./main.py
```
You will get:

```
Batch[100] - loss: 0.655424  acc: 59.3750%
Evaluation - loss: 0.672396  acc: 57.6923%(615/1066) 
```

## Test
If you has construct you test set, you make testing like:

```
/main.py -test -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt
```
The snapshot option means where your model load from. If you don't assign it, the model will start from scratch.

## Predict
* **Example1**

	```
	./main.py -predict="Hello my dear , I love you so much ." \
	          -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt" 
	```
	You will get:
	
	```
	Loading model from [./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt]...
	
	[Text]  Hello my dear , I love you so much .
	[Label] positive
	```
* **Example2**

	```
	./main.py -predict="You just make me so sad and I have to leave you ."\
	          -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt" 
	```
	You will get:
	
	```
	Loading model from [./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt]...
	
	[Text]  You just make me so sad and I have to leave you .
	[Label] negative
	```

Your text must be separated by space, even punctuation.And, your text should longer then the max kernel size.

## Reference
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

