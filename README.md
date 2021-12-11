# Text Style Transfer from SAE to AAVE

Dataset: [AAVE/SAE Paired Dataset](https://github.com/sophiegroenwold/AAVE_SAE_dataset)

## Metrics
### To run [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
* The code is modified from [cnn-text-classification-pytorch by Shawn1993](https://github.com/Shawn1993/cnn-text-classification-pytorch)
1. `cd cnn-text-classification-pytorch`
2. Download [pre-trained model weights](https://drive.google.com/file/d/13UlAdCaZUYjT69Mb8SZYckbntto9QrQY/view?usp=sharing) and put it under `./weights/`
3. `python main.py -test -snapshot="./weights/best_steps.pt" -field_dir "./weights/" -pred_path result_filepath.csv`

### To run BLEU score
1. `python bleu_score.py --filepath result_filepath.csv`

### To run BERT-Score
1. Please see [bert_score](https://github.com/Tiiiger/bert_score)

### To run DistilBERT style classifier
Open [this notebook](https://colab.research.google.com/drive/1bcs5FO68IdXUHSgU5y2psdOTfU5Ve-8E?usp=sharing) on Google Colab, put the data under `data/` and run it.

### To run GPT-2 perplexity
Run [this notebook](https://colab.research.google.com/drive/1io1q0jcPJmaoRtpDnsSWxCX1vKl4EmO8?usp=sharing) on your data.
It takes around 6 minutes to fine-tune a model on our dataset.

### To run RoBERTa fluency classifier
Follow the instructions for running STRAP below for the setup.
It runs [`evaluate_twitter.sh`](https://github.com/tingyulan/Leveraging-Non-Parallel-Text-in-Parallel-Style-Transfer-from-SAE-to-AAVE/blob/main/style-transfer-paraphrase/style_paraphrase/evaluation/scripts/evaluate_twitter.sh), which calls [`acceptability.py`](https://github.com/tingyulan/Leveraging-Non-Parallel-Text-in-Parallel-Style-Transfer-from-SAE-to-AAVE/blob/main/style-transfer-paraphrase/style_paraphrase/evaluation/scripts/acceptability.py) for evaluation with this metric.


## Baselines
### BART and T5
1. Add [this folder](https://drive.google.com/drive/folders/1BzAVyx6XSaOD-gC7ajay7qezBaBggZmU?usp=sharing)'s shortcut to your Google Drive home directory. It contains the original dataset, DOPE's dataset, training script as ipynb, and the checkpoint after our training.
2. For BART as baseline please run baseline_BART.ipynb. Although we have the notebook in this repository, we recommend running it via Google Colab in the above folder.
3. For T5 as baseline please run baseline_T5.ipynb. 

### GPT-2
Open [this notebook](https://colab.research.google.com/drive/1POoRzl9DBb3NsppP-gyeBKR1VxpCvihS?usp=sharing) on Google Colab, update the data at Colab's default directory, and run the notebook.

### [STRAP](https://github.com/martiansideofthemoon/style-transfer-paraphrase)
1. Add [this folder](https://drive.google.com/drive/folders/1fp5sTFtX3ZR9IaCXkk8j0sh4BSrIA5s9?usp=sharing)'s shortcut to your Google Drive home directory. It contains the shortcuts to the model checkpoints STRAP released too.
3. Run this [Colab Notebook](https://colab.research.google.com/drive/1xX1X3Pa_PzBxunE5aRHSoW7rGDFVbjNw?usp=sharing).

## DOPE - T5
1. Add [this folder](https://drive.google.com/drive/folders/1BzAVyx6XSaOD-gC7ajay7qezBaBggZmU?usp=sharing)'s shortcut to your Google Drive home directory. It contains the original dataset, DOPE's dataset, training script as ipynb, and the checkpoint after our training.
2. Please run DOPE_T5.ipynb. We recommend running it via Google Colab. Although we have the notebook in this repository, we recommend running it via Google Colab in the above folder.
