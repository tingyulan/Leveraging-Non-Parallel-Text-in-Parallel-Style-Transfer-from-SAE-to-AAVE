# Text Style Transfer from SAE to AAVE

## To run [STRAP](https://github.com/martiansideofthemoon/style-transfer-paraphrase) on [AAVE/SAE Paired Dataset](https://github.com/sophiegroenwold/AAVE_SAE_dataset)
1. Add [this folder](https://drive.google.com/drive/folders/1fp5sTFtX3ZR9IaCXkk8j0sh4BSrIA5s9?usp=sharing)'s shortcut to your Google Drive home directory. It contains the shortcuts to the model checkpoints STRAP released too.
3. Run this [Colab Notebook](https://colab.research.google.com/drive/1xX1X3Pa_PzBxunE5aRHSoW7rGDFVbjNw?usp=sharing).


## To run [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf) on [AAVE/SAE Paired Dataset](https://github.com/sophiegroenwold/AAVE_SAE_dataset)
* The code is modified from [cnn-text-classification-pytorch by Shawn1993](https://github.com/Shawn1993/cnn-text-classification-pytorch)
1. `cd cnn-text-classification-pytorch`
2. Download [pre-trained model weights](https://drive.google.com/file/d/13UlAdCaZUYjT69Mb8SZYckbntto9QrQY/view?usp=sharing) and put it under `./weights/`
3. `python main.py -test -snapshot="./weights/best_steps.pt" -field_dir "./weights/" -pred_path result_filepath.csv`


## To run bleu score
1. `python bleu_score.py --filepath result_filepath.csv`

## To run BERT-Score
1. Please see [bert_score](https://github.com/Tiiiger/bert_score)