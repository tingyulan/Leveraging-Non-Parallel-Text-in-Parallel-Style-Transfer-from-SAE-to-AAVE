import torchtext
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, help='path to filename',default='./results/output')
    opts = parser.parse_args()
    return opts


args = get_args()

filename = args.filepath
df = pd.read_csv(filename)
pred = df['generated_texts'].values.tolist()
candidate_corpus = list(torchtext.data.functional.simple_space_split(pred))

gt = df['references'].values.tolist()
references_corpus = list(torchtext.data.functional.simple_space_split(gt))
references_corpus = [[a] for a in references_corpus]

print(torchtext.data.metrics.bleu_score(candidate_corpus, references_corpus, max_n=4, weights=[0.25, 0.25, 0.25, 0.25]))