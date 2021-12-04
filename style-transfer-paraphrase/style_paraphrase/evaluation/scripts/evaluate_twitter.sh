#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

declare -a arr=(0.0 0.6 0.9)

gtype="nucleus_paraphrase"
split="test"

for top_p in "${arr[@]}"
do
    printf "\n-------------------------------------------\n"
    printf "Mode $gtype  --- "
    printf "top-p ${top_p}, split ${split}"
    printf "\n-------------------------------------------\n\n"

    mkdir -p "$1/eval_${gtype}_${top_p}"
    mkdir -p "$2/eval_${gtype}_${top_p}"

    path0="$1/eval_${gtype}_${top_p}/transfer_aae_${split}.txt"
    path1="$2/eval_${gtype}_${top_p}/transfer_tweets_${split}.txt"
    base_path0="$1/eval_${gtype}_${top_p}"


    printf "\ntranslate tweets to aae\n"
    python style_paraphrase/evaluation/scripts/style_transfer.py \
        --style_transfer_model "$1" \
        --input_file datasets/tweets/${split}.txt \
        --output_file transfer_aae_${split}.txt \
        --generation_mode $gtype \
        --detokenize \
        --post_detokenize \
        --paraphrase_model "$3" \
        --top_p ${top_p}

    printf "\ntranslate aae to tweets\n"
    python style_paraphrase/evaluation/scripts/style_transfer.py \
        --style_transfer_model "$2" \
        --input_file datasets/aae/${split}.txt \
        --output_file transfer_tweets_${split}.txt \
        --generation_mode $gtype \
        --detokenize \
        --post_detokenize \
        --paraphrase_model "$3" \
        --top_p ${top_p}

    cat $path0 $path1 > "${base_path0}/all_${split}_generated.txt"
    cat datasets/tweets/${split}.txt datasets/aae/${split}.txt > "${base_path0}/all_${split}_input.txt"
    cat datasets/aae/${split}.txt datasets/tweets/${split}.txt > "${base_path0}/all_${split}_gold.txt"
    cat datasets/aae/${split}.label datasets/tweets/${split}.label > "${base_path0}/all_${split}_labels.txt"

    python style_paraphrase/evaluation/scripts/flip_labels.py \
        --file1 datasets/tweets/${split}.label \
        --file2 datasets/aae/${split}.label \
        --output_file "${base_path0}/all_${split}_transfer_labels.txt"

    printf "\nRoBERTa ${split} classification\n\n"
    python style_paraphrase/evaluation/scripts/roberta_classify.py --input_file "${base_path0}/all_${split}_generated.txt" --label_file "${base_path0}/all_${split}_labels.txt" --model_dir style_paraphrase/evaluation/accuracy/cds_classifier --model_data_dir style_paraphrase/evaluation/accuracy/cds_classifier/cds-bin

    printf "\nRoBERTa acceptability classification\n\n"
    python style_paraphrase/evaluation/scripts/acceptability.py --input_file "${base_path0}/all_${split}_generated.txt"

    printf "\nParaphrase scores --- generated vs inputs..\n\n"
    python style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py --generated_path "${base_path0}/all_${split}_generated.txt" --reference_strs inputs --reference_paths ${base_path0}/all_${split}_input.txt --output_path ${base_path0}/generated_vs_inputs.txt

    printf "\nParaphrase scores --- generated vs gold..\n\n"
    python style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py --generated_path ${base_path0}/all_${split}_generated.txt --reference_strs gold --reference_paths "${base_path0}/all_${split}_gold.txt" --output_path ${base_path0}/generated_vs_gold.txt --store_scores

    printf "\n final normalized scores vs gold..\n\n"
    python style_paraphrase/evaluation/scripts/micro_eval.py --classifier_file ${base_path0}/all_${split}_generated.txt.roberta_labels --paraphrase_file ${base_path0}/all_${split}_generated.txt.pp_scores --generated_file ${base_path0}/all_${split}_generated.txt --acceptability_file ${base_path0}/all_${split}_generated.txt.acceptability_labels

done