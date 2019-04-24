#!/usr/bin/env bash

base_path=/home/aki24695/sprp-acl2018

model_name=sprp_onmt_baseline_128

model_file=sprp_onmt_baseline_128_acc_0.00_ppl_1.51_e1.pt

python $base_path/git/OpenNMT-py/translate.py \
-gpu 0 \
-batch_size 1 \
-model $base_path/git/phrasing/models/${model_name}/${model_file} \
-src $base_path/git/Split-and-Rephrase/baseline-seq2seq/test.complex.unique \
-output $base_path/git/phrasing/models/${model_name}/test.complex.unique.output \
-beam_size 12 \
-replace_unk

python ${base_path}/src/training_scripts/${model_name}/test.py



