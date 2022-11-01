CUDA_VISIBLE_DEVICES=2 python run_convdr_inference.py  --model_path=../thu_checkpoints/ad-hoc-ance-orquac.cp --eval_file=../datasets/or-quac/train.jsonl \
--query=target  --per_gpu_eval_batch_size=16  --ann_data_dir=../datasets/or-quac/embeddings  \
--qrels=../datasets/or-quac/qrels.tsv --processed_data_dir=../datasets/or-quac/tokenized  --raw_data_dir=../datasets/or-quac \
--output_file=../results/or-quac/manual_ance_train.jsonl  --output_trec_file=../results/or-quac/manual_ance_train.trec  \
--model_type=dpr  --output_query_type=train.manual  --use_gpu \

# CUDA device 不加会出错