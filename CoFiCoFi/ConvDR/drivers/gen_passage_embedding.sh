python3 -m torch.distributed.launch --nproc_per_node=4  gen_passage_embeddings.py  --data_dir=../datasets/or-quac/tokenized  \
--checkpoint=../thu_checkpoints/ad-hoc-ance-orquac.cp  --output_dir=../datasets/or-quac/embeddings \
--model_type=dpr 