CUDA_VISIBLE_DEVICES=2,3,0 python run_convdr_train.py  --output_dir=/home/xbli/ConvDR-main/reproduce_paper/  \
--model_name_or_path=../thu_checkpoints/ad-hoc-ance-orquac.cp  --train_file=../datasets/or-quac/train.rank.jsonl \
--query=no_res  --per_gpu_train_batch_size=6  --learning_rate=1e-6  --log_dir=logs/convdr_multi_orquac  \
--num_train_epochs=1  --model_type=dpr  --log_steps=100  --ranking_task --fp16=True --save_steps=100 \
--num_negatives=9 --overwrite_output_dir \

# nohup sh train_convdr_origin.sh >train_convdr_orig1.log 2>&1 & --output_dir=/home/xbli/ConvDR-main/my_checkpoints9/  \
# --per_gpu_train_batch_size=8  --learning_rate=1e-5
# CUDA_VISIBLE_DEVICES=2,3,0 python run_convdr_train.py  --output_dir=/home/xbli/ConvDR-main/my_checkpoints9/  \
# --model_name_or_path=../thu_checkpoints/ad-hoc-ance-orquac.cp  --train_file=../datasets/or-quac/train.rank.jsonl \
# --query=no_res  --per_gpu_train_batch_size=6  --learning_rate=1e-6  --log_dir=logs/convdr_multi_orquac  \
# --num_train_epochs=1  --model_type=dpr  --log_steps=100  --ranking_task --fp16=True --save_steps=100 \
# --num_negatives=9 --overwrite_output_dir \