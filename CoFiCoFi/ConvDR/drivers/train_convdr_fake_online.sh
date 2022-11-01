CUDA_VISIBLE_DEVICES=2,3 python convdr_rocketv2_online.py  --output_dir=/***/my_checkpoints6/  \
--model_name_or_path=../thu_checkpoints/convdr-multi-orquac.cp  --train_file=../datasets/or-quac/train.rank_multitask.jsonl \
--query=no_res  --per_gpu_train_batch_size=1  --learning_rate=5e-6  --log_dir=logs/convdr_multi_orquac  \
--num_train_epochs=1  --model_type=dpr  --log_steps=100  --ranking_task --no_mse --fp16=True --save_steps=2000 \
--reader_model_name_or_path=/***/my_checkpoints/reader_rocketqa_orig --num_negatives=9 --overwrite_output_dir \
--update_steps=300


# 原始 nohup CUDA_VISIBLE_DEVICES=2,3 python convdr_rocketv2_online.py  --output_dir=/home/xbli/ConvDR-main/my_checkpoints6/
#  --update_steps=1000
# train_convdr_online0.log

# 原始 nohup CUDA_VISIBLE_DEVICES=2,3 python convdr_rocketv2_online.py  --output_dir=/home/xbli/ConvDR-main/my_checkpoints6/
#  --update_steps=300
# train_convdr_online1.log
