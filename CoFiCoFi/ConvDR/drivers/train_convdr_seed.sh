for seed in {66,52}
do
    echo $seed
    CUDA_VISIBLE_DEVICES=2,3 python convdr_recketv2_real_online.py  --output_dir=/**/ConvDR-main/output/qafactor/retriever_5e-6reader2e-5_epoch1_weakreaderdrop01_seed$seed  \
    --model_name_or_path=../thu_checkpoints/ad-hoc-ance-orquac.cp  --train_file=../datasets/or-quac/train.rank_multitask.jsonl \
    --query=no_res  --per_gpu_train_batch_size=2  --learning_rate=5e-6  --log_dir=logs/convdr_multi_orquac  \
    --num_train_epochs=1  --model_type=dpr  --log_steps=100  --ranking_task --fp16=True --save_steps=1000 \
    --reader_model_name_or_path=/**/ConvDR-main/my_checkpoints/bert-base-uncased --num_negatives=9 --overwrite_output_dir \
    --seed=$seed
done