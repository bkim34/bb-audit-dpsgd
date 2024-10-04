# last layer only fine-tuning
for EPS in 1.0 2.0 4.0 10.0
do
    for SEED in $(seq 0 1 4)
    do
        # average-case
        python3 audit_model.py --data_name cifar10_half_finetune_last --model_name lr --n_reps 1000 --lr 4e-4 \
            --n_epochs 20 --epsilon $EPS --target_type target_samples/cifar10_half_finetune_last_clipbkd.npy \
            --fixed_init \
            --seed $SEED --out exp_data/model_init_finetune/fixed_average/seed$SEED/ --block_size 25000

        # worst-case
        python3 audit_model.py --data_name cifar10_half_finetune_last --model_name lr --n_reps 1000 --lr 4e-4 \
            --n_epochs 20 --epsilon $EPS --target_type target_samples/cifar10_half_finetune_last_clipbkd.npy \
            --fixed_init pretrained_models/cifar10_half_finetune_last.pt \
            --seed $SEED --out exp_data/model_init_finetune/fixed_worst/seed$SEED/ --block_size 25000
    done
done