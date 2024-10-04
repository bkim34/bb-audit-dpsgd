# last layer only fine-tuning
for EPS in 1.0 2.0 4.0 10.0
do
    for SEED in $(seq 0 1 4)
    do
        for N_SAMPLES in 100 1000
        do
            python3 audit_model.py --data_name cifar10_half_finetune_last --model_name lr --n_reps 1000 \
                --n_df $N_SAMPLES --lr 4e-4 --n_epochs 20 --epsilon $EPS \
                --target_type target_samples/cifar10_half_finetune_last_clipbkd.npy \
                --fixed_init pretrained_models/cifar10_half_finetune_last.pt --seed $SEED \
                --out exp_data/dataset_size_finetune/${N_SAMPLES}samples/seed$SEED/
        done
    done
done

# ensure model_init_finetune.sh is run first
# copy results from worst-case as full dataset
cp -r exp_data/model_init_finetune/fixed_worst exp_data/dataset_size_finetune/-1samples 