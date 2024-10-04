for EPS in 1.0 2.0 4.0 10.0
do
    for SEED in $(seq 0 1 4)
    do
        for N_SAMPLES in 100 1000
        do
            # MNIST
            python3 audit_model.py --data_name mnist_half --model_name cnn --n_df $N_SAMPLES --lr 1.33e-4 \
                --n_epochs 100 --epsilon $EPS --seed $SEED \
                --fixed_init pretrained_models/cnn_mnist_half.pt \
                --out exp_data/dataset_size/${N_SAMPLES}samples/seed$SEED/
            
            # CIFAR-10
            python3 audit_model.py --data_name cifar10_half --model_name cnn --n_df $N_SAMPLES --lr 4e-5 \
                --n_epochs 200 --epsilon $EPS --seed $SEED \
                --fixed_init pretrained_models/cnn_cifar100_cifar10_half.pt \
                --out exp_data/dataset_size/${N_SAMPLES}samples/seed$SEED/ 
        done
    done
done

# ensure model_init.sh is run first
# copy results from worst-case as full dataset
cp -r exp_data/model_init/fixed_worst exp_data/dataset_size/-1samples 