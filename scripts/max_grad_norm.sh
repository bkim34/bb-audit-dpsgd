for EPS in 1.0 2.0 4.0 10.0
do
    for SEED in $(seq 0 1 4)
    do
        for MAX_GRAD_NORM in 0.1 10.0
        do
            # MNIST
            python3 audit_model.py --data_name mnist_half --model_name cnn --n_epochs 100 --lr 1.33e-4 \
                --max_grad_norm $MAX_GRAD_NORM --epsilon $EPS --seed $SEED \
                --fixed_init pretrained_models/cnn_mnist_half.pt \
                --out exp_data/max_grad_norm/$MAX_GRAD_NORM/seed$SEED/ --block_size 30000

            # CIFAR-10
            python3 audit_model.py --data_name cifar10_half --model_name cnn --n_df 1000 --n_epochs 200 --lr 4e-5 \
                --max_grad_norm $MAX_GRAD_NORM --epsilon $EPS --seed $SEED \
                --fixed_init pretrained_models/cnn_cifar100_cifar10_half.pt \
                --out exp_data/max_grad_norm/$MAX_GRAD_NORM/seed$SEED/ --block_size 10000
        done
    done
done

# ensure model_init.sh is run first
# copy results from worst-case as 1.0
cp -r exp_data/model_init/fixed_worst exp_data/max_grad_norm/1.0