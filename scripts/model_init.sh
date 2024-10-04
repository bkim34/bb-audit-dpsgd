for EPS in 1.0 2.0 4.0 10.0
do
    for SEED in $(seq 0 1 4)
    do
        # MNIST
        ## average-case
        python3 audit_model.py --data_name mnist_half --model_name cnn --lr 1.33e-4 --epsilon $EPS \
            --fixed_init \
            --seed $SEED --out exp_data/model_init/fixed_average/seed$SEED/ --block_size 30000

        ## worst-case
        python3 audit_model.py --data_name mnist_half --model_name cnn --n_reps 200 --lr 1.33e-4 --epsilon $EPS \
            --fixed_init pretrained_models/cnn_mnist_half.pt \
            --seed $SEED --out exp_data/model_init/fixed_worst/seed$SEED/ --block_size 30000

        # CIFAR-10
        ## average-case
        python3 audit_model.py --data_name cifar10_half --model_name cnn --n_epochs 200 --lr 8e-5 --epsilon $EPS \
            --fixed_init \
            --seed $SEED --out exp_data/model_init/fixed_average/seed$SEED/ --block_size 10000

        ## worst-case
        python3 audit_model.py --data_name cifar10_half --model_name cnn --n_epochs 200 --lr 4e-5 --epsilon $EPS \
            --fixed_init pretrained_models/cnn_cifar100_cifar10_half.pt \
            --seed $SEED --out exp_data/model_init/fixed_worst/seed$SEED/ --block_size 10000
    done
done