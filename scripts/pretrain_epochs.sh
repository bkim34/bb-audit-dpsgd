EPS=10.0
for SEED in $(seq 0 1 4)
do
    for PRETRAIN_EPOCHS in 1 2 3 4
    do
        # MNIST
        python3 audit_model.py --data_name mnist_half --model_name cnn --lr 1.33e-4 --epsilon $EPS --seed $SEED \
            --fixed_init pretrained_models/cnn_mnist_half_epochs/${PRETRAIN_EPOCHS}epochs.pt \
            --out exp_data/pretrain_epochs/${PRETRAIN_EPOCHS}epochs/seed$SEED/ --block_size 30000 --save_grad_norms
    done
done

# ensure model_init.sh is run first
# copy results from worst-case as pre-train epochs 5
cp -r exp_data/model_init/fixed_worst exp_data/5epochs/