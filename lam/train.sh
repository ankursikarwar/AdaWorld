torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    main.py fit \
    --config config/lam.yaml \
    2>&1 | tee output_train.log