torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    main.py fit \
    --config config/lam_ssv2_our_mira.yaml \
    2>&1 | tee output_train_ssv2_our_mira.log
