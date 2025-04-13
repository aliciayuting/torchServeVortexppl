# torchserve --start --ncs \
    # --model-store model_store \
    # --models monoflmr=monoflmr.mar\

torchserve --start --model-store model_store --models monoflmr.mar --disable-token-auth
