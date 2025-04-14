torch-model-archiver \
  --model-name monoflmr \
  --version 1.1 \
  --handler monoflmr_handler.py \
  --extra-files monoflmr.py \
  --export-path model_store \
  --force