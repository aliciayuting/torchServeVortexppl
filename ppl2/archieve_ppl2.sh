rm model_store/*
torch-model-archiver \
  --model-name monospeech \
  --version 1.0 \
  --handler speech_handler.py \
  --extra-files "speechRetrieve.py,sensevoice_bundle.zip" \
  --export-path model_store \
  --force