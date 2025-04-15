rm model_store/*
EXTRA=$(find SenseVoice -type f | paste -sd "," -)
torch-model-archiver \
  --model-name monospeech \
  --version 1.0 \
  --handler speech_handler.py \
  --extra-files "speechRetrieve.py,${EXTRA}" \
  --export-path model_store \
  --force