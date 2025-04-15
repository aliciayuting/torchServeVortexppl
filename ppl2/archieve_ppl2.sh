rm model_store/*
EXTRA=$(find SenseVoice -type f | paste -sd "," -)

echo "Extra files to include:"
echo "$EXTRA"

torch-model-archiver \
  --model-name monospeech \
  --version 1.0 \
  --handler speech_handler.py \
  --extra-files "${EXTRA},speechRetrieve.py" \
  --export-path model_store \
  --force