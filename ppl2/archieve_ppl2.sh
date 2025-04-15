torch-model-archiver \
  --model-name monospeech \
  --version 1.0 \
  --handler speech_handler.py \
  --extra-files "speechRetrieve.py,$(find SenseVoice -type f | tr '\n' ',')" \
  --export-path model_store \
  --force