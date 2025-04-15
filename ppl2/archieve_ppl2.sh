rm model_store/*
#!/bin/bash

MODEL_NAME=monospeech
HANDLER=speech_handler.py
EXTRA_FILES="speechRetrieve.py"

# Add all files under SenseVoice/ recursively
EXTRA_FILES+=",$(find SenseVoice -type f | paste -sd "," -)"

torch-model-archiver \
  --model-name "$MODEL_NAME" \
  --version 1.0 \
  --handler "$HANDLER" \
  --extra-files "$EXTRA_FILES" \
  --export-path model_store \
  --force