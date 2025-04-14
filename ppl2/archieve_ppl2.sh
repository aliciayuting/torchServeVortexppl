torch-model-archiver \
  --model-name speech_pipeline \
  --version 1.0 \
  --handler handler.py \
  --extra-files model_pipeline.py,SenseVoice,data \
  --export-path model_store \
  --force