# export TS_LOG_CONFIG=./log4j.properties

torchserve --start \
  --model-store model_store \
  --models monoflmr=monoflmr.mar \
  --disable-token-auth \
  --ts-config config.properties \
#   --log-config ./log4j.properties