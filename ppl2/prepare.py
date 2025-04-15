import sys
import os
import io
import json
import torch
import time
import csv
import base64
import numpy as np
from ts.torch_handler.base_handler import BaseHandler
from speechRetrieve import AudioRecognition, SearchRetriever, TextChecker, TTSRunner
import pickle
model_dir = os.path.dirname(os.path.abspath(__file__))
sensevoice_path = os.path.join(model_dir, "SenseVoice")
if sensevoice_path not in sys.path:
    sys.path.append(sensevoice_path)

FAISS_INDEX_DIR = "/mydata/msmarco/msmarco_pq.index"
DOC_DIR = "/mydata/msmarco/msmarco_3_clusters/doc_list.pkl"


request_logs = []
log_threshold = 1000
log_path = "/users/TY373/workspace/torchServeVortexppl/ppl2/inference_times.csv"  # Customize per node if needed (e.g., ./n0/inference_times.csv)


try:
    print("~~~~~~~~~~~~~ [TorchServe] Initializing...", flush=True)
    device = "cuda"

    sensevoice_model_name = "iic/SenseVoiceSmall"
    faiss_index_path = FAISS_INDEX_DIR
    doc_pickle_path = DOC_DIR
    text_check_model = "facebook/bart-large-mnli"
    encoder_model_name = "BAAI/bge-small-en-v1.5"
    speech_model = AudioRecognition(device, sensevoice_model_name)
    text_encoder =TextEncoder(device, encoder_model_name)
    search_retriever = SearchRetriever(device, faiss_index_path, topk=5, doc_dir=doc_pickle_path)
    text_checker = TextChecker(device, text_check_model)
    tts_runner = TTSRunner(device)

    speech_model.load_model()
    search_retriever.load_model()
    text_checker.load_model()
    tts_runner.load_model()

    loaded_models = True
    initialized = True
    print("~~~~~~~~~~~~~~~~ Speech handler initialized ~~~~~~~~~~~~`", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)  # This is KEY!
    raise e


