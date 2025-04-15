import sys
import os
import json
import time
import csv
import base64
import zipfile
import torch
import numpy as np
from ts.torch_handler.base_handler import BaseHandler
# from speechRetrieve import AudioRecognition, SearchRetriever, TextChecker, TTSRunner

# ---------------------------------------------
# Constants
# ---------------------------------------------
model_dir = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(model_dir, "sensevoice_bundle.zip")
FAISS_INDEX_DIR = "/mydata/msmarco/msmarco_pq.index"
DOC_DIR = "/mydata/msmarco/msmarco_3_clusters/doc_list.pkl"

# ---------------------------------------------
# Extract ZIP and patch sys.path
# ---------------------------------------------
if os.path.exists(zip_path):
    print("[TorchServe] Extracting sensevoice_bundle.zip...", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

sensevoice_path = os.path.join(model_dir, "SenseVoice")
if sensevoice_path not in sys.path:
    sys.path.append(sensevoice_path)

# Delay import of dependent modules until after sys.path is patched
AudioRecognition = SearchRetriever = TextChecker = TTSRunner = None  # placeholders


# ---------------------------------------------
# TorchServe Handler
# ---------------------------------------------
class SpeechPipelineHandler(BaseHandler):
    request_logs = []
    log_threshold = 1000
    log_path = os.path.join(model_dir, "inference_times.csv")

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.loaded_models = False

    def initialize(self, ctx):
        global AudioRecognition,TextEncoder, SearchRetriever, TextChecker, TTSRunner

        try:
            print("~~~~~~~~~~~~~ [TorchServe] Initializing SpeechPipelineHandler...", flush=True)
            zip_path = os.path.join(model_dir, "sensevoice_bundle.zip")
            if os.path.exists(zip_path):
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(model_dir)

            sensevoice_path = os.path.join(model_dir, "SenseVoice")
            if sensevoice_path not in sys.path:
                sys.path.append(sensevoice_path)
            from speechRetrieve import AudioRecognition,TextEncoder, SearchRetriever, TextChecker, TTSRunner
            self.device = "cuda"
            self.sensevoice_model_name = "iic/SenseVoiceSmall"
            self.faiss_index_path = FAISS_INDEX_DIR
            self.doc_pickle_path = DOC_DIR
            self.text_check_model = "facebook/bart-large-mnli"
            self.encoder_model_name = "BAAI/bge-small-en-v1.5"

            self.speech_model = AudioRecognition(self.device, self.sensevoice_model_name)
            self.text_encoder = TextEncoder(self.device, self.encoder_model_name)
            self.search_retriever = SearchRetriever(self.device, self.faiss_index_path, topk=1, doc_dir=self.doc_pickle_path)
            self.text_checker = TextChecker(self.device, self.text_check_model)
            self.tts_runner = TTSRunner(self.device)

            self.speech_model.load_model()
            self.search_retriever.load_model()
            self.text_checker.load_model()
            self.tts_runner.load_model()

            self.loaded_models = True
            self.initialized = True
            print("~~~~~~~~~~~~~ Speech handler initialized successfully ~~~~~~~~~~~~~", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise e

    def preprocess(self, data):
        try:
            start_time = time.time()

            input_data = data[0].get("body", data[0])  # fallback to raw dict if no 'body'

            # decode if it's bytes
            if isinstance(input_data, (bytes, bytearray)):
                input_data = input_data.decode("utf-8")
                input_data = json.loads(input_data)
            elif isinstance(input_data, str):
                input_data = json.loads(input_data)


            audio_batch = input_data["audio_data"]
            ids = input_data.get("question_ids", ["?"])
            print(f"~~~~~~~~~~~~~~ [Preprocess] IDs: {ids}", flush=True)

            audio_list = [np.array(waveform, dtype=np.float32) for waveform in audio_batch]
            return audio_list, start_time
        except Exception as e:
            raise RuntimeError(f"[Preprocess] Error parsing input audio: {e}")

    def inference(self, inputs):
        audio_list, start_time = inputs
        print(f"~~~~~~~~~~~~~~ [Inference] Processing {len(audio_list)} audio files...", flush=True)
        if not self.loaded_models:
            self.initialize(None)

        text_list = self.speech_model.exec_model(audio_list)
        print(f"~~~~~~~~~~~~~~ [Inference] Text list: {text_list}", flush=True)
        embeddings = self.text_encoder.encoder_exec(text_list)
        print(f"~~~~~~~~~~~~~~ [Inference] Embeddings: {embeddings}", flush=True)
        doc_lists = self.search_retriever.search_docs(embeddings)
        print(f"~~~~~~~~~~~~~~ [Inference] Document lists: {doc_lists}", flush=True)
        check_results = self.text_checker.docs_check(doc_lists)
        print(f"~~~~~~~~~~~~~~ [Inference] Check results: {check_results}", flush=True)
        audio_outputs = self.tts_runner.model_exec(doc_lists)
        print(f"~~~~~~~~~~~~~~ [Inference] Audio outputs: {audio_outputs}", flush=True)
        end_time = time.time()
        duration = end_time - start_time
        batch_size = len(audio_list)

        per_item_duration = duration / batch_size if batch_size > 0 else 0
        for _ in range(batch_size):
            self.__class__.request_logs.append((start_time, end_time, per_item_duration))

        if len(self.__class__.request_logs) >= self.__class__.log_threshold:
            self.write_logs()

        results = []
        for idx in range(len(audio_list)):
            audio_base64 = [base64.b64encode(a.tobytes()).decode("utf-8") for a in audio_outputs[idx]]
            results.append({
                "query": text_list[idx],
                "docs": doc_lists[idx],
                "harm_scores": check_results[idx],
                "tts_audio_base64": audio_base64
            })
        return results

    def postprocess(self, inference_output):
        return inference_output

    def write_logs(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        if len(self.request_logs) < 2:
            return

        first_start = self.request_logs[0][0]
        last_end = self.request_logs[-1][1]
        total_requests = len(self.request_logs)
        time_span = last_end - first_start
        throughput = total_requests / time_span if time_span > 0 else 0.0
        throughput_str = f"{throughput:.2f}reqps"
        root, _ = os.path.splitext(self.log_path)
        final_log_path = f"{root}_{throughput_str}.csv"

        print(f"[TorchServe] Writing {len(self.request_logs)} log entries to {final_log_path}", file=sys.stderr, flush=True)

        with open(final_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["start_time", "end_time", "latency_sec"])
            for start, end, latency in self.request_logs:
                writer.writerow([start, end, latency])

        self.__class__.request_logs.clear()