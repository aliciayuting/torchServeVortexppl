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

model_dir = os.path.dirname(os.path.abspath(__file__))
sensevoice_path = os.path.join(model_dir, "SenseVoice")
if sensevoice_path not in sys.path:
    sys.path.append(sensevoice_path)

FAISS_INDEX_DIR = "/mydata/msmarco/msmarco_pq.index"
DOC_DIR = "/mydata/msmarco/msmarco_3_clusters/doc_list.pkl"

class SpeechPipelineHandler(BaseHandler):
    request_logs = []
    log_threshold = 1000
    log_path = "/users/TY373/workspace/torchServeVortexppl/ppl2/inference_times.csv"  # Customize per node if needed (e.g., ./n0/inference_times.csv)

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.loaded_models = False

    def initialize(self, ctx):
        print("[TorchServe] Handler initializing...", flush=True)
        self.device = "cuda"

        self.sensevoice_model_name = "iic/SenseVoiceSmall"
        self.faiss_index_path = FAISS_INDEX_DIR
        self.doc_pickle_path = DOC_DIR
        self.text_check_model = "facebook/bart-large-mnli"

        self.speech_model = AudioRecognition(self.device, self.sensevoice_model_name)
        self.text_encoder = self.speech_model  # or a separate encoder
        self.search_retriever = SearchRetriever(self.device, self.faiss_index_path, topk=5, doc_dir=self.doc_pickle_path)
        self.text_checker = TextChecker(self.device, self.text_check_model)
        self.tts_runner = TTSRunner(self.device)

        self.speech_model.load_model()
        self.search_retriever.load_model()
        self.text_checker.load_model()
        self.tts_runner.load_model()

        self.loaded_models = True
        self.initialized = True
        print("~~~~~~~~~~~~~~~~ Speech handler initialized ~~~~~~~~~~~~`", flush=True)


    def preprocess(self, data):
        try:
            start_time = time.time()  #Start timer here

            input_data = data[0].get("body")
            if isinstance(input_data, (bytes, bytearray)):
                input_data = input_data.decode("utf-8")
            input_json = json.loads(input_data)
            audio_batch = input_json["audio_data"]

            audio_list = [np.array(waveform, dtype=np.float32) for waveform in audio_batch]
            return audio_list, start_time  # Pass start_time to inference
        except Exception as e:
            raise RuntimeError(f"[Preprocess] Error parsing input audio: {e}")

    def inference(self, inputs):
        audio_list, start_time = inputs  # Use start_time from preprocess

        if not self.loaded_models:
            self.initialize(None)

        # Inference begins
        text_list = self.speech_model.exec_model(audio_list)
        embeddings = self.text_encoder.model.encode(text_list)
        doc_lists = self.search_retriever.search_docs(embeddings)
        check_results = self.text_checker.docs_check(doc_lists)
        audio_outputs = self.tts_runner.model_exec(doc_lists)

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