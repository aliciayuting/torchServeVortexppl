import sys
import os
import io
import json
import torch
import time
import csv
import numpy as np
from ts.torch_handler.base_handler import BaseHandler
from model_pipeline import AudioRecognition, SearchRetriever, TextChecker, TTSRunner

# Add SenseVoice to path if needed
model_dir = os.path.dirname(os.path.abspath(__file__))
sensevoice_path = os.path.join(model_dir, "SenseVoice")
if sensevoice_path not in sys.path:
    sys.path.append(sensevoice_path)

class SpeechPipelineHandler(BaseHandler):
    request_logs = []  # class-level log buffer
    log_threshold = 1000
    log_path = os.path.join(model_dir, "inference_times.csv")  # change if needed

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.loaded_models = False

    def initialize(self, ctx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Paths to model and index resources
        self.sensevoice_dir = os.path.join(model_dir, "SenseVoice")
        self.faiss_index_path = os.path.join(model_dir, "data", "index.faiss")
        self.doc_pickle_path = os.path.join(model_dir, "data", "doc_list.pkl")
        self.text_check_model = "facebook/bart-large-mnli"

        # Load models
        self.speech_model = AudioRecognition(device_name=self.device, model_dir=self.sensevoice_dir)
        self.text_encoder = self.speech_model  # Assuming encoder is same as speech model or extend as needed
        self.search_retriever = SearchRetriever(device, self.faiss_index_path, topk=5, doc_dir=self.doc_pickle_path)
        self.text_checker = TextChecker(device, self.text_check_model)
        self.tts_runner = TTSRunner(device)

        self.speech_model.load_model()
        self.search_retriever.load_model()
        self.text_checker.load_model()
        self.tts_runner.load_model()

        self.loaded_models = True
        self.initialized = True
        print("[TorchServe] Handler initialized.")

    def preprocess(self, data):
        """
        Expects input format:
        {
            "audio_batch": [
                [0.1, -0.2, 0.3, ...],   # list of float32 samples
                ...
            ]
        }
        """
        try:
            input_data = data[0].get("body")
            if isinstance(input_data, (bytes, bytearray)):
                input_data = input_data.decode("utf-8")
            input_json = json.loads(input_data)
            audio_batch = input_json["audio_batch"]

            audio_list = [np.array(waveform, dtype=np.float32) for waveform in audio_batch]
            return audio_list
        except Exception as e:
            raise RuntimeError(f"[Preprocess] Error parsing input audio: {e}")

    def inference(self, audio_list):
        if not self.loaded_models:
            self.initialize(None)

        start_time = time.time()

        text_list = self.speech_model.exec_model(audio_list)
        embeddings = self.text_encoder.model.encode(text_list)  # Replace with actual encoder logic
        doc_lists = self.search_retriever.search_docs(embeddings)
        check_results = self.text_checker.docs_check(doc_lists)
        audio_outputs = self.tts_runner.model_exec(doc_lists)

        end_time = time.time()
        latency = end_time - start_time

        self.__class__.request_logs.append((start_time, end_time, latency))
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
        print(f"[TorchServe] Writing {len(self.request_logs)} log entries to {self.log_path}")
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["start_time", "end_time", "latency_sec"])
            for start, end, latency in self.request_logs:
                writer.writerow([start, end, latency])
        self.__class__.request_logs.clear()