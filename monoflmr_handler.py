from ts.torch_handler.base_handler import BaseHandler
import torch
import json
import os
import time
import csv
from monoflmr import MONOFLMR
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

class MonoFLMRHandler(BaseHandler):
    # Class-level log storage
    request_logs = []
    log_threshold = 1000
    log_path = "./inference_times.csv"  # Change this if needed

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        index_root_path = "/mydata/EVQA/index/"
        index_name = "EVQA_PreFLMR_ViT-L"
        index_experiment_name = "EVQA_train_split"
        checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-L"
        image_processor_name = "openai/clip-vit-large-patch14"

        self.pipeline = MONOFLMR(
            index_root_path=index_root_path,
            index_name=index_name,
            index_experiment_name=index_experiment_name,
            checkpoint_path=checkpoint_path,
            image_processor_name=image_processor_name
        )

        self.pipeline.load_model_cpu()
        self.pipeline.load_model_gpu()

        self.initialized = True

    def preprocess(self, data):
        """
        Expects a JSON with fields:
        {
            "question_ids": [...],
            "text_sequence": [...],
            "input_ids": [[...]],
            "attention_mask": [[...]],
            "pixel_values": [[...]]
        }
        """
        input_data = data[0]["body"]
        if isinstance(input_data, (bytes, bytearray)):
            input_data = json.loads(input_data.decode("utf-8"))

        input_ids = torch.tensor(input_data["input_ids"])
        attention_mask = torch.tensor(input_data["attention_mask"])
        pixel_values = torch.tensor(input_data["pixel_values"])
        question_ids = input_data["question_ids"]
        text_sequence = input_data["text_sequence"]

        return input_ids, attention_mask, pixel_values, question_ids, text_sequence

    def inference(self, inputs):
        start_time = time.time()

        input_ids, attention_mask, pixel_values, question_ids, text_sequence = inputs
        result = self.pipeline.execFLMR(input_ids, attention_mask, pixel_values, question_ids, text_sequence)

        end_time = time.time()
        duration = end_time - start_time

        # Log this run
        self.__class__.request_logs.append((start_time, end_time, duration))
        if len(self.__class__.request_logs) >= self.__class__.log_threshold:
            self.write_logs()

        return result

    def postprocess(self, inference_output):
        return [inference_output]

    def write_logs(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        print(f"[TorchServe] Writing {len(self.request_logs)} log entries to {self.log_path}")
        with open(self.log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(["start_time", "end_time", "latency_sec"])
            for start, end, dur in self.request_logs:
                writer.writerow([start, end, dur])
        self.__class__.request_logs.clear()