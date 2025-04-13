from ts.torch_handler.base_handler import BaseHandler
import torch
import json
import os
from monoflmr import MONOFLMR  # Make sure monoflmr.py is in the same dir or use relative import
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

class MonoFLMRHandler(BaseHandler):

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        index_root_path = "/mydata/EVQA/index/"
        index_name = "EVQA_PreFLMR_ViT-L"  # can also pass via ctx or env
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
        input_ids, attention_mask, pixel_values, question_ids, text_sequence = inputs
        return self.pipeline.execFLMR(input_ids, attention_mask, pixel_values, question_ids, text_sequence)

    def postprocess(self, inference_output):
        return [inference_output]