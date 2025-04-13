#!/usr/bin/env python3
import numpy as np
import os, time, json
import torch
from PIL import Image
import requests
from datasets import load_dataset
from easydict import EasyDict
from transformers import AutoImageProcessor
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer

from torch.utils.data import DataLoader

# ---------------------------------------------
# Helper functions (same as your script)
# ---------------------------------------------
def prepare_text_sequence(sample):
    sample = EasyDict(sample)

    module = EasyDict(
        {"type": "QuestionInput", "option": "default", "separation_tokens": {"start": "", "end": ""}}
    )

    instruction = sample.instruction.strip()
    if instruction[-1] != ":":
        instruction = instruction + ":"
    instruction = instruction.replace(":", flmr_config.mask_instruction_token)

    text_sequence = " ".join(
        [instruction]
        + [module.separation_tokens.start]
        + [sample.question]
        + [module.separation_tokens.end]
    )

    sample["text_sequence"] = text_sequence
    return sample

def tokenize_inputs(examples, query_tokenizer, image_processor):
    encoding = query_tokenizer(examples["text_sequence"])
    examples["input_ids"] = encoding["input_ids"]
    examples["attention_mask"] = encoding["attention_mask"]

    pixel_values = []
    for img_path in examples["img_path"]:
        if img_path is None:
            image = Image.new("RGB", (336, 336), color='black')
        else:
            image = Image.open(img_path).convert("RGB")
        encoded = image_processor(image, return_tensors="pt")
        pixel_values.append(encoded.pixel_values)

    pixel_values = torch.stack(pixel_values, dim=0)
    examples["pixel_values"] = pixel_values
    return examples

def add_path_prefix_in_img_path(example, prefix):
    if example["img_path"] is not None:
        example["img_path"] = os.path.join(prefix, example["img_path"])
    return example

# ---------------------------------------------
# TorchServe Client Inference
# ---------------------------------------------
if __name__ == "__main__":
    BS = 1
    num_batches = 1000
    torchserve_url = "http://localhost:8080/predictions/monoflmr"

    checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
    image_processor_name = 'openai/clip-vit-large-patch14'
    ds_dir = "/mnt/nvme0/vortex_pipeline1/EVQA_data"
    image_root_dir = "/mnt/nvme0/vortex_pipeline1"
    use_split = "train"

    flmr_config = FLMRConfig.from_pretrained(checkpoint_path)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path,
                                                                 text_config=flmr_config.text_config,
                                                                 subfolder="query_tokenizer")
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)

    ds = load_dataset('parquet', data_files={
        'train': ds_dir + '/train-00000-of-00001.parquet',
        'test': ds_dir + '/test-00000-of-00001-2.parquet',
    })[use_split].select([i for i in range(166000, 167000)])

    ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": image_root_dir})
    ds = ds.map(prepare_text_sequence)
    ds = ds.map(
        tokenize_inputs,
        fn_kwargs={"query_tokenizer": query_tokenizer, "image_processor": image_processor},
        batched=True,
        batch_size=16,
        num_proc=8,
    )

    ds.set_format(type="torch",
                  columns=["input_ids", "attention_mask", "pixel_values", "text_sequence", "question_id", "question"])

    loader = DataLoader(
        ds,
        batch_size=BS,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        input_ids = batch["input_ids"].tolist()
        attention_mask = batch["attention_mask"].tolist()
        pixel_values = batch["pixel_values"].tolist()
        text_sequence = batch["question"]
        question_ids = batch["question_id"]

        payload = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "text_sequence": text_sequence,
            "question_ids": question_ids
        }

        try:
            response = requests.post(torchserve_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"[Batch {batch_idx}] Top results: {json.dumps(result, indent=2)}")
            else:
                print(f"[Batch {batch_idx}] Failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[Batch {batch_idx}] Error: {str(e)}")

        time.sleep(0.05)