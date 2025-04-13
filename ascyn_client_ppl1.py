#!/usr/bin/env python3

import os, time, json, asyncio, aiohttp, csv
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from easydict import EasyDict
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer

# ---------------------------------------------
# Config
# ---------------------------------------------
REMOTE_NODES = [
    "http://10.0.0.1:8080",
    "http://10.0.0.2:8080",
    "http://10.0.0.3:8080",
]
ENDPOINT_PATH = "/predictions/monoflmr"
HEADERS = {"Authorization": "GOJI"}  # Optional, remove if token auth is off

SEND_INTERVAL_SEC = 0.01
MAX_REQUESTS = 1000
RESULT_LOG_PATH = "/tmp/async_results.csv"

BS = 1
use_split = "train"
ds_dir = "/mydata/EVQA/EVQA_data"
image_root_dir = "/mydata/EVQA"
checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
image_processor_name = 'openai/clip-vit-large-patch14'

results_log = []

# ---------------------------------------------
# Dataset utils
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
# Async Request Logic
# ---------------------------------------------
async def send_request(session, batch_idx, payload, node_url):
    url = f"{node_url}{ENDPOINT_PATH}"
    start = time.time()
    try:
        async with session.post(url, json=payload, headers=HEADERS, timeout=60) as resp:
            end = time.time()
            latency = end - start
            status = resp.status
            success = status == 200
            if not success:
                text = await resp.text()
                print(f"[Batch {batch_idx}] {url} -> {status} -> {text}")
            results_log.append((batch_idx, node_url, status, latency, success))
    except Exception as e:
        end = time.time()
        latency = end - start
        print(f"[Batch {batch_idx}] {url} -> ERROR: {str(e)}")
        results_log.append((batch_idx, node_url, "EXCEPTION", latency, False))

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= MAX_REQUESTS:
                break

            node_url = REMOTE_NODES[batch_idx % len(REMOTE_NODES)]

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

            task = asyncio.create_task(send_request(session, batch_idx, payload, node_url))
            tasks.append(task)

            await asyncio.sleep(SEND_INTERVAL_SEC)

        await asyncio.gather(*tasks)

    # Write results to CSV
    with open(RESULT_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_idx", "node_url", "status", "latency_sec", "success"])
        writer.writerows(results_log)
    print(f"\n Results saved to {RESULT_LOG_PATH}")

# ---------------------------------------------
# Entrypoint
# ---------------------------------------------
if __name__ == "__main__":
    print("Loading model config/tokenizers...")
    flmr_config = FLMRConfig.from_pretrained(checkpoint_path)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
        checkpoint_path,
        text_config=flmr_config.text_config,
        subfolder="query_tokenizer"
    )
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)

    print("Preparing dataset...")
    ds = load_dataset('parquet', data_files={
        'train': ds_dir + '/train-00000-of-00001.parquet',
        'test': ds_dir + '/test-00000-of-00001-2.parquet',
    })[use_split].select(range(166000, 167000))

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

    asyncio.run(main())