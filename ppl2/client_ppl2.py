#!/usr/bin/env python3

import os, time, json, asyncio, aiohttp, csv, pickle
import numpy as np
from torch.utils.data import DataLoader

# ---------------------------------------------
# Config
# ---------------------------------------------
REMOTE_NODES = [
    "http://10.10.1.1:8080",
    "http://10.10.1.2:8080",
    "http://10.10.1.3:8080",
    "http://10.10.1.4:8080",
]
ENDPOINT_PATH = "/predictions/monospeech"
HEADERS = {"Authorization": "GOJI"}  # Optional, remove if token auth is off

SEND_INTERVAL_SEC = 0.01
MAX_REQUESTS = 10
RESULT_LOG_PATH = "/tmp/async_audio_results.csv"

BS = 1
AUDIO_PKL_PATH = "/mydata/msmarco/queries_audio5000.pkl"

results_log = []

# ---------------------------------------------
# Dataset utils
# ---------------------------------------------
def get_audio_data():
    with open(AUDIO_PKL_PATH, "rb") as f:
        waveforms = pickle.load(f)
    list_np_waveform = []
    for i, item in enumerate(waveforms):
        if len(item[-1]) > 200000:
            continue
        list_np_waveform.append(item[-1])
    return list_np_waveform

def collate_numpy(batch):
    return batch  # return list of np.ndarrays

# ---------------------------------------------
# Async Request Logic
# ---------------------------------------------
async def send_request(session, batch_idx, payload, node_url):
    url = f"{node_url}{ENDPOINT_PATH}"
    start = time.time()
    try:
        async with session.post(url, json=payload, headers=HEADERS, timeout=1200) as resp:
            end = time.time()
            latency = end - start
            status = resp.status
            success = status == 200
            if not success:
                text = await resp.text()
                print(f"[Batch {batch_idx}] {url} -> {status} -> {text}")
            # else:
            #     text = await resp.text()
            #     try:
            #         parsed = json.loads(text)
            #         print(f"[Batch {batch_idx}] {url} -> {status} -> {parsed}")
            #     except json.JSONDecodeError:
            #         print(f"[Batch {batch_idx}] {url} -> {status} (non-JSON) -> {text}")
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

            question_ids = [batch_idx * BS + i for i in range(len(batch))]
            audio_data = [arr.tolist() for arr in batch]

            payload = {
                "question_ids": question_ids,
                "audio_data": audio_data
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
    print(f"\nResults saved to {RESULT_LOG_PATH}")

# ---------------------------------------------
# Entrypoint
# ---------------------------------------------
if __name__ == "__main__":
    print("Loading audio waveform data...")
    audio_np_list = get_audio_data()

    loader = DataLoader(
        audio_np_list,
        batch_size=BS,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate_numpy
    )

    asyncio.run(main())