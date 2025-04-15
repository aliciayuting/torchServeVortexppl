#!/usr/bin/env python3
import logging
logging.disable(logging.CRITICAL)

import json
import numpy as np
import os
from typing import Any
import threading
import sys
import faiss
import pickle
import torch

import zipfile

from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from torch.nn.utils.rnn import pad_sequence

from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr.utils.load_utils import extract_fbank
from FlagEmbedding import FlagModel

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SenseVoice"))
from SenseVoice.utils.frontend import WavFrontend, WavFrontendOnline
from SenseVoice.model import SenseVoiceSmall

from transformers import BartTokenizer, BartForSequenceClassification


class AudioRecognition:
    def __init__(self, device_name, model_dir, language="en"):
        '''
        language: "zh", "en", "yue", "ja", "ko", "nospeech","auto"
        '''
        self.language = language
        self.model_dir = model_dir # "iic/SenseVoiceSmall"
        self.model = None
        self.device_name = device_name
        self.kwargs = None
        self.frontend = None

    def load_model(self):
        self.model, self.kwargs = SenseVoiceSmall.from_pretrained(model=self.model_dir, device=self.device_name)
        self.model.eval()
        self.kwargs["data_type"] = "fbank"
        self.kwargs["sound"] = "fbank"
        self.frontend = self.kwargs["frontend"]
        print("Speech to Text model loaded")
        
    def exec_model(self, batch_audios):
        if self.model is None:
            self.load_model()
        speech, speech_lengths = extract_fbank(
            batch_audios, data_type=self.kwargs.get("data_type", "sound"), frontend=self.frontend
        )
        res = self.model.inference(
            data_in=speech,
            data_lengths=speech_lengths,
            language=self.language, 
            use_itn=False,
            ban_emo_unk=True,
            **self.kwargs,
        )
        text_list = []
        for idx in range(len(res[0])):
            text_list.append(rich_transcription_postprocess(res[0][idx]["text"]))
        return text_list

    def exec_model_with_batch_size(self, batch_audios, batch_size=4):
        batched_exec_audios = []
        for i in range(0, len(batch_audios), batch_size):
            batched_exec_audios.append(batch_audios[i:i + batch_size])
        text_list = []
        for batch in batched_exec_audios:
            text_list.extend(self.exec_model(batch))
        return text_list



class TextEncoder:
    def __init__(self, device: str, model_name: str):
        self.encoder = None
        self.device = device
        self.model_name = model_name
        
    def load_model(self):
        self.encoder = FlagModel(self.model_name, devices=self.device)
        print("Text Encoder model loaded")

    def encoder_exec(self, query_list: list[str]) -> np.ndarray:
        # Generate embedding dimesion of 384
        if self.encoder is None:
            self.load_model()
        result =  self.encoder.encode(query_list)
        return result

class FaissSearcher:
    def __init__(self, device: str, index_dir: str, topk: int = 5):
        self.cpu_index = None
        self.res = None
        self.gpu_index = None
        self.device = device
        self.index_dir = index_dir
        self.topk = topk
        
    def load_model(self):
        self.cpu_index = faiss.read_index(self.index_dir)
        self.res = faiss.StandardGpuResources()
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        self.gpu_index.nprobe = 10
        print("Faiss index loaded")

    def searcher_exec(self, embeddings: np.ndarray) -> np.ndarray:
        if self.gpu_index is None:
            self.load_model()
        _, I = self.gpu_index.search(embeddings, self.topk)
        return I



class DocumentLoader:
    def __init__(self, doc_dir):
        self.doc_dir = doc_dir
        self.doc_list = None
        
    def load_docs(self):
        with open(self.doc_dir , 'rb') as file:
            self.doc_list = pickle.load(file)
        print("Document list loaded")


    def get_doc_list(self, doc_ids_list) -> list[list[str]]:
        '''
        doc_ids_list: list of list of doc_ids
        '''
        if self.doc_list is None:
            self.load_docs()
        doc_lists = []
        for doc_ids in doc_ids_list:
            cur_docs = []
            for doc_id in doc_ids:
                cur_docs.append(self.doc_list[doc_id])
            doc_lists.append(cur_docs)
        return doc_lists



class SearchRetriever:
    def __init__(self, device: str, index_dir: str, topk: int = 5, doc_dir: str = None):
        self.doc_loader = DocumentLoader(doc_dir)
        self.searcher = FaissSearcher(device, index_dir, topk)
        self.loaded_model = False        

    def load_model(self):
        self.searcher.load_model()
        self.doc_loader.load_docs()
        self.loaded_model = True

    def search_docs(self, embeddings: np.ndarray) -> list[list[str]]:
        if not self.loaded_model:
            self.load_model()
        I = self.searcher.searcher_exec(embeddings)
        # convert I to list of list of doc_ids
        doc_list_ids = []
        for i in range(I.shape[0]):
            doc_list_ids.append(I[i,:].tolist())
        # get doc_list from doc_loader
        doc_lists = self.doc_loader.get_doc_list(doc_list_ids)
        return doc_lists



class TextChecker:
    def __init__(self, device: str, model_name: str):
        self.model = None
        self.tokenizer = None
        self.device = device
        self.model_name = model_name
        self.hypothesis = "harmful."
        
    def load_model(self):
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        print("Text Check model loaded")

    def model_exec(self, batch_premise: list[str]) -> np.ndarray:
        '''
        batch_premise: list of text strings
        
        return: list of integer. probability in % that the content is harmful
        '''
        if self.model is None:
            self.load_model()
        try:
            inputs = self.tokenizer(batch_premise,
                        [self.hypothesis] * len(batch_premise),
                        return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                result = self.model(**inputs)
            logits = result.logits
            entail_contradiction_logits = logits[:, [0, 2]]  # entailment = index 2
            probs = entail_contradiction_logits.softmax(dim=1)
            true_probs = probs[:, 1] * 100  # entailment probability
            true_probs = [int(x) for x in true_probs]
            return true_probs
        except Exception as e:
            import traceback
            print(f"[TEXTCHECKER] ERROR during model_exec: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return [-1 for _ in batch_premise]  # or raise to fail hard
    
    def docs_check(self, doc_list: list[list[str]]) -> list[list[int]]:
        flattened_doc_list = [item for sublist in doc_list for item in sublist]
        types = self.model_exec(flattened_doc_list)
        # Reshape the types to match the original doc_list structure
        reshaped_types = []
        start = 0
        for sublist in doc_list:
            end = start + len(sublist)
            reshaped_types.append(types[start:end])
            start = end
        return reshaped_types
    


class TTSRunner:
    def __init__(self, device: str):
        self.fastpitch = None
        self.FASTPITCH_NAME = "nvidia/tts_en_fastpitch"
        self.hifigan = None
        self.HIFIGAN_NAME = "nvidia/tts_hifigan"
        self.device = device
        
    def load_model(self):
        self.fastpitch = FastPitchModel.from_pretrained(self.FASTPITCH_NAME).to(self.device).eval()
        self.hifigan = HifiGanModel.from_pretrained(model_name=self.HIFIGAN_NAME).to(self.device).eval()
        print("TTS model loaded")
        
    def run_tts(self, batch_texts: list[str]) -> np.ndarray:
        if self.fastpitch is None:
            self.load_model()
        token_list = [self.fastpitch.parse(text).squeeze(0) for text in batch_texts]
        tokens = pad_sequence(token_list, batch_first=True).to(self.device)
        with torch.no_grad():
            spectrograms = self.fastpitch.generate_spectrogram(tokens=tokens)
            audios = self.hifigan.convert_spectrogram_to_audio(spec=spectrograms)
        np_audios = audios.cpu().numpy()
        return np_audios
    
    def model_exec(self, batch_docs:list[list[str]]) -> list[list[np.ndarray]]:
        flattened_doc_list = [item for sublist in batch_docs for item in sublist]
        tts_audios = self.run_tts(flattened_doc_list)
        # Reshape the audios to match the original doc_list structure
        reshaped_audios = []
        start = 0
        for sublist in batch_docs:
            end = start + len(sublist)
            reshaped_audios.append(tts_audios[start:end])
            start = end
        return reshaped_audios