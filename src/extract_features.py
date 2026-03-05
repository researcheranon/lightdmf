print('Run started')
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["HF_HUB_DISABLE_MMAP"] = "1"

from models import (
    ASTEmbeddingModel,
    Wav2Vec2Embedding,
    WavLMEmbeddingModel,
    Qwen3Text,
    CLAPEmbeddingModel,
    Qwen2Audio,
    Qwen3TextInstruct,
    HFText,
    Whisper
)
import utils as u
from dataset import RawDataset, safe_collate, GoEmotionsDataset
import shutil
from sentence_transformers import SentenceTransformer
import time



debug =  u.parse_debug()

average_last_n = 1

datasets = [
    'GoEmotions',
    'IEMOCAP', 
    'MELD', 
    'CASE',
    'CREMA_D',    # no utterance
    'RAVDESS',    # no utterance
    'TESS',     # no utterance
    ]

if debug:
    print("""
    =================================
    WARNING: DEBUG MODE IS ENABLED!
    NO OUTPUT FILES WILL BE SAVED.
    =================================
    """.upper())

# Settings
SPLITS = [
    "train", 
    "dev", 
    "test"
    ]

FEATURES = [
    # "distil_whisper",
    # "llama",
    # 'qwen2_audio_tower',
    # 'minilm',
    # 'wav2vec2_xls'
    # 'Qwen3-0.6B',
    ]



BATCH_SIZE = 1

target_srs = {
    'qwen2_audio_tower': 16000,
    'wav2vec2_xls': 16000,
    'qwen3_text': None,
    'minilm': None,
    'Qwen3-Embedding-0.6B': None,
    'llama': None,
    'distil_whisper': 16000,
}


min_seconds = {

    'qwen2_audio_tower': -1, 
    'qwen3_text': -1, 
    'wav2vec2_xls': -1, 
    'whisper_tiny': -1, 
    'Qwen3-Embedding-0.6B': -1, 
    'llama': -1, 
    'distil_whisper': -1,
    'minilm': -1,

}

max_seconds = {
    'qwen2_audio_tower': -1, 
    'wav2vec2_xls': -1, 
    'Qwen3-Embedding-0.6B': -1, 
    'llama': -1, 
    'distil_whisper': -1,
    'minilm': -1,
}

def extract_audio_chunks(audio, sr, min_seconds=-1, max_seconds=30):
    """Extract chunks from audio tensor with optional minimum length padding"""
    min_len = int(sr * min_seconds) if min_seconds > 0 else 0
    chunk_len = int(sr * max_seconds)

    # If audio is shorter than minimum length, pad it
    if audio.shape[-1] < min_len and min_len > 0:
        padding = min_len - audio.shape[-1]
        audio = torch.nn.functional.pad(audio, (0, padding))

    # If audio fits within chunk length, return as single chunk
    if audio.shape[-1] <= chunk_len:
        return [audio]
    
    # Split into chunks
    chunks = []
    for start in range(0, audio.shape[-1], chunk_len):
        end = min(start + chunk_len, audio.shape[-1])
        chunk = audio[..., start:end]
        
        # Pad final chunk if it's too short and minimum length is required
        if chunk.shape[-1] < min_len and min_len > 0:
            padding = min_len - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, padding))
        
        chunks.append(chunk)
    
    return chunks

def save_feature(out_path, arr):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)

def fix_cp1252(s):
    cp1252_map = {"\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x95": "•", "\x96": "-", "\x97": "—", "\x85": "..."}
    return "".join(cp1252_map.get(c, c) for c in s)


for instruct in [False]:

    cache_dir = Path('../../../.cache')
    hf_dir = cache_dir / 'huggingface' / 'hub'
    if hf_dir.exists():
        downloaded_hf_models = hf_dir.glob('*')
    else:
        downloaded_hf_models = []


    for feature in FEATURES:
        current_model_dirname = f"models--Qwen--{feature}"

        # Only initialize one model at a time
        
        if feature == 'llama':
            model = HFText('meta-llama/Llama-3.2-1B', embed_only=True).eval()
        elif "qwen3" in feature.lower():
            model = Qwen3Text(feature).eval()
        elif feature == "qwen2_audio_tower":
            model = Qwen2Audio(instruct=False, average_last_n=1, no_prompt=False, embed_only=True, get_audio_tower_features=True)
        elif feature == "wav2vec2_xls":
            model = Wav2Vec2Embedding(model_name="facebook/wav2vec2-xls-r-2b").eval()
        elif feature == "minilm":
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        elif 'whisper' in feature:
            model = Whisper(feature, no_cuda=False).eval()

        no_audio = feature in ['minilm', 'llama'] or 'qwen3' in feature.lower()

        for dataset_name in datasets:
            
            METADATA_PATH = f"../../../../../datasets/public/{dataset_name}/metadata_split.csv"
            DATA_BASE = Path(METADATA_PATH).parent

            OUT_DIR = DATA_BASE / "features" / f"{feature}"

            OUT_DIR.mkdir(exist_ok=True, parents=True)
            
            split = 'all'

            target_sr = target_srs.get(feature, None)

            if dataset_name == 'GoEmotions':
                dataset = GoEmotionsDataset()
            else:
                dataset = RawDataset(METADATA_PATH, split, target_sr, target_length_seconds=-1, start_at_beginning=True, min_duration=0.2, no_audio=no_audio)
            loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=safe_collate)

            elapsed_running_average = 0
            total_audio_duration = 0
            total_time = 0

            for i, batch in enumerate(tqdm(loader, desc=f"Extracting {dataset_name} - {feature}", mininterval=120)):
                if batch is None:  # skip completely invalid batch (returned by collate function)
                    continue
                    
                features_dict, label_batch, files_batch = batch
                
                if True:
                
                    audio_batch = features_dict.get("audio_raw", None)
                    text_batch = features_dict.get("text_raw", None)

                    with torch.no_grad():
                        batch_features = []
                        batch_features_text = []
                        batch_features_audio = []
                        t0 = time.time()
                        if feature in ['qwen2_audio_tower', 'wav2vec2_xls'] or 'whisper' in feature:
                            input_ = audio_batch
                            
                            # Process each sample in the batch
                            for sample_idx in range(input_.shape[0]):
                                audio_sample = input_[sample_idx]  # Shape: [audio_length]
                                if min_seconds[feature] > 0 or max_seconds[feature] > 0:
                                    # Extract chunks from the audio sample

                                    chunks = extract_audio_chunks(
                                        audio_sample, 
                                        target_srs[feature], 
                                        min_seconds=min_seconds[feature], 
                                        max_seconds=max_seconds[feature]
                                    )
                                else:
                                    chunks = [audio_sample]
                                
                                # Process each chunk through the model
                                chunk_features = []

                                # whisper extracts both audio and text features
                                chunk_features_audio = []
                                chunk_features_text = []
                                for chunk in chunks:
                                    # Add batch dimension for model input
                                    # Dataloader already resamples to TARGET_SR
                                    if feature == 'qwen2_audio_tower':
                                        chunk_feat = model.get_audio_tower_features(chunk)
                                    else:
                                        chunk_feat = model(audio=chunk)   # Shape: [seq_len, feat_dim]
                                    total_audio_duration += chunk.shape[0] / 16000

                                    if 'whisper' in feature:
                                        chunk_feat = chunk_feat['audio_feature']

                                    chunk_feat = chunk_feat.squeeze(0).cpu().numpy()

                                    chunk_features.append(chunk_feat)  # Shape: [seq_len, feat_dim]
                                
                                # Concatenate all chunk features along sequence dimension
                                if len(chunk_features) > 1:
                                    chunk_features = torch.cat(chunk_features, dim=0)  # Shape: [total_seq_len, feat_dim]
                                else:
                                    chunk_features = chunk_features[0]
                                if torch.is_tensor(chunk_features):
                                    chunk_features = chunk_features.cpu().numpy()

                                batch_features.append(chunk_features)

                        elif feature == "llama" or 'qwen3' in feature.lower():
                            feats = model(text=text_batch).squeeze(0).to(torch.float32).cpu().numpy()
                            batch_features.append(feats)

                        elif feature == 'minilm':
                            feats = model.encode(text_batch, output_value='token_embeddings')[0].cpu().numpy()
                            batch_features.append(feats)

                        for j, file_id in enumerate(files_batch):
                            out_path = OUT_DIR / f"{file_id}.npy"
                            
                        if not debug:
                            save_feature(out_path, batch_features[j]) 
                        else:
                            print('Output shape:', batch_features[j].shape)

                        t1 = time.time()

                        total_time += t1-t0
                        if feature in ['qwen2_audio_tower', 'wav2vec2_xls', 'distil_whisper']:
                            elapsed_running_average = total_time / total_audio_duration 

        del model
    print("Feature extraction complete.")
