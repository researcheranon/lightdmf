import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # only show errors
try:
    from huggingface_hub import login
except ImportError:
    login = None
import transformers
import torch
from io import BytesIO
from urllib.request import urlopen
import librosa
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
from transformers import pipeline
from packaging import version
from typing import Optional
from copy import deepcopy
from torch import Tensor
import torch.nn.functional as F
import torchaudio
import time
import utils as u
from typing import Dict


try:
    from google import genai
except ImportError:  # Optional dependency (used only by specific model wrappers)
    genai = None

try:
    from qwen_omni_utils import process_mm_info
except ImportError:  # Optional dependency (used only by Qwen3Omni)
    process_mm_info = None

transformers_version = version.parse(transformers.__version__)

# Add device detection at the top
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _maybe_hf_login() -> None:
    """Authenticate to Hugging Face only when `HF_TOKEN` is provided."""
    if login is None:
        return
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)


_maybe_hf_login()



class HFText(torch.nn.Module):

    def __init__(self, model_name, embed_only=True):
        super(HFText, self).__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is not None and self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if len(self.tokenizer) > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.embed_only = embed_only

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)
        if self.embed_only:
            # Take the mean of the last hidden layer as embedding
            hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
            return hidden_states
        else:
            probs = F.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
            prediction = self.model.config.id2label[pred_label]
            return prediction

class Qwen2Audio(torch.nn.Module):# 

    def __init__(
            self, 
            instruct=False, 
            embed_only=False, 
            no_prompt=False, 
            get_audio_tower_features=False,
            labels=None,
            average_last_n=-1
            ):
        super(Qwen2Audio, self).__init__()
        self.instruct = instruct
        self.embed_only = embed_only
        self.no_prompt = no_prompt
        self.average_last_n = average_last_n

        if self.instruct:
            model_name="Qwen/Qwen2-Audio-7B-Instruct"
        else:
            model_name="Qwen/Qwen2-Audio-7B"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = transformers.AutoProcessor.from_pretrained(model_name)
        self.target_sr = self.processor.feature_extractor.sampling_rate


        if self.device == "cpu":
            print("Warning: GPU is not available, using CPU.")
            self.model = transformers.Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name,
                low_cpu_mem_usage=True
            )
        else:
            self.model = transformers.Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
            )


        if get_audio_tower_features:
            self.model.multi_modal_projection = None
            self.model.language_model = None

        self.labels = labels
        self.emotion_prompt = "Classify the emotion of the audio."

        if labels is not None:
            self.emotion_prompt += f' You can only answer with one of the following emotions, nothing else: {", ".join(labels)}. Audio:'

    def get_sampling_rate(self):
        return self.target_sr

    # batched run
    def run(self, prompts, audio_list):
        # prompts: list of prompt strings, audio_list: list of audio arrays
        # print(audio_list[0].shape)
        
        if isinstance(prompts, str):
            prompts = [prompts] * len(audio_list)

        if self.instruct:
            texts = []
            for prompt, audio in zip(prompts, audio_list):
                conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": [{"type": "audio", "audio_path": audio}]},
                ]
                text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                texts.append(text)
        else:
            texts = [f"<|audio_bos|><|AUDIO|><|audio_eos|>{p}:" for p in prompts]

        with torch.no_grad():
            inputs = self.processor(text=texts, audio=audio_list, return_tensors="pt", padding=True, sampling_rate=self.target_sr)

            for key, value in inputs.items():
                inputs[key] = value.to(self.device)

            if self.embed_only:
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

                output = outputs.hidden_states
                output = torch.stack(output, dim=1)
                if self.average_last_n > 0:
                    output = torch.mean(output[:, -self.average_last_n:, ...], dim=1)
                return output
            else:
                generate_ids = self.model.generate(**inputs, max_new_tokens=50, output_hidden_states=True)    # output_hidden_states=True doesnt work
                trimmed = []
                for i in range(generate_ids.size(0)):
                    trimmed.append(generate_ids[i, inputs.input_ids.size(1):])
                trimmed = torch.stack(trimmed, dim=0)
                decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                return decoded

    def __call__(self, audio=None, prompt=None, text=None, sr=None):
        if not isinstance(audio, list):
            audio = [audio]
        for i in range(len(audio)):
            if isinstance(audio[i], torch.Tensor):
                audio[i] = audio[i].cpu().numpy()

        if self.no_prompt:
            input_prompt = ''
        else:
            input_prompt = prompt if prompt is not None else self.emotion_prompt


        outputs = self.run(input_prompt, audio)
        if not self.embed_only:
            outputs = [o.strip().lower() for o in outputs]
        # print(outputs)
        return outputs

    def sanity_check(self):
        audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"
        audio_bytes = BytesIO(urlopen(audio_url).read())
        audio = [librosa.load(audio_bytes, sr=self.get_sampling_rate())[0]]
        prompt = 'Explain the sound in the following audio.'
        response = self.run(prompt, audio)[0]
        print("--- Sanity check ---")
        print('Fed the model with audio of glass breaking, asked what the sound is.')
        print(f"Model response: {response}")

    # batched sanity check with different prompts
    def sanity_check_batched(self):
        audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"
        audio_bytes = BytesIO(urlopen(audio_url).read())
        audio = librosa.load(audio_bytes, sr=self.get_sampling_rate())[0]

        # duplicate audio to make batch size = 2
        audios = [audio, audio]
        prompts = [
            'Explain the first sound in the audio.',
            'Explain the second sound in the audio.'
        ]

        responses = self.run(prompts, audios)

        print("--- Batched sanity check (batch=2) ---")
        for i, r in enumerate(responses):
            print(f"Sample {i}: {r}")

    def get_audio_tower_features(self, audios):
        if not isinstance(audios, list):
            audios = [audios]
        for i in range(len(audios)):
            if isinstance(audios[i], torch.Tensor):
                audios[i] = audios[i].squeeze().cpu().numpy()
            elif isinstance(audios[i], np.ndarray):
                audios[i] = audios[i].squeeze()
            if isinstance(audios[i], np.ndarray):
                audios[i] = audios[i].astype(np.float32, copy=False)
        self.model.eval()

        texts = [f"<|audio_bos|><|AUDIO|><|audio_eos|>" for _ in audios]

        # Preprocess audio
        inputs = self.processor(
            audio=audios,
            text=texts,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.target_sr
        )

        audio_input_features_len = inputs.feature_attention_mask.sum().item()

        # From Qwen2Audio source code modeling_qwen2_audio.py: _get_feat_extract_output_lengths
        # Truncates audio output features
        audio_output_features_len = (((audio_input_features_len - 1) // 2 + 1) - 2) // 2 + 1
        # .input_features.to(self.device)

        # Forward through audio tower only
        with torch.no_grad():
            output = self.model.audio_tower(inputs.input_features.to(self.device), output_hidden_states=True, return_dict=True)  # <-- 1280-dim output
            if self.average_last_n > 0:
                hidden_states = output.hidden_states  # Tuple of (layer1, layer2, ..., layerN)
                last_n_states = hidden_states[-self.average_last_n:]  # Get the last N layers
                # Stack and average
                stacked_states = torch.stack(last_n_states, dim=0)  # Shape: (N, batch_size, seq_len, feature_dim)
                output = torch.mean(stacked_states, dim=0)  # Shape: (batch_size, seq_len, feature_dim)
            else:   # Take last hidden layer and truncate as in Qwen2 source code 
                output = output.last_hidden_state[:, :audio_output_features_len, :]
        return output


def get_output_dim(model):
    return list(model.model.named_parameters())[-1][1].shape[-1]


def n_trainable_parameters(model):
    # number of trainable parameters
    return sum([p.nelement() for p in model.parameters() if p.requires_grad])


def n_parameters(model):
    # number of total parameters
    return sum([p.nelement() for p in model.parameters()])


class Wav2Vec2Embedding(torch.nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-xls-r-2b"):
        """
        Generic Wav2Vec2 embedding class that returns hidden states for any model.
        
        Parameters
        ----------
        model_name : str
            Hugging Face model name
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Prepare model kwargs
        model_kwargs = {"output_hidden_states": True, "return_dict": True}

        # 1. Check if hardware supports Flash Attention (Pascal/GTX 1080 is 6.1, FA needs 7.0+)
        major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        hardware_supports_fa = major >= 7

        # 2. Check if transformers version supports the 'attn_implementation' argument
        # Wav2Vec2 SDPA/FA2 support was introduced in transformers v4.36.0
        transformers_supports_attn = version.parse(transformers.__version__) >= version.parse("4.36.0")

        if transformers_supports_attn:
            if hardware_supports_fa:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                # Fallback to SDPA (Scaled Dot Product Attention)
                # On GTX 1080, this uses "Memory Efficient" kernels which are supported.
                model_kwargs["attn_implementation"] = "sdpa"

        # Load feature extractor only (backbone)
        self.processor = transformers.Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = transformers.Wav2Vec2Model.from_pretrained(
            model_name, **model_kwargs, dtype=self.dtype
        ).to(self.device)

        self.default_sampling_rate = self.processor.sampling_rate
            
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        
    def get_sampling_rate(self):
        return self.default_sampling_rate

    def forward(self, audio, sr=None):
        """
        Compute embeddings from audio. Always returns last_hidden_state, even for CTC models.
        
        Parameters
        ----------
        sampling_rate : int, optional

        Returns
        -------
        torch.Tensor
            Shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim] if averaged
        """

        if sr is None:
            sr = self.get_sampling_rate()

        if len(audio.shape) == 1:
            if isinstance(audio, np.ndarray):
                audio = np.expand_dims(audio, axis=0)
            else:
                audio = audio.unsqueeze(0)

        # Convert first dimension to list
        audio = [audio[i] for i in range(audio.shape[0])]

        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        # both processor and model wants to add channel dimension, so we squeeze it here
        inputs["input_values"] = inputs["input_values"].squeeze(1)
        
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_state = outputs.hidden_states[-1].to(torch.float32)

        return hidden_state
    

class Qwen3Text(torch.nn.Module):

    def __init__(self, model_type, instruct=False):
        super(Qwen3Text, self).__init__()

        self.instruct = instruct
        self.instruction = 'Classify the emotion.'

        model_name = f"Qwen/{model_type}"
        library = transformers.AutoModel
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.config = transformers.AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.config.return_dict_in_generate = True
        self.model = library.from_pretrained(
            model_name, 
            config=self.config,
            device_map="auto",
            torch_dtype="auto",
            )
        self.model.eval()
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def __call__(self, text):
        with torch.no_grad():

            if self.instruct:
                if isinstance(text, list):
                    text = [f"Instruction: {self.instruction}\nQuery: {t}" for t in text]
                else:
                    text = f"Instruction: {self.instruction}\nQuery: {text}"

            model_device = next(self.model.parameters()).device
            x = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            x = {k: v.to(model_device) for k, v in x.items()}
            y = self.model(**x)
            # Get features
            features = y.hidden_states[-1]
            return features



class Whisper(torch.nn.Module):
    ENCODER_FRAMES_PER_SEC: float = 50.0  # 50 encoder frames per second

    def __init__(self, model_name, no_cuda=False, chunk_length_s=30, debug=False, quantize_8bit=True):
        super().__init__()

        self.debug = debug
        self.device = torch.device('cuda') if torch.cuda.is_available() and not no_cuda else torch.device('cpu')
        self.model_name = model_name
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.chunk_length_s = chunk_length_s  # chunk length in seconds

        if model_name == 'distil_whisper':
            self.model_name = 'distil-whisper/distil-large-v3'
        else:
            self.model_name = f'openai/{model_name}'

        self.model = transformers.WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(self.device)


        # quantize linear layers
        if quantize_8bit and self.device.type == "cpu":
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        self.model.eval()

        self.processor = transformers.WhisperProcessor.from_pretrained(self.model_name)
        self.decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).to(self.device)
        self.decoder_attention_mask = torch.ones_like(self.decoder_input_ids).to(self.device)

    def __call__(self, audio=None, sr=16000):
        if audio.ndim == 2:
            audio = audio.mean(0)

        if self.debug:
            # Only use the first 5 seconds for debugging
            max_length = int(5 * sr)
            audio = audio[:max_length]

        # Resample if needed
        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(torch.tensor(audio))
        else:
            audio = torch.tensor(audio)

        # Split into chunks
        chunk_size = self.chunk_length_s * sr
        num_chunks = math.ceil(audio.shape[0] / chunk_size)

        # all_segments: list[dict] = []       # {"start", "end", "text"} per segment
        all_encoder_hiddens: list[torch.Tensor] = []   # last encoder hidden state per segment
        # full_text = []
        all_texts = []
        all_times = []

        for i in range(num_chunks):
            chunk_start_s = i * self.chunk_length_s   # wall-clock offset for this chunk
            chunk_audio = audio[int(i * chunk_size): int((i + 1) * chunk_size)]
            if len(chunk_audio) == 0:
                continue

            inputs = self.processor(
                audio=chunk_audio,
                sampling_rate=16000,
                return_tensors="pt",
                # return_attention_mask=True,
                # truncation=False,
                # padding="longest",
                # language='en'
            )
            input_features = inputs.input_features.to(self.device).to(self.dtype)
            # attention_mask = inputs.attention_mask.to(self.device).to(self.dtype)

            # with torch.no_grad():
            with torch.inference_mode():
                gen_out = self.model.generate(
                    input_features,
                    max_length=448,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    return_timestamps=True,
                )
                # whisper pads all segments to 30 seconds anyway so they have same hiddens
                segmented_output = gen_out['segments'][0]
                hidden = segmented_output[0]['result']['encoder_hidden_states'][-1].squeeze(0)

                segment_texts = [self.processor.tokenizer.decode(seg['tokens'], skip_special_tokens=True).strip() for seg in segmented_output]
                start_times = [float(seg['start'].item()) for seg in segmented_output]
                end_time = float(segmented_output[-1]['end'].item())
                times = start_times + [end_time]
                indices = [int(s * self.ENCODER_FRAMES_PER_SEC) for s in times]
                # end_index = int(end_time * self.ENCODER_FRAMES_PER_SEC)
                # indices.append(end_index)  # add end index for last segment
                # split hidden states into segments based on start indices
                encoder_hiddens = [hidden[indices[k]: indices[k + 1]] for k in range(len(indices) - 1)]
        
                all_texts += segment_texts
                all_encoder_hiddens += encoder_hiddens

                global_start_times = [chunk_start_s + s for s in start_times]
                global_end_time = chunk_start_s + end_time
                all_times += global_start_times
        all_times.append(global_end_time)
        return {
            "text": all_texts,
            "audio_feature": all_encoder_hiddens,
            "text_feature": None,
            "times": all_times
        }



class MSDynamicGate(nn.Module):
    """
    Multiscale pooling with input-dependent dynamic gating (attention across scales, per channel per timestep).
    Kernel sizes:
    12 frames (240 ms) -> syllable-to-short-word prosodic structure,
        local pitch movement, energy modulation, and stress patterns (Speech Communication; Scherer).
    24 frames (480 ms) -> multi-syllabic and short phrase-level contours,
         pitch trajectories, rhythm, and local tempo variations (Pattern Recognition; El Ayadi et al.).
    48 frames (960 ms) -> short phrase-level prosody and sustained trends in loudness and pitch baseline, 
        stable affective state (Speech Communication; Scherer).
    """

    def __init__(self, target_len=128, scales=(12, 24, 48)):
        super().__init__()
        self.target_len = target_len
        self.scales = scales
        self.num_scales = len(scales)
        self.embed_dim = 384

        # tiny linear layer to compute per-channel weights across scales
        self.gate_linear = nn.Linear(self.num_scales, self.num_scales)


    def forward(self, x):
        """
        x: (L, D)
        returns: (target_len, D)

        Scale 1: kernel=12, stride=12, out_len=125
        Scale 2: kernel=24, stride=12, out_len=124
        Scale 3: kernel=48, stride=12, out_len=128

        """
        B, L, D = x.shape  # Batch size, sequence length, feature dimension
        x_t = x.permute(0, 2, 1)  # (B, D, L)

        pooled_scales = []
        for kernel_len in self.scales:
            stride = max(1, math.ceil((L - kernel_len) / (self.target_len - 1)))
            pooled = F.avg_pool1d(x_t, kernel_size=kernel_len, stride=stride, padding=0)
            out_len = pooled.size(2)
            if out_len < self.target_len:
                pooled = F.pad(pooled, (0, self.target_len - out_len))
            elif out_len > self.target_len:
                pooled = pooled[:, :, :self.target_len]
            pooled_scales.append(pooled)  # (B, D, target_len)

        # Stack scales: (B, D, target_len, N)
        stacked = torch.stack(pooled_scales, dim=-1)  # (B, D, target_len, N)
        stacked = stacked.permute(0, 2, 1, 3)  # (B, target_len, D, N)

        # Compute input-dependent weights per timestep and channel
        weights = self.gate_linear(stacked)  # (B, target_len, D, N)
        weights = F.softmax(weights, dim=-1)  # softmax over scales

        # Weighted sum
        out = (stacked * weights).sum(dim=-1)  # (B, target_len, D)
        return out


class AttentionClassifier(nn.Module):
    """
    Multi-modal classifier with:
      - audio self-attention
      - text self-attention with sentinels
      - optional audio-to-text cross-attention

    Audio is always present.
    Text may be missing (handled via sentinels).

    Fusion modes:
      - no_cross=True:
            fusion = [audio, text]
      - no_cross=False, no_residuals=False:
            fusion = [audio, text, cross]
      - no_cross=False, no_residuals=True:
            fusion = [cross]
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        d_model: int,
        n_classes: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_audio_len: int = 200,
        max_text_len: int = 200,
        whisper_embedding_len: int = -1,
        single_scale: bool = False,
        shortcut_attention: bool = False,
        nonlinearity: Optional[str] = None,
    ):
        super().__init__()

        self.audio_dim = modality_dims["audio"]
        self.text_dim = modality_dims.get("text", 0)
        if self.text_dim is None:
            self.text_dim = 0
        self.use_text = self.text_dim > 0

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.n_classes = n_classes
        self.whisper_embedding_len = whisper_embedding_len
        self.single_scale = single_scale
        self.shortcut_attention = shortcut_attention
        self.nonlinearity = nonlinearity

        
        self.model_config = {
            "modality_dims": modality_dims,
            "d_model": d_model,
            "n_classes": n_classes,
            "dropout": dropout,
            "n_heads": n_heads,
            "max_audio_len": max_audio_len,
            "max_text_len": max_text_len,
            "whisper_embedding_len": whisper_embedding_len,
            "single_scale": single_scale,
            "shortcut_attention": shortcut_attention,
            "nonlinearity": nonlinearity,
        }


        # -----------------------
        # Averaging
        # -----------------------
        if whisper_embedding_len > 0:
            self.averaging_module = MSDynamicGate(target_len=whisper_embedding_len)

        # -----------------------
        # Sentinels
        # -----------------------
        self.audio_present = nn.Parameter(torch.randn(1, 1, self.audio_dim))
        self.audio_missing = nn.Parameter(torch.randn(1, 1, self.audio_dim))

        if self.use_text:
            self.text_present = nn.Parameter(torch.randn(1, 1, self.text_dim))
            self.text_missing = nn.Parameter(torch.randn(1, 1, self.text_dim))
        else:
            self.register_parameter("text_present", None)
            self.register_parameter("text_missing", None)

        # -----------------------
        # Encoders
        # -----------------------
        self.audio_enc = nn.Linear(self.audio_dim, d_model)
        self.text_enc = nn.Linear(self.text_dim, d_model) if self.use_text else None

        self.audio_ln = nn.LayerNorm(d_model)
        self.text_ln = nn.LayerNorm(d_model) if self.use_text else None

        self.dropout = nn.Dropout(dropout)

        # -----------------------
        # Attention projections
        # -----------------------
        self.audio_q_self = nn.Linear(d_model, d_model)
        self.audio_kv = nn.Linear(d_model, 2 * d_model)

        self.audio_q = nn.Linear(d_model, d_model)

        if self.use_text:
            self.text_q = nn.Linear(d_model, d_model)
            self.text_kv = nn.Linear(d_model, 2 * d_model)
        else:
            self.text_q = None

        # -----------------------
        # Heads
        # -----------------------
        self.audio_head = nn.Linear(d_model, n_classes)
        self.text_head = nn.Linear(d_model, n_classes) if self.use_text else None

        if self.use_text:
            fusion_in_dim = 3 * d_model
            self.fusion_head = nn.Linear(fusion_in_dim, n_classes)
        else:
            self.fusion_head = None

    # --------------------------------------------------
    # Attention helpers
    # --------------------------------------------------
    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def _attention(self, q, k, v, mask: Optional[torch.Tensor] = None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = scores.softmax(dim=-1)
        return torch.matmul(attn, v)

    def _apply_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlinearity is None:
            return x
        activation_name = self.nonlinearity.lower()
        if activation_name == "relu":
            return F.relu(x)
        if activation_name == "gelu":
            return F.gelu(x)
        raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}")
    
    def save_model(self, path):
        state = {
            'model': self.state_dict(),
            'config': self.model_config
        }
        torch.save(state, path)
        print("Model saved to:", path)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, input_: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}

        audio_in = input_["audio_feature"]  # [B, Ta, Da]
        has_text = input_["has_text"]       # [B]

        B, _, _ = audio_in.shape
        has_audio = input_.get("has_audio")
        if has_audio is None:
            has_audio = torch.ones(B, device=audio_in.device, dtype=torch.bool)
        if self.use_text:
            text_in = input_["text_feature"]    # [B, Tt, Dt]
            if text_in.dim() == 2:
                text_in = text_in.unsqueeze(1)
            _, Tt, _ = text_in.shape

        # =======================
        # AUDIO SELF-ATTENTION
        # =======================
        if self.whisper_embedding_len > 0:
            audio_out = torch.full(
            (B, self.whisper_embedding_len, audio_in.size(-1)),
            float("nan"),
            device=audio_in.device,
            dtype=audio_in.dtype,
            )
            if has_audio.any():
                audio_out[has_audio] = self.averaging_module(audio_in[has_audio])
                audio_in = audio_out

        _, Ta, _ = audio_in.shape

        sentinels = self.audio_present.expand(B, 1, self.audio_dim).clone()
        sentinels[~has_audio] = self.audio_missing

        audio_padded = torch.zeros_like(audio_in)
        audio_padded[has_audio] = audio_in[has_audio]

        audio_cat = torch.cat([sentinels, audio_padded], dim=1)
        a_enc = self.audio_ln(self.audio_enc(audio_cat))

        audio_mask = torch.ones(B, Ta + 1, device=audio_in.device, dtype=torch.bool)
        audio_mask[~has_audio, 1:] = False

        a_q = self.audio_q_self(a_enc)
        a_k, a_v = self.audio_kv(a_enc).chunk(2, dim=-1)

        a_out = self._merge_heads(
            self._attention(
                self._split_heads(a_q),
                self._split_heads(a_k),
                self._split_heads(a_v),
                audio_mask,
            )
        )
        a_out = self.dropout(a_out)
        a_out = self._apply_nonlinearity(a_out)
        a_feat = a_out.mean(dim=1)

        outputs["audio"] = self.audio_head(a_feat)

        if not self.use_text or not has_text.all().item():
            text_logits = torch.full((B, self.n_classes), float("nan"), device=audio_in.device, dtype=audio_in.dtype)
            fusion_logits = torch.full((B, self.n_classes), float("nan"), device=audio_in.device, dtype=audio_in.dtype)
            outputs["text"] = text_logits
            outputs["fusion"] = fusion_logits
            return outputs

        # =======================
        # TEXT SELF-ATTENTION
        # =======================
        sentinels = self.text_present.expand(B, 1, self.text_dim).clone()
        sentinels[~has_text] = self.text_missing

        text_padded = torch.zeros_like(text_in)
        text_padded[has_text] = text_in[has_text]

        text_cat = torch.cat([sentinels, text_padded], dim=1)
        t_enc = self.text_ln(self.text_enc(text_cat))

        text_mask = torch.ones(B, Tt + 1, device=text_in.device, dtype=torch.bool)
        text_mask[~has_text, 1:] = False

        t_q = self.text_q(t_enc)
        t_k, t_v = self.text_kv(t_enc).chunk(2, dim=-1)

        t_out = self._merge_heads(
            self._attention(
                self._split_heads(t_q),
                self._split_heads(t_k),
                self._split_heads(t_v),
                text_mask,
            )
        )
        t_out = self.dropout(t_out)
        t_out = self._apply_nonlinearity(t_out)
        t_feat = t_out.mean(dim=1)

        outputs["text"] = self.text_head(t_feat)

        # =======================
        # CROSS ATTENTION (optional)
        # =======================
        a_cross_in = a_enc if self.shortcut_attention else a_out
        t_cross_in = t_enc if self.shortcut_attention else t_out

        a_qc = self.audio_q(a_cross_in)

        t_kc, t_vc = self.text_kv(t_cross_in).chunk(2, dim=-1)

        c_out = self._merge_heads(
            self._attention(
                self._split_heads(a_qc),
                self._split_heads(t_kc),
                self._split_heads(t_vc),
                text_mask,
            )
        )
        c_out = self.dropout(c_out)
        c_out = self._apply_nonlinearity(c_out)
        c_feat = c_out.mean(dim=1)

        # =======================
        # FUSION HEAD
        # =======================
        fusion_feat = torch.cat([a_feat, t_feat, c_feat], dim=-1)
        outputs["fusion"] = self.fusion_head(fusion_feat)

        return outputs


class AveragingClassifier(nn.Module):
    """
    Multi-modal classifier that averages features over time.

    - No self- or cross-attention.
    - One sentinel per modality, only used when missing.
    - Fusion concatenates audio and text only when text is enabled.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        d_model: int,
        n_classes: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_audio_len: int = 200,
        max_text_len: int = 200,
        whisper_embedding_len: int = -1,
        shortcut_attention: bool = False,
        nonlinearity: Optional[str] = None,
    ):
        super().__init__()

        self.audio_dim = modality_dims["audio"]
        self.text_dim = modality_dims.get("text", 0)
        if self.text_dim is None:
            self.text_dim = 0
        self.use_text = self.text_dim > 0

        self.d_model = d_model
        self.n_classes = n_classes
        self.nonlinearity = nonlinearity

        self.model_config = {
            "modality_dims": modality_dims,
            "d_model": d_model,
            "n_classes": n_classes,
            "dropout": dropout,
            "n_heads": n_heads,
            "max_audio_len": max_audio_len,
            "max_text_len": max_text_len,
            "whisper_embedding_len": whisper_embedding_len,
            "shortcut_attention": shortcut_attention,
            "nonlinearity": nonlinearity,
        }

        # -----------------------
        # Sentinels (only used when modality is missing)
        # -----------------------
        self.audio_missing = nn.Parameter(torch.randn(1, self.audio_dim))
        if self.use_text:
            self.text_missing = nn.Parameter(torch.randn(1, self.text_dim))
        else:
            self.register_parameter("text_missing", None)

        # -----------------------
        # Encoders
        # -----------------------
        self.audio_enc = nn.Linear(self.audio_dim, d_model)
        self.text_enc = nn.Linear(self.text_dim, d_model) if self.use_text else None

        self.audio_ln = nn.LayerNorm(d_model)
        self.text_ln = nn.LayerNorm(d_model) if self.use_text else None

        self.dropout = nn.Dropout(dropout)

        # -----------------------
        # Heads
        # -----------------------
        self.audio_head = nn.Linear(d_model, n_classes)
        self.text_head = nn.Linear(d_model, n_classes) if self.use_text else None

        if self.use_text:
            fusion_in_dim = 2 * d_model
            self.fusion_head = nn.Linear(fusion_in_dim, n_classes)
        else:
            self.fusion_head = None

    def _apply_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlinearity is None:
            return x
        activation_name = self.nonlinearity.lower()
        if activation_name == "relu":
            return F.relu(x)
        if activation_name == "gelu":
            return F.gelu(x)
        raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}")

    def save_model(self, path):
        state = {
            "model": self.state_dict(),
            "config": self.model_config,
        }
        torch.save(state, path)
        print("Model saved to:", path)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, input_: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}

        audio_in = input_["audio_feature"]  # [B, Ta, Da]
        has_text = input_["has_text"]       # [B]

        B, _, _ = audio_in.shape
        has_audio = input_.get("has_audio")
        if has_audio is None:
            has_audio = torch.ones(B, device=audio_in.device, dtype=torch.bool)
        if self.use_text:
            text_in = input_["text_feature"]  # [B, Tt, Dt]
            if text_in.dim() == 2:
                text_in = text_in.unsqueeze(1)

        audio_mean = audio_in.mean(dim=1)
        if (~has_audio).any():
            audio_mean[~has_audio] = self.audio_missing

        a_feat = self.audio_ln(self.audio_enc(audio_mean))
        a_feat = self.dropout(a_feat)
        a_feat = self._apply_nonlinearity(a_feat)
        outputs["audio"] = self.audio_head(a_feat)

        if not self.use_text:
            text_logits = torch.full((B, self.n_classes), float("nan"), device=audio_in.device, dtype=audio_in.dtype)
            fusion_logits = torch.full((B, self.n_classes), float("nan"), device=audio_in.device, dtype=audio_in.dtype)
            outputs["text"] = text_logits
            outputs["fusion"] = fusion_logits
            return outputs

        text_mean = text_in.mean(dim=1)
        if (~has_text).any():
            text_mean[~has_text] = self.text_missing

        t_feat = self.text_ln(self.text_enc(text_mean))
        t_feat = self.dropout(t_feat)
        t_feat = self._apply_nonlinearity(t_feat)
        outputs["text"] = self.text_head(t_feat)

        fusion_feat = torch.cat([a_feat, t_feat], dim=-1)
        outputs["fusion"] = self.fusion_head(fusion_feat)

        return outputs
    


