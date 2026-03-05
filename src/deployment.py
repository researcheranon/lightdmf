from __future__ import annotations

import argparse
import time
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import contextlib
import logging
import os
import sys
import threading
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from models import Qwen3Text, Qwen2Audio, Wav2Vec2Embedding, HFText, AttentionClassifier, AveragingClassifier, Whisper


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("offline")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import transformers

    transformers.logging.set_verbosity_error()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Stage profiler
# ---------------------------------------------------------------------------

class StageProfiler:
    """Thread-safe accumulator for per-stage wall-clock times.

    Usage::

        profiler = StageProfiler()
        with profiler.track("my_stage"):
            do_something()
        profiler.report()
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._totals = defaultdict(float)
        self._counts = defaultdict(int)

    @contextlib.contextmanager
    def track(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            with self._lock:
                self._totals[stage] += elapsed
                self._counts[stage] += 1

    def report(self, title: str = "Stage profiling summary") -> str:
        lines = [f"\n{'─'*52}", f"  {title}", f"{'─'*52}"]
        if not self._totals:
            lines.append("  (no data)")
        else:
            total_wall = sum(self._totals.values())
            lines.append(f"  {'Stage':<26} {'Calls':>5}  {'Total(s)':>9}  {'Avg(ms)':>9}  {'%':>6}")
            lines.append(f"  {'─'*26}  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*6}")
            for stage, total in sorted(self._totals.items(), key=lambda x: -x[1]):
                n = self._counts[stage]
                pct = 100.0 * total / total_wall if total_wall > 0 else 0.0
                avg_ms = 1000.0 * total / n if n > 0 else 0.0
                lines.append(f"  {stage:<26}  {n:>5}  {total:>9.3f}  {avg_ms:>9.1f}  {pct:>5.1f}%")
            lines.append(f"  {'─'*26}  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*6}")
            lines.append(f"  {'TOTAL':<26}         {total_wall:>9.3f}")
        lines.append(f"{'─'*52}\n")
        return "\n".join(lines)

    def get_stage_time(self, stage: str) -> float:
        return float(self._totals.get(stage, 0.0))

    def get_total_time(self) -> float:
        return float(sum(self._totals.values()))


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


class Inference:
    def __init__(
        self,
        checkpoint_dir: Path,
        file_path: str | None = None,
        no_cuda: bool = True,
        time_enabled: bool = True,
        classifier=None,
        config=None,
        model_config=None,
        idx_to_emotion=None,
        sr: int = 16000,
        ignore_text: bool = False,
        num_workers: int = 1,
        text_batch_size: int | None = None,
        truncate_len: float = None,
        debug: bool = False,
        save_memory: bool = True
    ):
        self.debug = debug
        self.checkpoint_dir = Path(checkpoint_dir)
        self.file_path = file_path
        self.no_cuda = no_cuda
        self.time_enabled = time_enabled
        self.sr = sr
        self.ignore_text = ignore_text
        self.num_workers = max(1, int(num_workers))
        self.text_batch_size = text_batch_size
        self.truncate_len = truncate_len
        self.save_memory = save_memory

        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self.profiler = StageProfiler()

        self.cfg = config
        self.model_cfg = model_config
        self.idx_to_emotion = idx_to_emotion
        self.classifier = classifier
        self.text_model = None
        self.audio_model = None
        self.whisper = None
        self.n_text_features = None
        self.text_model_name = None
        self.audio_model_name = None
        self._whisper_is_audio_model = False

        print('Initializing...', flush=True, end=" ")
        self._load_models()

    def _move_to_device(self, obj):
        if obj is None:
            return None
        if hasattr(obj, "to"):
            try:
                obj = obj.to(self.device)
            except TypeError:
                try:
                    obj.to(self.device)
                except Exception:
                    pass
        return obj

    def _load_models(self):
        if self.cfg is None:
            checkpoint_path = Path(self.checkpoint_dir)
            config_path = checkpoint_path / "config.pth"
            self.cfg = torch.load(config_path, weights_only=False)
            model_path = checkpoint_path / "model.pth"
            model = torch.load(model_path, weights_only=False, map_location=torch.device("cpu"))
            state_dict = model["model"]
            self.model_cfg = model["config"]
            self.idx_to_emotion = self.cfg["idx_to_emotion"]
        else:
            if self.classifier is None:
                raise ValueError("classifier must be provided when config is passed in")
            if self.model_cfg is None:
                self.model_cfg = getattr(self.classifier, "model_config", None)
            if self.model_cfg is None:
                raise ValueError("model_config must be provided when using an injected classifier")
            if self.idx_to_emotion is None:
                self.idx_to_emotion = self.cfg.get("idx_to_emotion")
            if self.idx_to_emotion is None:
                raise ValueError("idx_to_emotion must be provided when config is passed in")
            state_dict = None

        self.text_model_name = self.cfg["text_model"].lower()
        self.audio_model_name = self.cfg["audio_model"].lower()
        self._whisper_is_audio_model = "whisper" in self.audio_model_name
        self._use_whisper_for_segments = not self._whisper_is_audio_model

        if self._whisper_is_audio_model:
            self.text_model = self._init_text_model(self.text_model_name)
            self.text_model = self._move_to_device(self.text_model)

        if "qwen3" in self.text_model_name:
            self.n_text_features = self.cfg["n_qwen3_features"]
            assert self.n_text_features == self.model_cfg["modality_dims"]["text"], (
                f"Expected text feature dim {self.model_cfg['modality_dims']['text']} but got {self.n_text_features}"
            )
        else:
            self.n_text_features = self.model_cfg["modality_dims"]["text"]

        use_whisper_text_flag = True

        if self._whisper_is_audio_model:
            self.whisper = self._init_whisper(use_whisper_text_flag)
            self.whisper = self._move_to_device(self.whisper)
            self.audio_model = self._init_audio_model()
            self.audio_model = self._move_to_device(self.audio_model)
        else:
            self.audio_model = self._init_audio_model()
            self.audio_model = self._move_to_device(self.audio_model)
            if not self.save_memory:
                self.whisper = self._init_whisper(use_whisper_text_flag)
                self.whisper = self._move_to_device(self.whisper)
                if not self.ignore_text and self.text_model_name != "none":
                    self.text_model = self._init_text_model(self.text_model_name)
                    self.text_model = self._move_to_device(self.text_model)

        if self.classifier is None:
            self.classifier = self._init_classifier(state_dict)
        else:
            self.classifier = self._move_to_device(self.classifier)
            self.classifier.eval()

    def _init_text_model(self, text_model_name: str):
        if text_model_name == "none":
            return None
        if "qwen3" in text_model_name:
            return Qwen3Text(text_model_name)
        if text_model_name == "minilm":
            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(self.device))
        if text_model_name == "roberta":
            return HFText("cardiffnlp/twitter-roberta-base-sentiment-latest", embed_only=True)
        if text_model_name == "llama":
            return HFText("meta-llama/Llama-3.2-1B", embed_only=True).eval()
        raise ValueError(f"Unsupported text_model: {text_model_name}")

    def _init_whisper(self, use_whisper_text_flag: bool):
        if use_whisper_text_flag:
            if self._whisper_is_audio_model:
                whisper_name = self.cfg["audio_model"]
            else:
                whisper_name = "distil_whisper"
            return Whisper(model_name=whisper_name, no_cuda=self.no_cuda)
        return None

    def _ensure_whisper_loaded(self):
        if self.whisper is None:
            self.whisper = self._init_whisper(True)
            self.whisper = self._move_to_device(self.whisper)

    def _ensure_audio_model_loaded(self):
        if self.audio_model is None:
            self.audio_model = self._init_audio_model()
            self.audio_model = self._move_to_device(self.audio_model)

    def _ensure_text_model_loaded(self):
        if self.text_model is None:
            self.text_model = self._init_text_model(self.text_model_name)
            self.text_model = self._move_to_device(self.text_model)

    def _release_whisper(self):
        if self.whisper is not None:
            self.whisper = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _release_audio_model(self):
        if self.audio_model is not None:
            self.audio_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _release_text_model(self):
        if self.text_model is not None:
            self.text_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _iter_text_batches(self, texts: list[str], batch_size: int | None):
        if batch_size is None or batch_size <= 0 or batch_size >= len(texts):
            yield texts
            return
        for start in range(0, len(texts), batch_size):
            yield texts[start:start + batch_size]

    def _encode_text_batch(self, texts: list[str], batch_size: int | None) -> list[torch.Tensor]:
        if not texts:
            return []
        if self.text_model is None:
            return [torch.full((1, 1), float("nan")) for _ in texts]

        features: list[torch.Tensor] = []
        for batch in self._iter_text_batches(texts, batch_size):
            if self.text_model_name == "minilm":
                batch_feats = self.text_model.encode(
                    batch, output_value="token_embeddings", show_progress_bar=False
                )
                for feat in batch_feats:
                    text_feat = torch.from_numpy(feat) if isinstance(feat, np.ndarray) else feat
                    text_feat = text_feat.unsqueeze(0).to(torch.float32)
                    if self.n_text_features < text_feat.shape[2]:
                        text_feat = text_feat[:, :, : self.n_text_features]
                    features.append(text_feat)
            else:
                batch_feats = self.text_model(batch)
                if isinstance(batch_feats, np.ndarray):
                    batch_feats = torch.from_numpy(batch_feats)
                if isinstance(batch_feats, torch.Tensor) and batch_feats.dim() >= 3:
                    for i in range(batch_feats.shape[0]):
                        text_feat = batch_feats[i:i + 1].to(torch.float32)
                        if self.n_text_features < text_feat.shape[2]:
                            text_feat = text_feat[:, :, : self.n_text_features]
                            if "qwen3" in self.text_model_name:
                                text_feat = F.normalize(text_feat, p=2, dim=2)
                        features.append(text_feat)
                elif isinstance(batch_feats, list):
                    for feat in batch_feats:
                        text_feat = torch.from_numpy(feat) if isinstance(feat, np.ndarray) else feat
                        text_feat = text_feat.to(torch.float32)
                        if text_feat.dim() == 2:
                            text_feat = text_feat.unsqueeze(0)
                        if self.n_text_features < text_feat.shape[2]:
                            text_feat = text_feat[:, :, : self.n_text_features]
                            if "qwen3" in self.text_model_name:
                                text_feat = F.normalize(text_feat, p=2, dim=2)
                        features.append(text_feat)
                else:
                    raise ValueError("Unsupported text model output format for batched encoding")

        return features

    @staticmethod
    def _pad_sequence_list(features: list[torch.Tensor]) -> torch.Tensor:
        if not features:
            return torch.empty(0)
        cleaned = []
        max_len = 0
        for feat in features:
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            if feat.dim() == 3 and feat.shape[0] == 1:
                feat = feat.squeeze(0)
            cleaned.append(feat)
            max_len = max(max_len, feat.shape[0])

        batch = []
        for feat in cleaned:
            pad_len = max_len - feat.shape[0]
            if pad_len > 0:
                feat = F.pad(feat, (0, 0, 0, pad_len))
            batch.append(feat)
        return torch.stack(batch, dim=0)

    def _classify_batch(
        self,
        audio_features: list[torch.Tensor],
        texts: list[str],
        text_features: list[torch.Tensor] | None,
        seg_lens_samples: list[int],
    ) -> list[dict]:
        if not audio_features:
            return []

        audio_batch = self._pad_sequence_list(audio_features).to(self.device, dtype=torch.float32)

        if text_features is None or self.ignore_text or self.text_model_name == "none":
            text_batch = torch.full((len(audio_features), 1, 1), float("nan"), device=self.device)
            has_text = torch.zeros(len(audio_features), device=self.device, dtype=torch.bool)
        else:
            text_batch = self._pad_sequence_list(text_features).to(self.device, dtype=torch.float32)
            has_text = torch.ones(len(audio_features), device=self.device, dtype=torch.bool)

        has_audio = torch.ones(len(audio_features), device=self.device, dtype=torch.bool)

        with self.profiler.track("classifier"):

            outputs = self.classifier(
                {
                    "audio_feature": audio_batch,
                    "text_feature": text_batch,
                    "has_audio": has_audio,
                    "has_text": has_text,
                }
            )

        logits_audio = outputs["audio"]
        logits_text = outputs["text"]
        logits_fusion = outputs["fusion"]

        prob_audio = F.softmax(logits_audio, dim=1).detach().cpu().numpy()
        prob_text = F.softmax(logits_text, dim=1).detach().cpu().numpy()
        prob_fusion = F.softmax(logits_fusion, dim=1).detach().cpu().numpy()

        results = []
        for idx, text in enumerate(texts):
            audio_emotion_index = int(np.argmax(prob_audio[idx]))
            text_emotion_index = int(np.argmax(prob_text[idx]))
            fusion_emotion_index = int(np.argmax(prob_fusion[idx]))

            audio_emotion_label = self.idx_to_emotion.get(audio_emotion_index, f"class_{audio_emotion_index}")
            text_emotion_label = self.idx_to_emotion.get(text_emotion_index, f"class_{text_emotion_index}")
            fusion_emotion_label = self.idx_to_emotion.get(fusion_emotion_index, f"class_{fusion_emotion_index}")

            audio_confidence = float(np.max(prob_audio[idx]))
            text_confidence = float(np.max(prob_text[idx]))
            fusion_confidence = float(np.max(prob_fusion[idx]))

            results.append(
                {
                    "text": text,
                    "audio_emotion_label": audio_emotion_label,
                    "text_emotion_label": text_emotion_label,
                    "fusion_emotion_label": fusion_emotion_label,
                    "audio_confidence": audio_confidence,
                    "text_confidence": text_confidence,
                    "fusion_confidence": fusion_confidence,
                    "processing_duration": float("nan"),
                    "end_to_end_latency": float("nan"),
                    "audio_len_sec": seg_lens_samples[idx] / self.sr,
                }
            )

        return results

    def _init_audio_model(self):
        if "wav2vec2" in self.audio_model_name:
            model_name = "facebook/wav2vec2-xls-r-2b"
            return Wav2Vec2Embedding(model_name=model_name).eval()
        if self.audio_model_name == "qwen2_audio_tower":
            return Qwen2Audio(
                instruct=False,
                average_last_n=1,
                no_prompt=False,
                embed_only=True,
                get_audio_tower_features=True,
            )
        if "whisper" in self.audio_model_name:
            return None
        raise ValueError(f"Unsupported audio_model: {self.cfg['audio_model']}")

    def _init_classifier(self, state_dict):
        if self.cfg["fusion_method"] == "attention":
            clf = AttentionClassifier(
                modality_dims=self.model_cfg["modality_dims"],
                d_model=self.model_cfg["d_model"],
                n_classes=self.model_cfg["n_classes"],
                n_heads=self.model_cfg["n_heads"],
                dropout=self.model_cfg.get("dropout", self.cfg.get("dropout", 0.0)),
                max_audio_len=self.model_cfg.get("max_audio_len", 200),
                max_text_len=self.model_cfg.get("max_text_len", 200),
                whisper_embedding_len=self.model_cfg.get("whisper_embedding_len", -1),
                single_scale=self.model_cfg.get("single_scale", self.cfg.get("single_scale", False)),
                shortcut_attention=self.model_cfg.get("shortcut_attention", False),
                nonlinearity=self.model_cfg.get("nonlinearity", None),
            ).to(self.device)
        elif self.cfg["fusion_method"] == "average":
            clf = AveragingClassifier(
                modality_dims=self.model_cfg["modality_dims"],
                d_model=self.model_cfg["d_model"],
                n_classes=self.model_cfg["n_classes"],
                dropout=self.cfg["dropout"],
            ).to(self.device)
        else:
            raise ValueError(f"Unknown fusion method: {self.cfg['fusion_method']}")

        if state_dict is not None:
            clf.load_state_dict(state_dict, strict=False)
        clf.eval()
        return clf

    def run_audio(self, file_path: str | None = None, num_workers: int | None = None) -> list[dict]:
        file_path = file_path or self.file_path
        if not file_path:
            raise ValueError("file_path must be provided")
        num_workers = self.num_workers if num_workers is None else max(1, int(num_workers))

        log_path = self.checkpoint_dir / "inference_time.txt" if self.time_enabled else None
        if self.time_enabled and log_path is not None and log_path.parent.exists():
            with log_path.open("w", encoding="utf-8") as log_file:
                with contextlib.redirect_stdout(_Tee(sys.stdout, log_file)), contextlib.redirect_stderr(_Tee(sys.stderr, log_file)):
                    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
                    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
                    logger.addHandler(file_handler)
                    try:
                        return self._run_audio(file_path, num_workers)
                    finally:
                        logger.removeHandler(file_handler)
                        file_handler.close()

        return self._run_audio(file_path, num_workers)

    def _run_audio(self, file_path: str, num_workers: int) -> list[dict]:
        if not self.time_enabled:
            logger.disabled = True

        run_start_time = time.time()

        audio, _ = librosa.load(file_path, sr=self.sr)

        if self.truncate_len is not None:
            print(f"Truncating audio to first {self.truncate_len} seconds for debugging.", flush=True)
            audio = audio[: int(self.sr * self.truncate_len)]

        results = self.run_segment_list(audio, num_workers=num_workers)

        run_end_time = time.time()
        total_elapsed = run_end_time - run_start_time
        audio_length_sec = len(audio) / max(self.sr, 1)

        print("\nRun summary:", flush=True)
        print(f"Text model: {self.cfg['text_model']}", flush=True)
        print(f"Audio model: {self.cfg['audio_model']}", flush=True)
        print(f"Audio length (s): {audio_length_sec:.2f}", flush=True)
        print(f"Total time (s): {total_elapsed:.2f}", flush=True)
        if not self._whisper_is_audio_model:
            whisper_time = self.profiler.get_stage_time("whisper_full_pass")
            non_whisper_time = max(0.0, total_elapsed - whisper_time)
            print(f"Total time excl. Whisper (s): {non_whisper_time:.2f}", flush=True)
        if audio_length_sec > 0:
            print(f"Rate (sec/sec): {total_elapsed / audio_length_sec:.3f}", flush=True)
        print(f"Segments processed: {len(results)}", flush=True)

        print(self.profiler.report(), flush=True)

        print("\nFinished.", flush=True)

        return results

    def run_segment_list(self, audio: np.ndarray, num_workers: int = 1) -> list[dict]:
        num_workers = max(1, int(num_workers))

        if self._whisper_is_audio_model:
            results_ordered = self._run_whisper_path(audio, num_workers)
        else:
            results_ordered = self._run_whisper_guided_audio_path(audio, num_workers)

        results = []
        for result in results_ordered:
            if result is None:
                continue
            results.append(result)
            msg = 'Transcript: ' + result["text"]
            msg += (
                '\nPredicted emotions - ' +
                f"Audio: {result['audio_emotion_label']:9s} {result['audio_confidence']*100:2.0f}% - "
                f"Text: {result['text_emotion_label']:9s} {result['text_confidence']*100:2.0f}% - "
                f"Fusion: {result['fusion_emotion_label']:9s} {result['fusion_confidence']*100:2.0f}% \n"
            )
            print(msg, flush=True)

        return results

    def _run_whisper_path(self, audio: np.ndarray, num_workers: int) -> list[dict | None]:
        self._ensure_whisper_loaded()
        if self.whisper is None:
            raise ValueError("Whisper model is required but not initialized.")

        with self.profiler.track("whisper_full_pass"):
            whisper_out = self.whisper(audio=audio, sr=self.sr)

        texts: list[str] = whisper_out["text"]
        audio_feats: list[torch.Tensor] = whisper_out["audio_feature"]

        if not texts:
            print("No speech segments detected.", flush=True)
            return []

        seg_lens_samples = []
        for audio_feat in audio_feats:
            if audio_feat is None or (isinstance(audio_feat, torch.Tensor) and audio_feat.numel() == 0):
                seg_lens_samples.append(0)
                continue
            n_frames = audio_feat.shape[0] if audio_feat.dim() >= 1 else 1
            seg_len_samples = int(n_frames / self.whisper.ENCODER_FRAMES_PER_SEC * self.sr)
            seg_lens_samples.append(seg_len_samples)

        text_features = None
        if not self.ignore_text and self.text_model_name != "none":
            if self.save_memory and not self._whisper_is_audio_model:
                self._ensure_text_model_loaded()
            with self.profiler.track("text_encode"):
                text_features = self._encode_text_batch(texts, self.text_batch_size)
            if self.save_memory and not self._whisper_is_audio_model:
                self._release_text_model()

        return self._classify_batch(audio_feats, texts, text_features, seg_lens_samples)

    def _run_whisper_guided_audio_path(self, audio: np.ndarray, num_workers: int) -> list[dict | None]:
        if self.save_memory and not self._whisper_is_audio_model:
            self._ensure_audio_model_loaded()
            if self.audio_model is None:
                raise ValueError("Audio model is required but not initialized.")

            audio_feat_full = self._encode_full_audio_features(audio)
            audio_feat_full = self._standardize_audio_feature_dims(audio_feat_full)
            if audio_feat_full is None or audio_feat_full.numel() == 0:
                return []

            audio_duration_sec = len(audio) / max(self.sr, 1)
            if audio_duration_sec <= 0:
                return []
            frames_per_sec = audio_feat_full.shape[0] / audio_duration_sec

            self._release_audio_model()

            self._ensure_whisper_loaded()
            if self.whisper is None:
                raise ValueError("Whisper model is required but not initialized.")

            with self.profiler.track("whisper_full_pass"):
                whisper_out = self.whisper(audio=audio, sr=self.sr)

            texts: list[str] = whisper_out.get("text", [])
            times: list[float] | None = whisper_out.get("times")

            if not texts:
                print("No speech segments detected.", flush=True)
                return []
            if not times or len(times) < len(texts) + 1:
                raise ValueError("Whisper did not return valid 'times' for segment slicing.")

            self._release_whisper()

            audio_features = []
            seg_lens_samples = []
            for seg_idx, _ in enumerate(texts):
                start_time = float(times[seg_idx])
                end_time = float(times[seg_idx + 1])
                start_idx = int(max(0.0, start_time * frames_per_sec))
                end_idx = int(max(start_idx + 1, end_time * frames_per_sec))
                start_idx = min(start_idx, audio_feat_full.shape[0])
                end_idx = min(end_idx, audio_feat_full.shape[0])
                audio_feat = audio_feat_full[start_idx:end_idx]
                if audio_feat is None or audio_feat.numel() == 0:
                    audio_feat = torch.empty(0)
                audio_features.append(audio_feat)
                seg_lens_samples.append(int(max(0.0, end_time - start_time) * self.sr))

            text_features = None
            if not self.ignore_text and self.text_model_name != "none":
                self._ensure_text_model_loaded()
                with self.profiler.track("text_encode"):
                    text_features = self._encode_text_batch(texts, self.text_batch_size)
                self._release_text_model()

            return self._classify_batch(audio_features, texts, text_features, seg_lens_samples)

        self._ensure_whisper_loaded()
        if self.whisper is None:
            raise ValueError("Whisper model is required but not initialized.")

        with self.profiler.track("whisper_full_pass"):
            whisper_out = self.whisper(audio=audio, sr=self.sr)

        texts: list[str] = whisper_out.get("text", [])
        times: list[float] | None = whisper_out.get("times")

        if not texts:
            print("No speech segments detected.", flush=True)
            return []
        if not times or len(times) < len(texts) + 1:
            raise ValueError("Whisper did not return valid 'times' for segment slicing.")

        self._ensure_audio_model_loaded()
        if self.audio_model is None:
            raise ValueError("Audio model is required but not initialized.")

        audio_feat_full = self._encode_full_audio_features(audio)
        audio_feat_full = self._standardize_audio_feature_dims(audio_feat_full)
        if audio_feat_full is None or audio_feat_full.numel() == 0:
            return []

        audio_duration_sec = len(audio) / max(self.sr, 1)
        if audio_duration_sec <= 0:
            return []
        frames_per_sec = audio_feat_full.shape[0] / audio_duration_sec

        audio_features = []
        seg_lens_samples = []
        for seg_idx, _ in enumerate(texts):
            start_time = float(times[seg_idx])
            end_time = float(times[seg_idx + 1])
            start_idx = int(max(0.0, start_time * frames_per_sec))
            end_idx = int(max(start_idx + 1, end_time * frames_per_sec))
            start_idx = min(start_idx, audio_feat_full.shape[0])
            end_idx = min(end_idx, audio_feat_full.shape[0])
            audio_feat = audio_feat_full[start_idx:end_idx]
            if audio_feat is None or audio_feat.numel() == 0:
                audio_feat = torch.empty(0)
            audio_features.append(audio_feat)
            seg_lens_samples.append(int(max(0.0, end_time - start_time) * self.sr))

        text_features = None
        if not self.ignore_text and self.text_model_name != "none":
            self._ensure_text_model_loaded()
            with self.profiler.track("text_encode"):
                text_features = self._encode_text_batch(texts, self.text_batch_size)

        return self._classify_batch(audio_features, texts, text_features, seg_lens_samples)


    def _encode_full_audio_features(self, audio: np.ndarray):
        self._ensure_audio_model_loaded()
        with torch.no_grad():
            if self.audio_model_name == "qwen2_audio_tower":
                with self.profiler.track("audio_encode"):
                    return self.audio_model.get_audio_tower_features(audio)
            with self.profiler.track("audio_encode"):
                return self.audio_model(audio=audio, sr=self.sr)

    @staticmethod
    def _standardize_audio_feature_dims(audio_feat):
        if audio_feat is None:
            return None
        if isinstance(audio_feat, np.ndarray):
            audio_feat = torch.from_numpy(audio_feat)
        if audio_feat.dim() == 3 and audio_feat.shape[0] == 1:
            audio_feat = audio_feat.squeeze(0)
        if audio_feat.dim() == 1:
            audio_feat = audio_feat.unsqueeze(0)
        return audio_feat


class Deploy:
    def __init__(
        self,
        file_path: str | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_code: str | None = "light",
        sr: int = 16000,
        time_enabled: bool = False,
        no_cuda: bool = True,
        ignore_text: bool = False,
        num_workers: int = 4,
        text_batch_size: int = 128,
        truncate_len: float | None = None,
        debug: bool = False,
        save_memory: bool = True,
    ):
        if file_path is None:
            file_path = "../audio_samples/case_concat.wav"

        if checkpoint_dir is None:
            if checkpoint_code is None:
                raise ValueError("checkpoint_dir or checkpoint_code must be provided")
            checkpoint_dir = f"../output/{checkpoint_code}"

        self.file_path = file_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.runner = Inference(
            checkpoint_dir=self.checkpoint_dir,
            file_path=self.file_path,
            no_cuda=no_cuda,
            time_enabled=time_enabled,
            sr=sr,
            ignore_text=ignore_text,
            num_workers=num_workers,
            text_batch_size=text_batch_size,
            truncate_len=truncate_len,
            debug=debug,
            save_memory=save_memory,
        )

    def run(self, file_path: str | None = None) -> list[dict]:
        if file_path is not None:
            self.file_path = file_path
            self.runner.file_path = file_path
        return self.runner.run_audio(file_path=self.file_path)


 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="../audio_samples/case_concat.wav")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_code", type=str, default='lightweight')
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--ignore_text", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--text_batch_size", type=int, default=128)
    parser.add_argument("--save_memory", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.checkpoint_code is not None and args.checkpoint_dir is None:
        args.checkpoint_dir = f"../output/{args.checkpoint_code}"
        # print(f"Inferred checkpoint_dir: {args.checkpoint_dir}", flush=True)

    truncate_len = None
    if args.debug:
        # truncate_len = 3
        args.num_workers = 1
        print("Debug mode: forcing num_workers=1 for deterministic output and easier debugging.", flush=True)

    deployer = Deploy(
        file_path=args.file,
        checkpoint_dir=Path(args.checkpoint_dir),
        checkpoint_code=args.checkpoint_code,
        sr=args.sr,
        time_enabled=args.time,
        no_cuda=args.no_cuda,
        ignore_text=args.ignore_text,
        num_workers=args.num_workers,
        text_batch_size=args.text_batch_size,
        truncate_len=truncate_len,
        debug=args.debug,
        save_memory=args.save_memory,
    )
    deployer.run()


'''
Right now, save_memory results in text model being none.
Restructure the memory saving mechanism.
Save memory will only apply to the audio model, if audio model is not whisper.
Load the audio model and keep it on the device.
If save memory is True, don't yet load whisper and text models. If false, do load them.
When audio arrives, process it using the audio model.
If save memory is True, then release the audio model, load the whisper and text models.
Do the rest of the processing. Then release the whisper and text models.

Last cell takes the audio and does the processing so if save memory is true, you will need to
re-initialize the audio model, do the processing, and then release it again, load the whisper and text, process, and so on....
'''