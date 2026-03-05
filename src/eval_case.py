import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import os
import time
import json
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # only show errors

from sentence_transformers import SentenceTransformer
from models import Whisper, Qwen3Text, Wav2Vec2Embedding, Qwen2Audio, HFText, AttentionClassifier, AveragingClassifier

from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import Optional, Union
from tqdm import tqdm
from dataset import (
	CASEDataset,
	get_emotion_mapping,
	apply_truncation,
	apply_padding,
	setup_target_lengths,
)
from pprint import pprint


def _compute_metrics(results_df: pd.DataFrame, use_text: bool) -> dict:
	audio_acc = accuracy_score(results_df['audio_label'], results_df['audio_pred'])
	metrics = {
		"audio_micro_accuracy": float(audio_acc),
	}
	if use_text:
		text_acc = accuracy_score(results_df['text_label'], results_df['text_pred'])
		metrics["text_micro_accuracy"] = float(text_acc)

	audio_labels_unique = np.unique(results_df['audio_label'])
	audio_f1_per_class = f1_score(
		results_df['audio_label'],
		results_df['audio_pred'],
		labels=audio_labels_unique,
		average=None,
		zero_division=0,
	)
	audio_f1_macro = f1_score(
		results_df['audio_label'],
		results_df['audio_pred'],
		average='macro',
		zero_division=0,
	)
	metrics["audio_macro_f1"] = float(audio_f1_macro)
	metrics["audio_f1_per_class"] = dict(zip(audio_labels_unique.tolist(), audio_f1_per_class.tolist()))

	if use_text:
		text_labels_unique = np.unique(results_df['text_label'])
		text_f1_per_class = f1_score(
			results_df['text_label'],
			results_df['text_pred'],
			labels=text_labels_unique,
			average=None,
			zero_division=0,
		)
		text_f1_macro = f1_score(
			results_df['text_label'],
			results_df['text_pred'],
			average='macro',
			zero_division=0,
		)
		metrics["text_macro_f1"] = float(text_f1_macro)
		metrics["text_f1_per_class"] = dict(zip(text_labels_unique.tolist(), text_f1_per_class.tolist()))

	if "fusion_pred" in results_df.columns:
		fusion_acc_audio = accuracy_score(results_df['audio_label'], results_df['fusion_pred'])
		fusion_acc_text = accuracy_score(results_df['text_label'], results_df['fusion_pred'])
		metrics["fusion_micro_accuracy_audio"] = float(fusion_acc_audio)
		metrics["fusion_micro_accuracy_text"] = float(fusion_acc_text)

		fusion_labels_unique = np.unique(results_df['audio_label'])
		fusion_f1_per_class_audio = f1_score(
			results_df['audio_label'],
			results_df['fusion_pred'],
			labels=fusion_labels_unique,
			average=None,
			zero_division=0,
		)
		fusion_f1_macro_audio = f1_score(
			results_df['audio_label'],
			results_df['fusion_pred'],
			average='macro',
			zero_division=0,
		)
		metrics["fusion_macro_f1_audio"] = float(fusion_f1_macro_audio)
		metrics["fusion_f1_per_class_audio"] = dict(zip(fusion_labels_unique.tolist(), fusion_f1_per_class_audio.tolist()))

		fusion_labels_unique_text = np.unique(results_df['text_label'])
		fusion_f1_per_class_text = f1_score(
			results_df['text_label'],
			results_df['fusion_pred'],
			labels=fusion_labels_unique_text,
			average=None,
			zero_division=0,
		)
		fusion_f1_macro_text = f1_score(
			results_df['text_label'],
			results_df['fusion_pred'],
			average='macro',
			zero_division=0,
		)
		metrics["fusion_macro_f1_text"] = float(fusion_f1_macro_text)
		metrics["fusion_f1_per_class_text"] = dict(zip(fusion_labels_unique_text.tolist(), fusion_f1_per_class_text.tolist()))

	return metrics


def batch_evaluate(start_time: str, output_dir: str = '../output', debug=False, use_extracted_features: bool = False, use_whisper_text: bool = True) -> pd.DataFrame:
	
	output_path = Path(output_dir)
	start_ts = pd.to_datetime(start_time, format='%Y%m%d-%H%M%S-%f')

	rows = []
	run_dirs = [d for d in output_path.iterdir() if d.is_dir()]
	run_dirs = sorted(run_dirs, key=lambda x: x.name, reverse=True)

	for run_dir in tqdm(run_dirs):
		run_time = pd.to_datetime(run_dir.name, format='%Y%m%d-%H%M%S-%f', errors='coerce')
		if pd.isna(run_time) or run_time < start_ts:
			continue

		config_path = run_dir / 'config.pth'
		if not config_path.exists():
			continue

		try:
			config = torch.load(config_path, map_location='cpu', weights_only=False)
		except Exception:
			config = {}

		if not config["case_mapping"]:
			continue

		metrics = {
			"micro_accuracy": float('nan'),
			"macro_f1": float('nan'),
			"runtime_rate": float('nan'),
			"device": float('nan'),
			"emotion_match": float('nan'),
		}
		evaluator = CASEEvaluator(
			checkpoint_dir=str(run_dir),
			no_cuda=False,
			debug=debug,
			ignore_text=False,
			use_extracted_features=use_extracted_features,
			use_whisper_text=use_whisper_text,
		)
		results_payload = evaluator.evaluate()
		metrics["micro_accuracy"] = results_payload["results"]["audio"]["micro_accuracy"]
		metrics["macro_f1"] = results_payload["results"]["audio"]["macro_f1"]
		metrics["runtime_rate"] = results_payload["results"]["runtime"]["forward_rate_sec_per_sec_audio"]
		metrics["device"] = results_payload["device"]
		metrics["emotion_match"] = results_payload["emotion_match"]

		row = {
			"start_time": run_dir.name,
			"audio_model": config["audio_model"],
			"text_model": config["text_model"],
			"fusion_method": config["fusion_method"],
			"accuracy_averaging": config["accuracy_averaging"],
			"dataset": config["dataset"],
			"merged_datasets": config["merged_datasets"],
			"device": metrics["device"],
			"emotion_match": metrics["emotion_match"],
			"micro_accuracy": metrics["micro_accuracy"],
			"macro_f1": metrics["macro_f1"],
			"runtime_rate": metrics["runtime_rate"],
			
		}
		rows.append(row)

	print(f"Evaluated {len(rows)}")

	results_df = pd.DataFrame(rows)
	output_csv = output_path / 'case_batched_results.csv'
	if not debug:
		results_df.to_csv(output_csv, index=False)
		print(f"Saved batched CASE results to {output_csv}")
	return results_df



class CASEEvaluator:
	def __init__(
		self,
		checkpoint_dir: str = None,
		classifier=None,
		model_config=None,
		config=None,
		sr: int = 16000,
		time_flag: bool = False,
		no_cuda: bool = False,
		debug: bool = False,
		ignore_text: bool = False,
		training=False,
		use_extracted_features: bool = False,
		use_whisper_text: bool = True,
		max_samples = -1
	) -> None:
		
		self.max_samples = max_samples
		self.training = training
		self.device = torch.device('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')

		if checkpoint_dir is None:
			assert classifier is not None and model_config is not None and config is not None, "If checkpoint_dir is not provided, classifier, model_config, and config must be provided"
			self.config = config
			self.model_config = model_config
			self.classifier = classifier
			self.checkpoint_dir = None
		else:
			self.checkpoint_dir = Path(checkpoint_dir)
			config_path = self.checkpoint_dir / 'config.pth'
			self.config = torch.load(config_path, weights_only=False)
			model_path = self.checkpoint_dir / 'model.pth'
			model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
			self.state_dict = model['model']
			self.model_config = model['config']
			if self.config['fusion_method'] == 'attention':
				self.classifier = AttentionClassifier(
					modality_dims=self.model_config['modality_dims'],
					d_model=self.model_config['d_model'],
					n_classes=self.model_config['n_classes'],
					n_heads=self.config['n_heads'],
					dropout=self.model_config['dropout'],
					whisper_embedding_len=self.config['whisper_embedding_len'],
				).to(self.device)
			elif self.config['fusion_method'] == 'average':
				self.classifier = AveragingClassifier(
					modality_dims=self.model_config['modality_dims'],
					d_model=self.model_config['d_model'],
					n_classes=self.model_config['n_classes'],
					dropout=self.config['dropout'],
				).to(self.device)
			else:
				raise ValueError(f"Unknown fusion method: {self.config['fusion_method']}")
			
			self.classifier.load_state_dict(self.state_dict, strict=False)

		
		self.sr = sr
		self.time_flag = time_flag
		self.no_cuda = no_cuda
		self.debug = debug
		self.ignore_text = ignore_text
		self.use_extracted_features = use_extracted_features
		self.use_whisper_text = use_whisper_text
		self.device = torch.device('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')

		self.emotion_mapping = get_emotion_mapping()
		self.feature_names = {
			'audio': self.config['audio_model'],
			'text': self.config['text_model'],
		}
		self.target_length_seconds = float(self.config['target_length_seconds'])
		self.target_lengths = setup_target_lengths(self.feature_names, self.target_length_seconds)

		metadata_path = Path('../../../../../datasets/public/CASE/metadata_split.csv')
		feature_dir = metadata_path.parent / 'features'
		self.dataset = CASEDataset(
			sr=sr,
			debug=False,
			use_extracted_features=self.use_extracted_features,
			feature_names=self.feature_names if self.use_extracted_features else None,
			feature_dir=feature_dir,
			target_length_seconds=self.target_length_seconds,
			n_qwen3_features=self.config['n_qwen3_features'],
			truncate_from_start=True,
		)
		self.feature_dir = feature_dir

		self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

		model_emotions = sorted(set(self.config['idx_to_emotion'].values()))
		if hasattr(self.dataset, 'emotion_mapping'):
			dataset_emotions = sorted(set(self.dataset.emotion_mapping.values()))
		else:
			dataset_emotions = sorted(set(self.emotion_mapping.values()))
		if "remove" in dataset_emotions:
			dataset_emotions.remove("remove")
		self.emotion_match = model_emotions == dataset_emotions
		if not self.emotion_match:
			print(
				f"Warning: Emotions in model ({model_emotions}) do not match emotions in dataset ({dataset_emotions}). "
				"This may lead to incorrect evaluation results."
			)


		self.idx_to_emotion = self.config['idx_to_emotion']
		assert self.model_config['n_classes'] == len(self.idx_to_emotion), (
			f"Expected {len(self.idx_to_emotion)} classes but model was trained with {self.model_config['n_classes']}"
		)

		text_model_name = self.config['text_model']
		text_dim = self.model_config['modality_dims']['text']
		self.use_text = (
			(not self.ignore_text)
			and (text_model_name != 'none')
			and (text_dim is not None)
			and (text_dim > 0)
		)

		self.text_model = None
		self.n_text_features = None
		if self.use_text:
			if self.use_extracted_features:
				if 'qwen3' in text_model_name.lower():
					self.n_text_features = self.config['n_qwen3_features']
				else:
					self.n_text_features = self.model_config['modality_dims']['text']
			else:
				if 'qwen3' in text_model_name.lower():
					self.text_model = Qwen3Text(text_model_name)
					self.n_text_features = self.config['n_qwen3_features']
					assert self.n_text_features == self.model_config['modality_dims']['text'], (
						f"Expected text feature dim {self.model_config['modality_dims']['text']} but got {self.n_text_features}"
					)
				elif text_model_name.lower() == "minilm":
					self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
					self.n_text_features = self.model_config['modality_dims']['text']
				elif text_model_name.lower() == "llama":
					self.text_model =  HFText('meta-llama/Llama-3.2-1B', embed_only=True)
					self.n_text_features = self.model_config['modality_dims']['text']

		self.whisper_model = None
		self.audio_model = None
		if not self.use_extracted_features:
			if self.use_whisper_text or 'whisper' in self.config['audio_model'].lower():
				self.whisper_model = Whisper(model_name='distil_whisper')

			if 'wav2vec2' in self.config['audio_model'].lower():
				self.audio_model = Wav2Vec2Embedding(model_name="facebook/wav2vec2-large-960h").eval()
			elif 'qwen2' in self.config['audio_model'].lower():
				self.audio_model = Qwen2Audio(
					instruct=False,
					average_last_n=1,
					no_prompt=False,
					embed_only=True,
					get_audio_tower_features=True,
				)
			else:
				raise ValueError(f"Unsupported audio_model for CASE evaluation: {self.config['audio_model']}")


	@staticmethod
	def _ensure_audio_tensor(audio_feat: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
		if isinstance(audio_feat, np.ndarray):
			audio_feat = torch.from_numpy(audio_feat)
		if audio_feat.dim() == 1:
			audio_feat = audio_feat.unsqueeze(0)
		if audio_feat.dim() == 2:
			audio_feat = audio_feat.unsqueeze(0)
		return audio_feat

	def _apply_target_length(self, feature: torch.Tensor, modality: str) -> torch.Tensor:
		feature_name = self.feature_names.get(modality, 'none')
		target_length = self.target_lengths.get(feature_name)
		if target_length is None or target_length <= 0:
			return feature
		if feature.dim() == 3 and feature.size(0) == 1:
			feature_2d = feature.squeeze(0)
			feature_2d = apply_truncation(feature_2d, target_length, from_start=True)
			feature_2d = apply_padding(feature_2d, target_length)
			return feature_2d.unsqueeze(0)
		if feature.dim() == 2:
			feature = apply_truncation(feature, target_length, from_start=True)
			feature = apply_padding(feature, target_length)
		return feature

	def evaluate(self) -> dict:
		dtype = next(self.classifier.parameters()).dtype

		files = []
		texts = []
		audio_labels = []
		text_labels = []
		audio_preds = []
		text_preds = []
		fusion_preds = []
		audio_probs = []
		text_probs = []
		fusion_probs = []
		forward_rates = []

		self.classifier.eval()
		with torch.no_grad():
			if not self.training:
				self.loader = tqdm(self.loader, total=len(self.loader), desc="Evaluating CASE samples")

			for i, sample in enumerate(self.loader):

				if self.max_samples > 0 and i >= self.max_samples:
					break
				features = None
				if isinstance(sample, dict):
					file_ = sample['file'][0]
					audio = sample['audio']
					text = sample['text'][0]
					audio_label = sample['audio_label'][0]
					text_label = sample['text_label'][0]

				elif isinstance(sample, (list, tuple)):
					features, labels, file_, _ = sample
					if isinstance(file_, (list, tuple)):
						file_ = file_[0]
					audio_label = labels['audio']
					text_label = labels['text']
					if isinstance(audio_label, (list, tuple)):
						audio_label = audio_label[0] if len(audio_label) > 0 else float('nan')
					if isinstance(text_label, (list, tuple)):
						text_label = text_label[0] if len(text_label) > 0 else float('nan')
					audio = torch.tensor([], dtype=torch.float32)
					text = ""
				else:
					raise ValueError("Unsupported sample format in CASEEvaluator.")

				start_time = time.perf_counter()

				if self.use_extracted_features:
					if features is None:
						raise ValueError("Expected CASEDataset extracted-features output when use_extracted_features is true.")
					whisper_text = text.strip()
					audio_feat = features['audio_feature'].to(device=self.device, dtype=dtype).detach()
					has_audio = features['has_audio'].to(device=self.device)
					if self.use_text:
						text_feat = features['text_feature'].to(device=self.device, dtype=dtype).detach()
						has_text = features['has_text'].to(device=self.device)
					else:
						text_feat = torch.full((1, 1, 1), float('nan'), device=self.device, dtype=dtype)
						has_text = torch.tensor([False], dtype=torch.bool, device=self.device)
					audio_len = float('nan')
				else:
					if isinstance(audio, torch.Tensor):
						audio_np = audio.cpu().numpy()
					else:
						audio_np = np.asarray(audio)
					
					if 'whisper' in self.config['audio_model'].lower():
						if self.whisper_model is None:
							raise ValueError("Whisper model is required for whisper audio features but use_whisper_text is False.")
						whisper_out = self.whisper_model(audio_np, self.sr)
						audio_feat = whisper_out["audio_feature"]
						whisper_text = whisper_out["text"].strip()
					else:
						if self.config['audio_model'].lower() == 'qwen2_audio_tower':
							audio_feat = self.audio_model.get_audio_tower_features(audio_np.squeeze(0))
						else:
							audio_feat = self.audio_model(audio=audio_np, sr=self.sr)

						if self.use_whisper_text:
							if self.whisper_model is None:
								raise ValueError("Whisper model is required for use_whisper_text=True but is not initialized.")
							whisper_out = self.whisper_model(audio_np, self.sr)
							whisper_text = whisper_out["text"].strip()
						else:
							whisper_text = text.strip()

					if isinstance(audio_feat, np.ndarray):
						audio_feat = torch.from_numpy(audio_feat)
					if audio_feat.dim() == 1:
						audio_feat = audio_feat.unsqueeze(0)
					audio_feat = audio_feat.unsqueeze(0) if audio_feat.dim() == 2 else audio_feat
					audio_feat = audio_feat.to(device=self.device, dtype=dtype).detach()
					has_audio = torch.tensor([True], dtype=torch.bool, device=self.device)

					if self.use_text:
						if self.config['text_model'].lower() == "minilm":
							text_feat = self.text_model.encode([whisper_text], output_value='token_embeddings')[0].unsqueeze(0)
						else:
							text_feat = self.text_model([whisper_text])
						if isinstance(text_feat, np.ndarray):
							text_feat = torch.from_numpy(text_feat)
						text_feat = text_feat.to(torch.float32).detach()
						if self.n_text_features is not None and self.n_text_features < text_feat.shape[2]:
							text_feat = text_feat[:, :, :self.n_text_features]
							if 'qwen3' in self.config['text_model'].lower():
								text_feat = F.normalize(text_feat, p=2, dim=2)
						text_feat = self._apply_target_length(text_feat, 'text')
						text_feat = text_feat.to(device=self.device, dtype=dtype)
						has_text = torch.tensor([True], dtype=torch.bool, device=self.device)
					else:
						text_feat = torch.full((1, 1, 1), float('nan'), device=self.device, dtype=dtype)
						has_text = torch.tensor([False], dtype=torch.bool, device=self.device)
					audio_len = float(audio_np.shape[-1]) / float(self.sr) if audio_np.shape[-1] > 0 else float('nan')

				if isinstance(audio_feat, np.ndarray):
					audio_feat = torch.from_numpy(audio_feat)
				audio_feat = self._apply_target_length(audio_feat, 'audio')
				audio_feat = self._ensure_audio_tensor(audio_feat)
				audio_feat = audio_feat.to(device=self.device, dtype=dtype).detach()

				inputs = {
					"audio_feature": audio_feat,
					"text_feature": text_feat,
					"has_audio": has_audio,
					"has_text": has_text,
				}

				outputs = self.classifier(inputs)
				logits_audio = outputs["audio"].detach()
				prob_audio = F.softmax(logits_audio, dim=1).cpu().numpy()[0]
				audio_index = int(np.argmax(prob_audio))
				max_prob_audio = prob_audio[audio_index]
				audio_pred = self.idx_to_emotion[audio_index]
				audio_pred = self.emotion_mapping[audio_pred]

				if self.use_text:
					logits_text = outputs["text"].detach()
					logits_fusion = outputs["fusion"].detach()
					prob_text = F.softmax(logits_text, dim=1).cpu().numpy()[0]
					prob_fusion = F.softmax(logits_fusion, dim=1).cpu().numpy()[0]

					text_index = int(np.argmax(prob_text))
					fusion_index = int(np.argmax(prob_fusion))
					max_prob_text = prob_text[text_index]
					max_prob_fusion = prob_fusion[fusion_index]
					text_pred = self.idx_to_emotion[text_index]
					text_pred = self.emotion_mapping[text_pred]
					fusion_pred = self.idx_to_emotion[fusion_index]
					fusion_pred = self.emotion_mapping[fusion_pred]
				else:
					text_pred = None
					fusion_pred = None
					max_prob_text = float('nan')
					max_prob_fusion = float('nan')

				end_time = time.perf_counter()
				rate = (end_time - start_time) / audio_len if audio_len and not np.isnan(audio_len) else -1
				forward_rates.append(rate)

				audio_label = self.emotion_mapping[audio_label]
				text_label = self.emotion_mapping[text_label]

				texts.append(whisper_text)
				files.append(file_)
				audio_labels.append(audio_label)
				text_labels.append(text_label)
				audio_preds.append(audio_pred)
				text_preds.append(text_pred)
				fusion_preds.append(fusion_pred)
				text_probs.append(max_prob_text)
				audio_probs.append(max_prob_audio)
				fusion_probs.append(max_prob_fusion)


				if len(files) == 0:
					raise RuntimeError(
						"CASE evaluation produced zero samples. "
						f"dataset_len={len(self.dataset)}, max_samples={self.max_samples}, "
						f"use_extracted_features={self.use_extracted_features}, "
						f"use_text={self.use_text}"
					)

			results = {
				'file': files,
				'text': texts,
				'audio_label': audio_labels,
				'audio_pred': audio_preds,
				'audio_prob': np.round(audio_probs, 1),
			}

			if self.use_text:
				results.update({
					'text_label': text_labels,
					'text_pred': text_preds,
					'text_prob': np.round(text_probs, 1),
					'fusion_pred': fusion_preds,
					'fusion_prob': np.round(fusion_probs, 1),
				})

			results_df = pd.DataFrame(results)
			metrics = _compute_metrics(results_df=results_df, use_text=self.use_text)
			avg_forward_rate = float(np.nanmean(forward_rates)) if forward_rates else float('nan')
			results_path = Path('../output/case_results.csv') if not self.debug else Path('../output/case_results_DEBUG.csv')
			if not self.training:
				results_df.to_csv(results_path, index=False)
				print(f"Saved results to {results_path}")
				print(results_df)
				print(f"Average forward-pass rate (sec/sec audio): {avg_forward_rate:.6f}")

				print(f"Audio micro accuracy: {metrics['audio_micro_accuracy']:.4f}")
				print(f"Audio macro F1: {metrics['audio_macro_f1']:.4f}")
				print("Audio per-class F1:")
				print(metrics['audio_f1_per_class'])
				if self.use_text:
					print(f"Text micro accuracy: {metrics['text_micro_accuracy']:.4f}")
					print(f"Text macro F1: {metrics['text_macro_f1']:.4f}")
					print("Text per-class F1:")
					print(metrics['text_f1_per_class'])

			results_payload = {
				"parameters": {
					"audio_model": self.config["audio_model"],
					"text_model": self.config["text_model"],
					"fusion_method": self.config["fusion_method"],
					"checkpoint_dir": str(self.checkpoint_dir),
					"accuracy_averaging": self.config["accuracy_averaging"],
				},
				"emotion_match": self.emotion_match,
				"device": self.device.type,
				"results": {
					"audio": {
						"micro_accuracy": metrics['audio_micro_accuracy'],
						"macro_f1": metrics['audio_macro_f1'],
						"per_class_f1": metrics['audio_f1_per_class'],
					},
					"runtime": {
						"forward_rate_sec_per_sec_audio": avg_forward_rate,
					},
				},
			}

			if self.use_text:
				results_payload["results"]["text"] = {
					"micro_accuracy": metrics['text_micro_accuracy'],
					"macro_f1": metrics['text_macro_f1'],
					"per_class_f1": metrics['text_f1_per_class'],
				}

			if 'fusion_macro_f1_text' in metrics:
				results_payload["results"]["fusion"] = {
					"micro_accuracy_audio": metrics['fusion_micro_accuracy_audio'],
					"micro_accuracy_text": metrics['fusion_micro_accuracy_text'],
					"macro_f1_audio": metrics['fusion_macro_f1_audio'],
					"per_class_f1_audio": metrics['fusion_f1_per_class_audio'],
					"macro_f1_text": metrics['fusion_macro_f1_text'],
					"per_class_f1_text": metrics['fusion_f1_per_class_text'],
				}


			
			if not self.debug and not self.training:
				json_path = self.checkpoint_dir / "case_results.json"
				with open(json_path, "w", encoding="utf-8") as json_file:
					json.dump(results_payload, json_file, indent=2)
				print(f"Saved JSON results to {json_path}")

			return results_payload



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint_dir', type=str, default=None)	# wav2vec2, qwen0.6, iemocap, multi, no emotion match
	parser.add_argument('--sr', type=int, default=16000)
	parser.add_argument('--time', action='store_true')
	parser.add_argument('--no_cuda', action='store_true')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--ignore_text', action='store_true')
	parser.add_argument('--use_extracted_features', action='store_true')
	parser.add_argument('--use_whisper_text', action='store_true')
	parser.add_argument('--batch_start_time', type=str, default=None)
	parser.add_argument('--tqdm', action='store_true', help="Use tqdm progress bars for batch evaluation")
	args = parser.parse_args()

	with torch.no_grad():
		if args.batch_start_time:
			batch_evaluate(
				args.batch_start_time,
				debug=args.debug,
				use_extracted_features=args.use_extracted_features,
				use_whisper_text=args.use_whisper_text,
			)
		else:
			evaluator = CASEEvaluator(
				checkpoint_dir=args.checkpoint_dir,
				sr=args.sr,
				time_flag=args.time,
				no_cuda=args.no_cuda,
				debug=args.debug,
				ignore_text=args.ignore_text,
				use_extracted_features=args.use_extracted_features,
				use_whisper_text=args.use_whisper_text,
				training=args.tqdm
			)
			result = evaluator.evaluate()
			pprint(result)
