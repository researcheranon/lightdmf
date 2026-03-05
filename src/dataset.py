import torch
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F
import pandas as pd
import torchaudio
import numpy as np
from pathlib import Path
import utils as u
import librosa
import warnings
from tqdm import tqdm
from time import time


warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")
warnings.filterwarnings(
	"ignore",
	message="PySoundFile failed. Trying audioread instead.",
	category=UserWarning,
	module="librosa"
)

# Unified emotion mapping for case (lowercase, unique keys)
def get_emotion_mapping():

	CASE_MAPPING = {
		# Angry
		'anger': 'angry',
		'angry': 'angry',
		'frustration': 'angry',
		'annoyance': 'angry',
		'disapproval': 'angry',
		'contempt': 'angry',

		# Happy
		'happy': 'happy',
		'happiness': 'happy',
		'joy': 'happy',
		'admiration': 'happy',
		'amusement': 'happy',
		'approval': 'happy',
		'caring': 'happy',
		'gratitude': 'happy',
		'love': 'happy',
		'optimism': 'happy',
		'pride': 'happy',
		'relief': 'happy',

		# Excited
		'excited': 'excited',
		'ps': 'excited',
		'surprise': 'excited',
		'surprised': 'excited',
		'desire': 'excited',
		'excitement': 'excited',

		# Sad
		'sad': 'sad',
		'sadness': 'sad',
		'disappointment': 'sad',
		'embarrassment': 'sad',
		'grief': 'sad',
		'remorse': 'sad',

		# Neutral
		'calm': 'neutral',
		'neutral': 'neutral',

		# Remove / discard
		'confusion': 'remove',
		'curiosity': 'remove',
		'nervousness': 'remove',
		'realization': 'remove',
		'fear': 'remove',
		'fearful': 'remove',
		'disgust': 'remove',
		'no_agreement': 'remove',
		'other': 'remove'
	}

	case_emotion_mapping = {
		'anger': 'angry',
		'angry': 'angry',
		'calm': 'neutral',
		'contempt': 'remove',
		'disgust': 'remove',
		'excited': 'excited',
		'fear': 'remove',
		'fearful': 'remove',
		'frustration': 'angry',
		'happy': 'happy',
		'happiness': 'happy',
		'joy': 'happy',
		'neutral': 'neutral',
		'no_agreement': 'remove',
		'other': 'remove',
		'ps': 'excited',
		'sad': 'sad',
		'sadness': 'sad',
		'surprise': 'excited',
		'surprised': 'excited',

		# GoEmotions to CASE
		'admiration': 'happy',
		'amusement': 'happy',
		'annoyance': 'angry',
		'approval': 'happy',
		'caring': 'happy',
		'confusion': 'neutral',
		'curiosity': 'neutral',
		'desire': 'excited',
		'disappointment': 'sad',
		'disapproval': 'angry',
		'embarrassment': 'sad',
		'excitement': 'excited',
		'gratitude': 'happy',
		'grief': 'sad',
		'love': 'happy',
		'nervousness': 'neutral',
		'optimism': 'happy',
		'pride': 'happy',
		'realization': 'neutral',
		'relief': 'happy',
		'remorse': 'sad',
	}
	return case_emotion_mapping


def merge_metadata():

	output_path = Path('../metadata_merged_case.csv')

	main_dir = Path('../../../../../datasets/public')

	merged_metadata = []
	for dataset_name in ('CREMA_D', 'IEMOCAP', 'MELD', 'RAVDESS', 'TESS', 'GoEmotions'):
		dataset_dir = main_dir / dataset_name
		metadata_path = dataset_dir / 'metadata_split.csv'
		df = pd.read_csv(metadata_path)
		df['dataset_name'] = dataset_name
		df['dataset_dir'] = str(dataset_dir)
		emotion_mapping = get_emotion_mapping()
		df['Emotion'] = df['Emotion'].str.lower().map(emotion_mapping)
		df = df.drop_duplicates()

		unique_utterance = dataset_name in ('IEMOCAP', 'MELD', 'GoEmotions')
		unique_audio = dataset_name != 'GoEmotions'
		df['unique_utterance'] = unique_utterance
		df['unique_audio'] = unique_audio

		if dataset_name == 'GoEmotions':
			df['duration'] = 1
			# Drop rows with the same file_path, split, and Utterance but with one of the Emotion values as 'neutral'
			df = df[~df.duplicated(subset=['file_path', 'split', 'Utterance'], keep=False) | (df['Emotion'] != 'neutral')]

		df = df [['file_path', 'split', 'Emotion', 'Utterance', 'dataset_name', 'dataset_dir', 'unique_utterance', 'unique_audio',  'duration']]
		merged_metadata.append(df)
	merged_df = pd.concat(merged_metadata, ignore_index=True)
	merged_df = merged_df[merged_df['Emotion'] != 'remove']
	merged_df = merged_df[merged_df['duration'] > 0]

	merged_df.to_csv(output_path, index=False)




cp1252_map = {
	"\x91": "'",
	"\x92": "'",
	"\x93": '"',
	"\x94": '"',
	"\x95": "•",
	"\x96": "-",
	"\x97": "—",
	"\x85": "...",
}

def get_target_lengths(target_length_seconds, feature_name):
	lengths = {
		'minilm': 64,
		'llama': 96,
		'qwen2_audio_tower': int(target_length_seconds * 25),
		'qwen3_text': 96,
		'wav2vec2_xls': int(target_length_seconds * 50),
		'Qwen3-Embedding-0.6B': 96,
		'whisper_tiny_audio': 1500,
		'none': -1
	}
	return lengths.get(feature_name, 96)  # Default to 96 if unknown


def setup_feature_directories(feature_dir, feature_names):
	"""Setup feature subdirectories."""
	feature_main_dir = Path(feature_dir)
	feature_sub_dirs = {}
	for feature_name in feature_names.values():
		sub_dir = feature_main_dir / feature_name
		feature_sub_dirs[feature_name] = sub_dir
	return feature_main_dir, feature_sub_dirs


def setup_target_lengths(feature_names, target_length_seconds):
	"""Calculate target lengths for all features."""
	target_lengths = {}
	for modality, feature_name in feature_names.items():
		# if feature_name != 'none':
		target_lengths[feature_name] = get_target_lengths(target_length_seconds, feature_name)
	return target_lengths


def setup_emotion_mappings(metadata):
	"""Setup emotion to index mappings."""
	emotions = sorted(metadata['Emotion'].str.lower().unique().tolist())
	emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}
	idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
	return emotions, emotion_to_idx, idx_to_emotion


def load_feature_file(feature_path, feature_name, n_qwen3_features=None):
	"""Load a single feature file with proper handling."""
	# try:
	if feature_name == 'mfcc':
		feature_data = np.load(feature_path, allow_pickle=True).item()['mfcc'].T
	else:
		feature_data = np.load(feature_path)
	
	feature_data = torch.from_numpy(feature_data).float()
	if feature_data.ndim == 1:
		feature_data = feature_data.unsqueeze(0)

	if n_qwen3_features is not None and 'qwen3' in feature_name.lower():
		if feature_data.shape[1] > n_qwen3_features:
			# Qwen3 is trained with Matryoshka Representation Learning (MRL)
			# We can take the first n features and normalize
			feature_data = feature_data[:, :n_qwen3_features]
		feature_data = F.normalize(feature_data, p=2, dim=1)

	return feature_data


def apply_truncation(feature_data, target_length, from_start=False):
	"""Apply truncation to feature data."""
	if feature_data.shape[0] > target_length and target_length < float('inf'):
		start = 0 if from_start else torch.randint(0, feature_data.shape[0] - target_length + 1, (1,)).item()
		feature_data = feature_data[start:start + target_length, :]
	return feature_data


def apply_padding(feature_data, target_length):
	"""Apply padding to feature data."""
	if feature_data.shape[0] < target_length:
		padding = target_length - feature_data.shape[0]
		feature_data = torch.nn.functional.pad(feature_data, (0, 0, 0, padding))
	return feature_data


def get_modality_dims(sample, feature_names, feature_dir=None, n_qwen3_features=None):
	"""Get modality dimensions."""
	filestem = Path(sample['file_path']).stem
	modality_dims = {}
	if feature_dir is None:
		feature_dir = Path(sample['dataset_dir']) / 'features'
	for modality, feature_name in feature_names.items():
		if feature_name == 'none':
			modality_dims[modality] = -1
		else:
			feature_path = Path(feature_dir) / feature_name / f"{filestem}.npy"
			sample_data = load_feature_file(feature_path, feature_name, n_qwen3_features=n_qwen3_features)
			modality_dims[modality] = sample_data.shape[-1]
	return modality_dims

def validate_feature_files(metadata, feature_dir, feature_names):
	"""Validate that all required feature files exist."""
	valid_indices = []
	for idx, row in metadata.iterrows():
		filename = Path(str(row['file_path'])).stem
		all_features_exist = True
		for modality, feature_name in feature_names.items():
			if feature_name != 'none':
				feature_path = Path(feature_dir) / feature_name / f"{filename}.npy"
				if not feature_path.exists():
					all_features_exist = False
					break
		if all_features_exist:
			valid_indices.append(idx)
	return valid_indices


def apply_label_balancing(metadata):
	"""Apply label balancing to metadata and return file paths."""
	label_counts = metadata['Emotion'].value_counts().to_dict()
	max_count = max(label_counts.values())
	filepaths = []
	for emotion, count in label_counts.items():
		emotion_filepaths = metadata[metadata['Emotion'] == emotion]['file_path'].tolist()
		multiplier = max_count // count
		remainder = max_count % count
		filepaths.extend(emotion_filepaths * multiplier)
		filepaths.extend(np.random.choice(emotion_filepaths, remainder, replace=False).tolist())
	return filepaths


def balance_unique_pairs(metadata):
	"""Balance rows by (unique_utterance, unique_audio) pairs.

	Ensures equal counts for:
	- (True, True)
	- (True, False)
	- (False, True)

	Returns a list of file_path values.
	"""
	pairs_to_keep = [(True, True), (True, False), (False, True)]
	filepaths = []
	counts = {}
	for pair in pairs_to_keep:
		pair_df = metadata[(metadata['unique_utterance'] == pair[0]) & (metadata['unique_audio'] == pair[1])]
		counts[pair] = len(pair_df)

	max_count = max(counts.values()) if counts else 0
	if max_count == 0:
		return metadata['file_path'].tolist()

	for pair in pairs_to_keep:
		pair_df = metadata[(metadata['unique_utterance'] == pair[0]) & (metadata['unique_audio'] == pair[1])]
		if pair_df.empty:
			continue
		pair_filepaths = pair_df['file_path'].tolist()
		multiplier = max_count // len(pair_filepaths)
		remainder = max_count % len(pair_filepaths)
		filepaths.extend(pair_filepaths * multiplier)
		if remainder > 0:
			replace = remainder > len(pair_filepaths)
			sampled = pair_df.sample(n=remainder, replace=replace, random_state=42)
			filepaths.extend(sampled['file_path'].tolist())

	if not filepaths:
		return metadata['file_path'].tolist()

	return filepaths


def get_naive_micro_baseline_from_metadata(metadata):
	"""Calculate naive baseline from metadata."""
	emotions = [str(e).strip().lower() for e in metadata['Emotion'].tolist()]
	if len(emotions) == 0:
		print("Warning: get_naive_micro_baseline called on empty split.")
		return None, 0.0
	most_frequent_emotion = max(set(emotions), key=emotions.count)
	correct = sum(1 for emotion in emotions if emotion == most_frequent_emotion)
	accuracy = correct / len(emotions) if len(emotions) > 0 else 0.0
	return most_frequent_emotion, accuracy


def get_naive_micro_baseline_from_metadata_with_idx(metadata, emotion_to_idx):
	"""Calculate naive baseline from metadata, returning index."""
	emotions = metadata[metadata['split'] == 'train']['Emotion'].to_numpy() if 'split' in metadata.columns else metadata['Emotion'].to_numpy()
	if len(emotions) == 0:
		print("Warning: get_naive_micro_baseline called on empty split.")
		return None, 0.0
	most_frequent_emotion = max(set(emotions), key=list(emotions).count)
	most_frequent_idx = emotion_to_idx[most_frequent_emotion]
	correct = sum(1 for emotion in metadata['Emotion'] if emotion == most_frequent_emotion)
	accuracy = correct / len(metadata) if len(metadata) > 0 else 0.0
	return most_frequent_idx, accuracy


def fix_cp1252(s: str) -> str:
	if not isinstance(s, str):
		s = str(s)
	return "".join(cp1252_map.get(c, c) for c in s)


def modify_iemocap_labels(metadata_orig):
	'''
	# Only for IEMOCAP, as done by Chen and Rudnicky (2023):

	Only four emotion categories are considered:
	neutral, sad, angry, and happy. In particular, the “excited”
	category is merged with “happy” due to its sparsity in the dataset. 
	'''
	metadata = metadata_orig.copy()
	metadata.loc[metadata['Emotion'].str.strip().str.lower() == 'excited', 'Emotion'] = 'happy'
	metadata = metadata[metadata['Emotion'].str.strip().str.lower().isin(['neutral', 'sadness', 'anger', 'happy'])]
	metadata = metadata.reset_index(drop=True)
	return metadata


# Only for batch size = 1
class CASEDataset(Dataset):
	def __init__(
		self,
		emotions=['angry', 'excited', 'happy', 'neutral', 'sad'],
		sr=16000,
		debug=False,
		use_extracted_features=False,
		feature_names=None,
		feature_dir=None,
		target_length_seconds=10,
		n_qwen3_features=None,
		truncate_from_start=True,
		rms_norm=True,
		target_rms=0.1,
	):
		self.sr = sr
		self.emotions = emotions
		self.debug = debug
		self.use_extracted_features = use_extracted_features
		self.feature_names = feature_names
		self.n_qwen3_features = n_qwen3_features
		self.truncate_from_start = truncate_from_start
		self.rms_norm = rms_norm
		self.target_rms = target_rms

		dataset_dir = Path('../../../../../datasets/public/CASE')
		self.audio_dir = dataset_dir / 'audios'
		metadata_path = dataset_dir / 'metadata_split.csv'
		self.metadata = pd.read_csv(metadata_path)
		if debug:
			self.metadata = self.metadata.head(4)

		# map emotions
		self.emotion_mapping = get_emotion_mapping()
		self.metadata['Audio_Emotion'] = self.metadata['Audio_Emotion'].map(self.emotion_mapping)
		self.metadata['Text_Emotion'] = self.metadata['Text_Emotion'].map(self.emotion_mapping)

		# keep selected emotions
		self.metadata['Audio_Emotion'] = self.metadata['Audio_Emotion'].str.strip().str.lower()
		self.metadata['Text_Emotion'] = self.metadata['Text_Emotion'].str.strip().str.lower()
		self.metadata = self.metadata[self.metadata['Audio_Emotion'].isin(self.emotions) & self.metadata['Text_Emotion'].isin(self.emotions)].reset_index(drop=True)

		self.feature_main_dir = None
		self.target_lengths = None
		self.modality_dims = None
		if self.use_extracted_features:
			if self.feature_names is None:
				raise ValueError("feature_names must be provided when use_extracted_features is True")
			if feature_dir is None:
				feature_dir = dataset_dir / 'features'
			self.feature_main_dir, _ = setup_feature_directories(feature_dir, self.feature_names)
			self.target_lengths = setup_target_lengths(self.feature_names, target_length_seconds)
			sample = self.metadata.iloc[0]
			self.modality_dims = get_modality_dims(
				sample,
				self.feature_names,
				feature_dir=self.feature_main_dir,
				n_qwen3_features=self.n_qwen3_features,
			)

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, idx):
		row = self.metadata.loc[idx]
		file_path = row['file_path']
		text = row['Utterance'].strip()
		audio_emotion = row['Audio_Emotion'].strip().lower()
		text_emotion = row['Text_Emotion'].strip().lower()

		if not self.use_extracted_features:
			audio_path = self.audio_dir / file_path
			audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
			# keep audio as a CPU numpy-backed tensor (no grads)
			audio = torch.from_numpy(audio).float()
			if self.rms_norm:
				rms = audio.pow(2).mean().sqrt()
				if rms > 0:
					audio = audio * (self.target_rms / rms)
			return {
				'file': file_path,
				'audio': audio,
				'text': text,
				'audio_label': audio_emotion,
				'text_label': text_emotion,
			}

		features = {}
		for modality, feature_name in self.feature_names.items():
			if feature_name == 'none':
				features[feature_name] = torch.full((1, 1), float('nan'))
				features['has_' + modality] = torch.tensor(False)
				continue

			filestem = Path(file_path).stem
			feature_path = Path(self.feature_main_dir) / feature_name / f"{filestem}.npy"
			try:
				feature_data = load_feature_file(feature_path, feature_name, n_qwen3_features=self.n_qwen3_features)
				features['has_' + modality] = torch.tensor(True)
				truncate_from_start = self.truncate_from_start or self.debug
				feature_data = apply_truncation(feature_data, self.target_lengths[feature_name], from_start=truncate_from_start)
				feature_data = apply_padding(feature_data, self.target_lengths[feature_name])
				features[feature_name] = feature_data
			except Exception:
				features[feature_name] = torch.full((self.target_lengths[feature_name], self.modality_dims[modality]), float('nan'))
				features['has_' + modality] = torch.tensor(False)

		labels = {
			'label': float('nan'),
			'audio': audio_emotion,
			'text': text_emotion,
		}

		return (
			{
				'audio_feature': features[self.feature_names['audio']],
				'text_feature': features[self.feature_names['text']],
				'has_audio': features['has_audio'],
				'has_text': features['has_text'],
			},
			labels,
			file_path,
			"CASE",
		)


class GoEmotionsDataset(Dataset):
	def __init__(self, debug=False, half=None):

		processed_output_path = Path("../../../../../datasets/public/GoEmotions/metadata_split.csv")
		
		if not processed_output_path.exists():
			from datasets import load_dataset
			self.data = load_dataset("go_emotions")

			labels = self.data['train'].features['labels'].feature.names
			idx2label = {idx: label.lower().strip() for idx, label in enumerate(labels)}
			label2idx = {label: idx for idx, label in idx2label.items()}
			neutral_idx = label2idx['neutral']

			expanded_data = []
			for split_name, data in self.data.items():
				for item in tqdm(data, desc=f"Processing {split_name} split"):
					label_ids = item['labels']
					if len(label_ids) > 1 and neutral_idx in label_ids:
						# If there are multiple labels and one of them is neutral, remove neutral
						label_ids = [id for id in label_ids if id != neutral_idx]
					for label_id in label_ids:
						expanded_data.append({
							'file_path': item['id'],
							'Utterance': item['text'],
							'Emotion': idx2label[label_id].strip().lower(),
							'split': split_name,
							'unique_utterance': True
						})
			self.data = pd.DataFrame(expanded_data)
			self.data = self.data[['split', 'file_path', 'Emotion', 'Utterance', 'unique_utterance']]

			self.data.to_csv(processed_output_path, index=False)
			print(f"Processed raw CSV and saved to {processed_output_path}")
		else:
			self.data = pd.read_csv(processed_output_path)

		if half == 1:
			self.data = self.data.iloc[:len(self.data) // 2].reset_index(drop=True)
		elif half == 2:
			self.data = self.data.iloc[len(self.data) // 2:].reset_index(drop=True)

		if debug:
			self.data = self.data.head(4)
		

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):

		audio = np.array([])  # Dummy audio since this dataset is for text only
		transcript = self.data.loc[idx, 'Utterance']
		labels = self.data.loc[idx, 'Emotion'].split(',')
		labels = [label.strip().lower() for label in labels]
		file_stem = self.data.loc[idx, 'file_path']

		return {'audio_raw': audio, 'text_raw': transcript}, labels, file_stem


class FeatureDatasetLazy(Dataset):
	def __init__(self, 
				 metadata_path, 
				 feature_dir,
				 feature_names,
				 split,
				 dataset_balancing=False,
				 debug=False,
				 target_length_seconds=10,
				 label_balancing=False,
				 case_mapping=False,
				 n_qwen3_features=None,
				 whisper_feature_len=128,
				no_neutral=False,
				exclude_msp_test=True,
				exclude_msp_train=True,
				drop_prob=False,
				 consolidate_test=False,
					 datasets_to_merge=None
				 ):

		
		self.feature_names = feature_names
		self.debug = debug
		self.split = split
		self.n_qwen3_features = n_qwen3_features
		self.whisper_feature_len = whisper_feature_len
		self.drop_prob = drop_prob

		metadata_path = Path(metadata_path)
		if 'merged' in metadata_path.stem:
			self.dataset_name = 'merged'
		else:
			self.dataset_name = metadata_path.parent.name

		# Setup directories and target lengths
		if feature_dir is None:
			# Merged dataset case, handle in __getitem__
			self.feature_main_dir, self.feature_sub_dirs = None, None
		else:
			self.feature_main_dir, self.feature_sub_dirs = setup_feature_directories(feature_dir, feature_names)
		self.target_lengths = setup_target_lengths(feature_names, target_length_seconds)
		
		# Load and process metadata
		metadata_path = Path(metadata_path)
		self.metadata = pd.read_csv(metadata_path, low_memory=False)

		if 'unique_utterance' not in self.metadata.columns:
			self.metadata['unique_utterance'] = self.dataset_name in ('IEMOCAP', 'MELD', 'MSP_podcast')

		if 'dataset_name' not in self.metadata.columns:
			self.metadata['dataset_name'] = self.dataset_name

		if self.dataset_name == 'merged':
			if exclude_msp_train:
				self.metadata = self.metadata[self.metadata['dataset_name'] != 'MSP_podcast'].reset_index(drop=True)
			elif exclude_msp_test:
					self.metadata = self.metadata[~(self.metadata['dataset_name'].isin(['MSP_podcast_test1', 'MSP_podcast_test2']))].reset_index(drop=True)

		if datasets_to_merge is not None and self.dataset_name == 'merged':
			self.metadata = self.metadata[self.metadata['dataset_name'].str.lower().isin([d.lower() for d in datasets_to_merge])].reset_index(drop=True)

		if self.dataset_name == 'IEMOCAP':
			self.metadata = modify_iemocap_labels(self.metadata)
		elif case_mapping:
			# self.metadata = modify_iemocap_labels(self.metadata)
			emotion_mapping = get_emotion_mapping()
			if 'Emotion' not in self.metadata.columns:
				self.metadata['Emotion'] = self.metadata['Text_Emotion']
			self.metadata['Emotion'] = self.metadata['Emotion'].str.lower().map(emotion_mapping)

		if no_neutral:
			# Throw away samples with label "neutral"
			self.metadata = self.metadata[self.metadata['Emotion'] != 'neutral'].reset_index(drop=True)

		# Throw away samples with label "remove"
		self.metadata = self.metadata[self.metadata['Emotion'] != 'remove'].reset_index(drop=True)

		# Setup emotion mappings
		self.emotions, self.emotion_to_idx, self.idx_to_emotion = setup_emotion_mappings(self.metadata)

		# Filter by split
		if consolidate_test and split == 'train':
			self.metadata = self.metadata[self.metadata['split'].isin(['train', 'test'])].reset_index(drop=True)
		else:
			self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)

		balanced_filepaths = None
		if case_mapping and split != 'dev':
			balanced_filepaths = balance_unique_pairs(self.metadata)

		if feature_dir is None:
			# Merged dataset: there is a lot to verify so just skip
			pass
		else:
			pass
				
		# Extract modality dimensions from the first sample
		idx_with_text = self.metadata['unique_utterance']
		if idx_with_text.any():     
			# if there is any useful text data, get the sample that has both text and audio
			sample = self.metadata[idx_with_text].iloc[0]
		else:
			# if not, get the first sample (with audio, no text)
			sample = self.metadata.iloc[0]
		self.modality_dims = get_modality_dims(sample, self.feature_names, feature_dir=self.feature_main_dir, n_qwen3_features=self.n_qwen3_features)

		if label_balancing:
			raise NotImplementedError("Label balancing is currently not implemented for the merged dataset due to complexity with multiple datasets and case mapping. Please set label_balancing=False or use a non-merged dataset.")
		elif balanced_filepaths is not None:
			self.filepaths = balanced_filepaths
		else:
			self.filepaths = self.metadata['file_path'].tolist()

		self.metadata = self.metadata.set_index('file_path')


	def get_idx_to_emotion_map(self):
		return self.idx_to_emotion
	
	def get_emotions(self):
		return self.emotions

	def get_modality_dims(self):
		return self.modality_dims

	def __len__(self):
		return len(self.filepaths)
	
	def get_naive_micro_baseline(self):
		return get_naive_micro_baseline_from_metadata(self.metadata)

	def get_files_to_utterances(self):
		return get_files_to_utterances(self.metadata)

	def __getitem__(self, idx):
		
		filename = self.filepaths[idx]
		filestem = Path(filename).stem

		row = self.metadata.loc[filename]
		# GoEmotions can have multiple entries with the same id because multiple co-existing labels are split up.
		# So we take random sample.
		if isinstance(row, pd.DataFrame):
			row = row.sample(n=1).iloc[0]

		dataset = row['dataset_name']

		labels = {
			'label': float('nan'),
			'audio': float('nan'),
			'text': float('nan'),
		}
		if 'Emotion' in row:
			emotion = str(row['Emotion']).strip().lower()
			labels['label'] = torch.tensor(self.emotion_to_idx[emotion], dtype=torch.long)
		if 'Audio_Emotion' in row and 'Text_Emotion' in row:
			labels['audio'] = str(row['Audio_Emotion']).strip().lower()
			labels['text'] = str(row['Text_Emotion']).strip().lower()
		
		features = {}
		
		if 'unique_audio' not in row:
			has_audio = self.dataset_name != 'GoEmotions'
		else:
			has_audio = row['unique_audio']
		has_text = row['unique_utterance']
		drop_choice = None
		if has_audio and has_text and self.drop_prob > np.random.rand():
			drop_choice = 'audio' if np.random.rand() < 0.5 else 'text'

		for modality, feature_name in self.feature_names.items():
			if feature_name == 'none':
				features[feature_name] = torch.full((1, 1), float('nan'))
				features['has_' + modality] = torch.tensor(False)
				continue

			drop_text = (modality == 'text') and ((drop_choice == 'text') or (not has_text))
			drop_audio = (modality == 'audio') and ((drop_choice == 'audio') or (not has_audio))

			if drop_text:
				features[feature_name] = torch.full((self.target_lengths[feature_name], self.modality_dims[modality]), float('nan'))
				features['has_' + modality] = torch.tensor(False)
				continue
			
			if drop_audio:
				features[feature_name] = torch.full((self.target_lengths[feature_name], self.modality_dims[modality]), float('nan'))
				features['has_' + modality] = torch.tensor(False)
				continue

			if self.feature_main_dir is None:
				feature_path = Path(row['dataset_dir']) / 'features' / feature_name / f"{filestem}.npy"
			else:
				feature_path = Path(self.feature_main_dir) / feature_name / f"{filestem}.npy"
			try:
				feature_data = load_feature_file(feature_path, feature_name, n_qwen3_features=self.n_qwen3_features)
				features['has_' + modality] = torch.tensor(True)
				truncate_from_start = self.debug or 'test' in self.split
				feature_data = apply_truncation(feature_data, self.target_lengths[feature_name], from_start=truncate_from_start)
				feature_data = apply_padding(feature_data, self.target_lengths[feature_name])
				features[feature_name] = feature_data
			except Exception:
				features[feature_name] = torch.full((self.target_lengths[feature_name], self.modality_dims[modality]), float('nan'))
				features['has_' + modality] = torch.tensor(False)
		
		return {
			'audio_feature': features[self.feature_names['audio']], 
			'text_feature': features[self.feature_names['text']],
			'has_audio': features['has_audio'],
			'has_text': features['has_text'],
		}, labels, filename, dataset


class FeatureDatasetExhaustiveIterable(IterableDataset):
	def __init__(self,
				 metadata_path,
				 feature_dir,
				 feature_names,
				 split,
				 qwen_layer_range=None,
				 debug=False,
				 target_length_seconds=10,
				 overlap=0.0,
				 drop_last=False,
				 n_qwen3_features=None,
				 case_mapping=False,
				 no_neutral=False,
				 exclude_msp_test=True,
				 shuffle=False,
				datasets_to_merge=None
				 ):

		metadata_path = Path(metadata_path)
		self.dataset_name = metadata_path.parent.name

		self.debug = debug
		self.feature_names = feature_names
		self.split = split
		self.qwen_layer_range = qwen_layer_range
		self.n_qwen3_features = n_qwen3_features

		metadata_path = Path(metadata_path)
		if 'merged' in metadata_path.stem:
			self.dataset_name = 'merged'
		else:
			self.dataset_name = metadata_path.parent.name

		# Setup directories and target lengths
		if feature_dir is None:
			# Merged dataset case, handle in __getitem__
			self.feature_main_dir, self.feature_sub_dirs = None, None
		else:
			self.feature_main_dir, self.feature_sub_dirs = setup_feature_directories(feature_dir, feature_names)
		self.target_lengths = setup_target_lengths(feature_names, target_length_seconds)

		self.overlap_ratio = float(overlap)
		self.drop_last = drop_last

		metadata_path = Path(metadata_path)
		self.metadata = pd.read_csv(metadata_path)
		self.metadata['Emotion'] = self.metadata['Emotion'].astype(str).str.strip().str.lower()

		if 'unique_utterance' not in self.metadata.columns:
			self.metadata['unique_utterance'] = self.dataset_name in ('IEMOCAP', 'MELD', 'MSP_podcast')

		if 'dataset_name' not in self.metadata.columns:
			self.metadata['dataset_name'] = self.dataset_name

		if exclude_msp_test and self.dataset_name == 'merged':
			self.metadata = self.metadata[~(self.metadata['dataset_name'].isin(['MSP_podcast_test1', 'MSP_podcast_test2']))].reset_index(drop=True)

		if datasets_to_merge is not None and self.dataset_name == 'merged':
			self.metadata = self.metadata[self.metadata['dataset_name'].str.lower().isin([d.lower() for d in datasets_to_merge])].reset_index(drop=True)

		if self.dataset_name == 'IEMOCAP':
			self.metadata = modify_iemocap_labels(self.metadata)
		elif case_mapping:
			emotion_mapping = get_emotion_mapping()
			self.metadata['Emotion'] = self.metadata['Emotion'].str.lower().map(emotion_mapping)

		if no_neutral:
			# Throw away samples with label "neutral"
			self.metadata = self.metadata[self.metadata['Emotion'] != 'neutral'].reset_index(drop=True)

		# Throw away samples with label "remove"
		self.metadata = self.metadata[self.metadata['Emotion'] != 'remove'].reset_index(drop=True)

		self.emotions, self.emotion_to_idx, self.idx_to_emotion = setup_emotion_mappings(self.metadata)

		self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)

		# Extract modality dimensions
		idx_with_text = self.metadata['unique_utterance']
		if idx_with_text.any():
			# if there is any useful text data, get the sample that has both text and audio
			sample = self.metadata[idx_with_text].iloc[0]
		else:
			# if not, get the first sample (with audio, no text)
			sample = self.metadata.iloc[0]
		self.modality_dims = get_modality_dims(sample, self.feature_names, feature_dir=self.feature_main_dir, n_qwen3_features=self.n_qwen3_features)

		self.audio_feature_key = self.feature_names['audio']
		self.text_feature_key = self.feature_names['text']

		if shuffle:
			self.metadata = self.metadata.sample(frac=1, random_state=42).reset_index(drop=True)


	def __len__(self):
		# This is not well-defined for an iterable dataset, but we can return the number of unique utterances as a proxy
		return len(self.metadata['unique_utterance'])

	def __iter__(self):
		audio_key = self.audio_feature_key
		text_key = self.text_feature_key
		target_lengths = self.target_lengths
		overlap_ratio = self.overlap_ratio
		drop_last = self.drop_last

		for idx, row in self.metadata.iterrows():
			filename = row['file_path'].split('/')[-1].split('.')[0]

			# Load text feature on-the-fly
			if text_key == 'none':
				text = torch.full((1, 1), float('nan'))
				has_text = torch.tensor(False)
			else:
				if not row['unique_utterance']:
					text = torch.full((target_lengths[text_key], self.modality_dims['text']), float('nan'))
					has_text = torch.tensor(False)
				else:
					if self.feature_sub_dirs is None:
						# Merged dataset, get path from metadata
						text_path = Path(row['dataset_dir']) / 'features' / text_key / f"{filename}.npy"
					else:
						text_path = self.feature_sub_dirs[text_key] / f"{filename}.npy"

					if text_path.exists():
						text = load_feature_file(text_path, text_key, n_qwen3_features=self.n_qwen3_features)
						text = apply_truncation(text, target_lengths[text_key], from_start=True)
						text = apply_padding(text, target_lengths[text_key])
						has_text = torch.tensor(True)
					else:
						text = torch.full((target_lengths[text_key], self.modality_dims['text']), float('nan'))
						has_text = torch.tensor(False)

			# Label
			labels = {
				'label': float('nan'),
				'audio': float('nan'),
				'text': float('nan'),
			}
			if 'Emotion' in row:
				label_idx = self.emotion_to_idx.get(row['Emotion'], 0)
				labels['label'] = torch.tensor(label_idx, dtype=torch.long)
			if 'Audio_Emotion' in row and 'Text_Emotion' in row:
				labels['audio'] = str(row['Audio_Emotion']).strip().lower()
				labels['text'] = str(row['Text_Emotion']).strip().lower()

			# Load audio feature on-the-fly
			if audio_key == 'none':
				audio = torch.full((1, 1), float('nan'))
				has_audio = torch.tensor(False)
				yield {'audio_feature': audio, 'text_feature': text, 'has_audio': has_audio, 'has_text': has_text}, labels, row['file_path'], row['dataset_name']
				continue

			if 'unique_audio' not in row:
				has_audio = self.dataset_name != 'GoEmotions'
			else:
				has_audio = row['unique_audio']

			if not has_audio:
				audio = torch.full((target_lengths[audio_key], self.modality_dims['audio']), float('nan'))
				has_audio = torch.tensor(False)
				yield {'audio_feature': audio, 'text_feature': text, 'has_audio': has_audio, 'has_text': has_text}, labels, row['file_path'], row['dataset_name']
				continue

			if self.feature_sub_dirs is None:
				# Merged dataset, get path from metadata
				audio_path = Path(row['dataset_dir']) / 'features' / audio_key / f"{filename}.npy"
			else:
				audio_path = self.feature_sub_dirs[audio_key] / f"{filename}.npy"

			if not audio_path.exists():
				audio = torch.full((target_lengths[audio_key], self.modality_dims['audio']), float('nan'))
				has_audio = torch.tensor(False)
				yield {'audio_feature': audio, 'text_feature': text, 'has_audio': has_audio, 'has_text': has_text}, labels, row['file_path'], row['dataset_name']
				continue

			data = np.load(audio_path)
			has_audio = torch.tensor(True)

			data = torch.from_numpy(data).float()
			if data.ndim == 1:
				data = data.unsqueeze(0)

			if audio_key == 'whisper_tiny_audio':
				# For whisper, we do learned pooling in the model so just pad and truncate to fixed length
				data = apply_truncation(data, target_lengths[audio_key], from_start=True)
				data = apply_padding(data, target_lengths[audio_key])
				yield {'audio_feature': data, 'text_feature': text, 'has_audio': has_audio, 'has_text': has_text}, labels, row['file_path'], row['dataset_name']
				continue

			total_frames = data.shape[0]
			target_length = target_lengths.get(audio_key, total_frames)
			step = max(1, target_length - int(overlap_ratio * target_length))

			start = 0
			while start < total_frames:
				end = start + target_length
				if end <= total_frames:
					chunk = data[start:end, :]
					yield {'audio_feature': chunk, 'text_feature': text, 'has_audio': has_audio, 'has_text': has_text}, labels, row['file_path'], row['dataset_name']
					start += step
				else:
					if drop_last:
						break
					else:
						padding = target_length - (total_frames - start)
						chunk = data[start:total_frames, :]
						chunk = torch.nn.functional.pad(chunk, (0, 0, 0, padding))
						yield {'audio_feature': chunk, 'text_feature': text, 'has_audio': has_audio, 'has_text': has_text}, labels, row['file_path'], row['dataset_name']
						break

	def get_idx_to_emotion_map(self):
		return self.idx_to_emotion

	def get_emotions(self):
		return self.emotions

	def get_modality_dims(self):
		return self.modality_dims

	def get_files_to_utterances(self):
		return get_files_to_utterances(self.metadata)


def get_files_to_utterances(metadata):
	"""
	Convert metadata into a dictionary mapping file paths to utterances.

	Args:
		metadata (pd.DataFrame): Metadata containing 'file_path' and 'Utterance' columns.

	Returns:
		dict: A dictionary mapping file paths to utterances.
	"""
	table = metadata.reset_index()[['file_path', 'Utterance']]
	for cp1252_char, replacement in cp1252_map.items():
		table['Utterance'] = table['Utterance'].str.replace(cp1252_char, replacement)
	output_dict = table.set_index('file_path')['Utterance'].to_dict()
	return output_dict


def merged_collate(batch):

	audio_key = next((key for key in batch[0][0].keys() if key in ['audio', 'audio_raw', 'audio_feature']), None)
	if audio_key is None:
		raise ValueError("safe_collate expects one of the keys 'audio', 'audio_raw', or 'audio_feature' in the batch items.")
	text_key = next((key for key in batch[0][0].keys() if key in ['text', 'text_raw', 'text_feature']), None)
	if text_key is None:
		raise ValueError("safe_collate expects one of the keys 'text', 'text_raw', or 'text_feature' in the batch items.")
	# Check if all values for 'audio_feature' and 'text_feature' are None

	# Only drop samples where both audio and text are NaN or None
	batch = [
		b for b in batch
		if not (
			(isinstance(b[0][audio_key], torch.Tensor) and torch.isnan(b[0][audio_key]).all()) and
			(isinstance(b[0][text_key], torch.Tensor) and torch.isnan(b[0][text_key]).all())
		)
	]

	if len(batch) == 0:
		return None
	return torch.utils.data._utils.collate.default_collate(batch)


def safe_collate(batch):

	audio_key = next((key for key in batch[0][0].keys() if key in ['audio', 'audio_raw', 'audio_feature']), None)
	if audio_key is None:
		raise ValueError("safe_collate expects one of the keys 'audio', 'audio_raw', or 'audio_feature' in the batch items.")
	text_key = next((key for key in batch[0][0].keys() if key in ['text', 'text_raw', 'text_feature']), None)
	if text_key is None:
		raise ValueError("safe_collate expects one of the keys 'text', 'text_raw', or 'text_feature' in the batch items.")
	# Check if all values for 'audio_feature' and 'text_feature' are None
	all_audio_nan = all(
		(torch.isnan(b[0][audio_key]).all() if isinstance(b[0][audio_key], torch.Tensor) else not isinstance(b[0][audio_key], str))
		for b in batch
	)
	all_text_nan = all(
		(torch.isnan(b[0][text_key]).all() if isinstance(b[0][text_key], torch.Tensor) else not isinstance(b[0][text_key], str))
		for b in batch
	)

	# If at least one value is not NaN, drop elements with NaN for these keys
	if not all_audio_nan:
		batch = [b for b in batch if not (isinstance(b[0][audio_key], torch.Tensor) and torch.isnan(b[0][audio_key]).all())]
	if not all_text_nan:
		batch = [b for b in batch if not (isinstance(b[0][text_key], torch.Tensor) and torch.isnan(b[0][text_key]).all())]

	if len(batch) == 0:
		return None
	return torch.utils.data._utils.collate.default_collate(batch)




class RawDataset(Dataset):
	def __init__(self, metadata_path, split, target_sr, min_duration=0, max_duration=float('inf'), mono=True, target_length_seconds=8, overfit=False, start_at_beginning=False, no_text=False, no_audio=False, rms_norm=True, target_rms=0.1):
		self.overfit = overfit
		self.start_at_beginning = start_at_beginning
		metadata_path = Path(metadata_path)
		self.dataset_dir = metadata_path.parent / 'audios'

		dataset_name = metadata_path.parent.name

		self.metadata = pd.read_csv(metadata_path)

		if dataset_name == 'CASE':
			self.metadata['Emotion'] = self.metadata['Text_Emotion']
			self.metadata['duration'] = 1

		self.emotions = sorted(self.metadata['Emotion'].str.lower().unique().tolist())

		if split != 'all':
			self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
		
		# Filter out rows where the audio file does not exist
		existing_rows = []
		for row in self.metadata.itertuples():
			audio_path = self.dataset_dir / row.file_path
			if audio_path.exists():
				existing_rows.append(row.Index)
			else:
				print(f"Audio file not found: {audio_path}")
		self.metadata = self.metadata.loc[existing_rows].reset_index(drop=True)

		self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
		self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}

		self.metadata['duration'] = pd.to_numeric(self.metadata['duration'], errors='coerce')
		self.metadata = self.metadata[self.metadata['duration'] > 0].reset_index(drop=True)
		self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
		self.min_duration = min_duration
		self.max_duration = max_duration
		self.target_sr = target_sr
		self.mono = mono
		

		self.no_text = no_text
		self.no_audio = no_audio
		self.rms_norm = rms_norm
		self.target_rms = target_rms

		if self.no_audio:
			self.target_length = -1
		else:
			self.target_length = int(target_length_seconds * target_sr)

	def get_emotions(self):
		return self.emotions

	def convert_idx_to_emotion(self, idx):
		return self.idx_to_emotion[idx]

	def __len__(self):
		return len(self.metadata)

	def get_naive_micro_baseline(self):
		return get_naive_micro_baseline_from_metadata_with_idx(self.metadata, self.emotion_to_idx)

	def __getitem__(self, idx):
		if self.overfit:
			idx = 0
		sample = self.metadata.loc[idx]
		emotion = str(sample['Emotion']).strip().lower()
		if not self.emotions:
			raise RuntimeError("A lista de emoções está vazia! Chama fill_emotions_from_csv(metadata_path) antes de usar o dataset.")
		label_idx = self.emotion_to_idx[emotion]
		labels = {
			'label': torch.tensor(label_idx, dtype=torch.long),
			'audio': float('nan'),
			'text': float('nan'),
		}
		if 'Audio_Emotion' in sample and 'Text_Emotion' in sample:
			labels['audio'] = str(sample['Audio_Emotion']).strip().lower()
			labels['text'] = str(sample['Text_Emotion']).strip().lower()
		assert 0 <= label_idx < len(self.emotions), f"Label fora do intervalo! label={label_idx}, n_classes={len(self.emotions)}"
		file_stem = str(Path(sample['file_path']).stem)
		
		full_path = self.dataset_dir / sample['file_path']
		# audio, sr = torchaudio.load(full_path, normalize=True)

		if self.no_text:
			transcript = ""
		else:
			transcript = fix_cp1252(sample['Utterance'])

		if self.no_audio:
			audio = torch.tensor([0.0])
		else:
			audio, sr = librosa.load(full_path, sr=self.target_sr, mono=self.mono)
			audio = torch.from_numpy(audio).float()

			if self.rms_norm:
				rms = audio.pow(2).mean().sqrt()
				if rms > 0:
					audio = audio * (self.target_rms / rms)

			# Adjust audio length to self.target_length
			if self.target_length > 0:
				if audio.shape[0] > self.target_length:
					start = 0 if self.overfit or self.start_at_beginning else torch.randint(0, audio.shape[0] - self.target_length + 1, (1,)).item()
					audio = audio[start:start + self.target_length]
				elif audio.shape[0] < self.target_length:
					padding = self.target_length - audio.shape[0]
					audio = torch.nn.functional.pad(audio, (0, padding))
		return {'audio_raw': audio, 'text_raw': transcript}, labels, file_stem



class AudioFeatureDataset(Dataset):
	def __init__(self, 
				 metadata_path, 
				 split,
				 min_duration=0, 
				 max_duration=float('inf'), 
				 target_sr=16000, 
				 mono=True, 
				 target_length_seconds=10, 
				 overfit=False, 
				 start_at_beginning=False):
		
		self.overfit = overfit
		self.start_at_beginning = start_at_beginning
		
		# Audio processing parameters
		self.target_sr = target_sr
		self.mono = mono
		self.target_length = target_length_seconds * target_sr
		self.min_duration = min_duration
		self.max_duration = max_duration

		metadata_path = Path(metadata_path)
		self.dataset_dir = metadata_path.parent / 'audios'

		self.metadata = pd.read_csv(metadata_path)
		self.emotions = sorted(self.metadata['Emotion'].unique().tolist())
		self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
		
		# Filter out rows where the audio file does not exist
		existing_rows = []
		for row in self.metadata.itertuples():
			audio_path = self.dataset_dir / row.file_path
			if audio_path.exists():
				existing_rows.append(row.Index)
			else:
				print(f"Audio file not found: {audio_path}")
		self.metadata = self.metadata.loc[existing_rows].reset_index(drop=True)

		self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions )}

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, idx):
		if self.overfit:
			idx = 0
			
		sample = self.metadata.loc[idx]
		
		# Check duration constraints
		if sample['duration'] < self.min_duration or sample['duration'] > self.max_duration:
			return None
			
		# Load audio
		full_path = self.dataset_dir / sample['file_path']
		# audio, sr = torchaudio.load(full_path, normalize=True)
		audio, sr = librosa.load(full_path, sr=None)
		audio = torch.from_numpy(audio).float()
		
		# Convert to mono if needed
		if self.mono and audio.shape[0] > 1:
			audio = torch.mean(audio, dim=0, keepdim=False)
			
		# Resample if needed
		if sr != self.target_sr:
			resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
			audio = resampler(audio)

		# Adjust audio length
		if self.target_length > 0:
			if audio.shape[0] > self.target_length:
				if self.overfit or self.start_at_beginning:
					start = 0
				else:
					start = torch.randint(0, audio.shape[0] - self.target_length + 1, (1,)).item()
				audio = audio[start:start + self.target_length]
			elif audio.shape[0] < self.target_length:
				padding = self.target_length - audio.shape[0]
				audio = torch.nn.functional.pad(audio, (0, padding))

		# Convert to numpy for feature extraction
		audio_np = audio.numpy()
		
		# Extract all three features
		mfcc_features = u.get_mfcc(audio_np, sr=self.target_sr)
		mel_features = u.get_melspectrogram(audio_np, sr=self.target_sr)
		chroma_features = u.get_chroma(audio_np, sr=self.target_sr)
		
		# Convert back to tensors
		mfcc_tensor = torch.from_numpy(mfcc_features).float()
		mel_tensor = torch.from_numpy(mel_features).float()
		chroma_tensor = torch.from_numpy(chroma_features).float()

		# Get label and file info
		labels = {
			'label': torch.tensor(self.emotion_to_idx[sample['Emotion']], dtype=torch.long),
			'audio': float('nan'),
			'text': float('nan'),
		}
		file_stem = str(Path(sample['file_path']).stem)
		
		return {'mfcc': mfcc_tensor, 'mel': mel_tensor, 'chroma': chroma_tensor}, labels, file_stem
		# return mfcc_tensor, mel_tensor, chroma_tensor, label, file_stem


def get_label_frequencies(metadata_path):
	metadata = pd.read_csv(metadata_path)
	emotions = metadata[metadata['split'] == 'train']['Emotion'].to_numpy()
	unique_emotions, counts = np.unique(emotions, return_counts=True)
	frequencies = dict(zip(unique_emotions, counts / len(emotions)))
	return frequencies


def adjust_msp_splits(metadata_path):
	df = pd.read_csv(metadata_path)

	idx_msp = df['dataset_name'] == 'MSP_podcast'
	idx_test1 = df['split'] == 'test1'
	idx_test2 = df['split'] == 'test2'
	df.loc[idx_msp & (idx_test1 | idx_test2), 'dataset_name'] = \
		df.loc[idx_msp & (idx_test1 | idx_test2), 'dataset_name'] + '_' + df.loc[idx_msp & (idx_test1 | idx_test2), 'split']
	

	df.loc[idx_test1 | idx_test2, 'split'] = 'test'

	df.to_csv(metadata_path, index=False)
	print(f"Adjusted MSP splits in {metadata_path}.")


if __name__ == "__main__":
	# merge_metadata()
	# GoEmotionsDataset()

	# from dataset import CASEDataset
	ds = CASEDataset(debug=False)
	print("len:", len(ds))
	print("first row:", ds.metadata.head(3)[['Audio_Emotion','Text_Emotion']])
	print("any NaN audio:", ds.metadata['Audio_Emotion'].isna().any())
	print("any NaN text:", ds.metadata['Text_Emotion'].isna().any())