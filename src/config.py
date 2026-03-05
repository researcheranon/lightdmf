import argparse
import time
from pathlib import Path
import getpass
from datetime import datetime


parser = argparse.ArgumentParser(description="SER Training Config")
parser.add_argument('--note', type=str, default='', help='Optional note for the experiment')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--lr_finder_steps', type=int, default=-1, help='LR finder steps (0 disables)')
parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--output_dir', type=str, default='../output', help='Directory for saving logs and models')
parser.add_argument('--target_length_seconds', type=float, default=8)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--drop_prob', type=float, default=0.0, help='Randomly drop one modality during training for robustness')
parser.add_argument('--skip_test', action='store_true', help='Include test split in training and skip test evaluation')

parser.add_argument('--fusion_method', type=str, default='attention', choices=['transformer', 'attention', 'attention_sentinel', 'average', 'shared_attention'])
parser.add_argument('--d_model', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--text_model', type=str, default='roberta')
parser.add_argument('--audio_model', type=str, default='wavlm')
parser.add_argument('--max_lr', type=float, default=3e-3, help='Maximum learning rate for One Cycle LR scheduler')
parser.add_argument('--warmup_pct', type=float, default=0.05, help='Percentage of warmup steps for One Cycle LR scheduler')
parser.add_argument('--start_lr_factor', type=float, default=0.01, help='Starting learning rate factor (start_lr = max_lr * start_lr_factor)')
parser.add_argument('--final_lr_factor', type=float, default=0.001, help='Final learning rate factor (final_lr = max_lr * final_lr_factor)')


parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value')
parser.add_argument('--dataset', type=str, choices=['merged', 'crema_d', 'iemocap', 'meld', 'msp_podcast', 'ravdess', 'tess'])
parser.add_argument('--datasets_to_merge', nargs='+', type=str, default=None, help='Datasets to merge when using the "merged" dataset option')

parser.add_argument('--case_mapping', action='store_true', help='Use CASE dataset emotion mapping')
parser.add_argument('--use_preextracted_case_features', action='store_true', help='Use pre-extracted features for CASE dataset (only applicable if --case_mapping is enabled)')
parser.add_argument('--n_qwen3_features', type=int, default=256)
parser.add_argument('--whisper_embedding_len', type=int, default=128, help='Number of time-steps to average Whisper embeddings')
parser.add_argument('--modality_weights', nargs='+', type=float, default=[0.33, 0.33, 0.33], help='Weights for each modality when using average fusion')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model checkpoint to resume training')

parser.add_argument('--accuracy_averaging', type=str, default='macro', choices=['macro', 'micro'])
parser.add_argument('--label_balancing', action='store_true', help='Enable data multiplier based on class frequencies')
parser.add_argument('--loss_weighting', action='store_true', help='Enable loss weighting based on class frequencies')
parser.add_argument('--dataset_balancing', action='store_true', help='Enable data multiplier based on class frequencies')
parser.add_argument('--no_neutral', action='store_true', help='Disable neutral class')
parser.add_argument('--test_only', action='store_true', help='Only perform testing')
parser.add_argument('--num_workers', type=int, default=8, help='Number of DataLoader workers')
parser.add_argument('--seed', type=int, default=42, help='Random seed (0 for random)')
parser.add_argument('--log_step', type=int, default=200, help='Logging interval in steps')
parser.add_argument('--test_step', type=int, default=200, help='Testing interval in steps')
parser.add_argument('--max_test_step', type=int, default=-1, help='Max test steps per evaluation (-1 for all)')
parser.add_argument('--max_train_step', type=int, default=20000, help='Max train steps')
parser.add_argument('--pin_memory', action='store_true', help='Use pin memory for DataLoader')
parser.add_argument('--persistent_workers', action='store_true', help='Use persistent workers for DataLoader')
parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of samples loaded in advance by each worker')
parser.add_argument('--plot_step_zero', action='store_true', help='Plot performance at step zero')
parser.add_argument('--dont_plot_modality_losses', action='store_true', help='Do not plot audio/text/fusion losses when using multi-classifier')
parser.add_argument('--nonlinearity', type=str, default='gelu', choices=[None, 'relu', 'gelu'], help='Nonlinearity to apply after attention in fusion module (default: None)')

args = parser.parse_args()

if args.datasets_to_merge is not None and args.dataset != 'merged':
    raise ValueError("datasets_to_merge can only be specified when dataset is 'merged'")

# ==== MELD DATASET ACCESS ====
args.output_dir = Path(args.output_dir)
args.start_time = datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]
if args.debug:
    args.work_dir = args.output_dir / ("DEBUG_" + args.start_time)
elif args.no_cuda:
    args.work_dir = args.output_dir / ("CPU_" + args.start_time)
else:
    args.work_dir = args.output_dir / args.start_time

args = vars(args)
# Assert logic for audio_model and text_model
audio_none = args['audio_model'] == 'none'
text_none = args['text_model'] == 'none'
if audio_none and text_none:
    raise ValueError("Both audio_model and text_model cannot be None.")
if args['lr_finder_steps'] > 0 and (args['max_train_step'] <= 0 or args['max_train_step'] > args['lr_finder_steps']):
    args['max_train_step'] = args['lr_finder_steps']
    # Don't log or test during LR finder
    args['log_step'] = args['lr_finder_steps'] + 1
    args['test_step'] = args['lr_finder_steps'] + 1
    print(f"LR Finder enabled: Setting max_train_step to {args['lr_finder_steps']}")

if args['num_workers'] == 0:
    print("WARNING: num_workers=0. Runtime may be very slow.")


if args['dataset'] == 'crema_d':
    args['dataset_dir'] = Path('../../../../../datasets/public/CREMA_D')
elif args['dataset'] == 'iemocap':
    args['dataset_dir'] = Path('../../../../../datasets/public/IEMOCAP')
elif args['dataset'] == 'meld':
    args['dataset_dir'] = Path('../../../../../datasets/public/MELD')
elif args['dataset'] == 'ravdess':
    args['dataset_dir'] = Path('../../../../../datasets/public/RAVDESS')
elif args['dataset'] == 'tess':
    args['dataset_dir'] = Path('../../../../../datasets/public/TESS')

if args['dataset'] == 'msp_podcast':
    args['features_dir'] = Path('../../../../../datasets/public/MSP_podcast/features')
elif args['dataset'] == 'merged':
    args['features_dir'] = None  # Handled in dataset.py
else:
    args['features_dir'] = args['dataset_dir'] / 'features'

if args['num_workers'] == 0:
    args['prefetch_factor'] = None

# Some datasets have the same utterances for different emotions, so discard text features

audio_only_datasets = {'crema_d', 'ravdess', 'tess'}
if args['dataset'] in audio_only_datasets:
    args['text_model'] = 'none'
    print(f"Dataset {args['dataset']} has the same utterances for different emotions, so discarding text features.")

if args['dataset'] == 'merged' and args['datasets_to_merge']:
    merged_set = set(args['datasets_to_merge'])
    if merged_set.issubset(audio_only_datasets):
        args['text_model'] = 'none'
        print(
            "Merged dataset contains only audio-only datasets, so discarding text features."
        )

if args['dataset'] == 'merged':
    args['dataset_dir'] = None  # Handled in dataset.py
    if args['case_mapping']:
        args['metadata_path'] = '../metadata_merged_case.csv'
    else:
        args['metadata_path'] = '../metadata_merged.csv'
else:
    args['metadata_path'] = args['dataset_dir'] / 'metadata_split.csv'

if 'qwen' not in args['audio_model']:
    args['qwen_layer_range'] = None

if args['accuracy_averaging'] == 'macro' and not args['loss_weighting'] and not args['label_balancing'] and not args['test_only']:
    print("Specified macro accuracy averaging without loss weighting or label balancing. Turning on loss weighting.")
    args['loss_weighting'] = True

