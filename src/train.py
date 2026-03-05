print('Run started', flush=True)

import time
t0_import = time.time()

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # only show errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from models import (
    n_parameters,
    n_trainable_parameters,
)
from dataset import (
    FeatureDatasetLazy,
    get_label_frequencies,
    FeatureDatasetExhaustiveIterable,
    merged_collate,
)

from models import AttentionClassifier, AveragingClassifier
from eval_case import CASEEvaluator
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from config import args
import utils as u
import numpy as np
import random
from pathlib import Path
import contextlib

if args['seed'] > 0:
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    random.seed(args['seed'])
    torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id: int) -> None:
    if args['seed'] > 0:
        worker_seed = args['seed'] + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)


class SERTrainer:
    def __init__(self):
        self.train_step = 0
        self.n_sequences_total = 0
        self.init_hours = 0
        self.epoch = 0
        self.init_time = time.time()
        self.hours_total = 0 
        self.best_dev_accuracy = 0
        self.best_train_loss = float('inf')
        self.best_dev_loss = float('inf')
        self.csv_in = None

        self.logging = u.create_exp_dir(args['work_dir'], debug=args['debug'] or args['test_only'], print_=args['lr_finder_steps'] <= 0)
        if args['note'] != '' and not args['debug'] and not args['test_only']:
            # write the note in folder for ease of comparison and viewing
            open(os.path.join(args['work_dir'], 'aa ' + args['note']), "x").close()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_loader_generator = None
        if args['seed'] > 0:
            self.data_loader_generator = torch.Generator()
            self.data_loader_generator.manual_seed(args['seed'])
        
        # Data and model
        metadata_path = args['metadata_path']

        if args['checkpoint_path'] is not None:
            checkpoint_dir = Path(args['checkpoint_path']).parent
            config_path = checkpoint_dir / "config.pth"
            loaded_config = torch.load(config_path, weights_only=False)

            if args['audio_model'] != loaded_config['audio_model']:
                print(f"Warning: Audio model mismatch. Using {loaded_config['audio_model']} from checkpoint instead of {args['audio_model']}.")
                args['audio_model'] = loaded_config['audio_model']
            if args['text_model'] != loaded_config['text_model']:
                print(f"Warning: Text model mismatch. Using {loaded_config['text_model']} from checkpoint instead of {args['text_model']}.")
                args['text_model'] = loaded_config['text_model']
            if args['fusion_method'] != loaded_config['fusion_method']:
                print(f"Warning: Fusion method mismatch. Using {loaded_config['fusion_method']} from checkpoint instead of {args['fusion_method']}.")
                args['fusion_method'] = loaded_config['fusion_method']

        feature_names = {'audio': args['audio_model'], 'text': args['text_model']}

        collate_fn = merged_collate if args['dataset'] == 'merged' else None

        train_dataset = FeatureDatasetLazy(
            metadata_path, 
            args['features_dir'], 
            feature_names, 
            'train', 
            n_qwen3_features=args['n_qwen3_features'],
            debug=args['debug'], 
            dataset_balancing=args['dataset_balancing'],
            target_length_seconds=args['target_length_seconds'], 
            label_balancing=args['label_balancing'],
            case_mapping=args['case_mapping'],
            no_neutral=args['no_neutral'],
            drop_prob=args['drop_prob'],
            consolidate_test=args['skip_test'],
            datasets_to_merge=args['datasets_to_merge']
            )
        
        dev_dataset = FeatureDatasetLazy(
            metadata_path, args['features_dir'], 
            feature_names, 
            'dev', 
            n_qwen3_features=args['n_qwen3_features'],
            debug=args['debug'], 
            target_length_seconds=args['target_length_seconds'],
            case_mapping=args['case_mapping'],
            no_neutral=args['no_neutral'],
            datasets_to_merge=args['datasets_to_merge']
            )

        test_name = 'test'
        self.test2_loader = None
            
        if args['skip_test']:
            test_dataset = None
        else:
            test_dataset = FeatureDatasetExhaustiveIterable(
                metadata_path, 
                args['features_dir'], 
                feature_names, 
                test_name, 
                n_qwen3_features=args['n_qwen3_features'],
                debug=args['debug'], 
                target_length_seconds=args['target_length_seconds'], 
                overlap=0.0, 
                drop_last=False,
                case_mapping=args['case_mapping'],
                no_neutral=args['no_neutral'],
                shuffle=args['debug'],
                datasets_to_merge=args['datasets_to_merge']
                )


        if test_dataset is None:
            self.test_loader = None
        else:
            self.test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn,
                                  drop_last=False, num_workers=args['num_workers'], pin_memory=args['pin_memory'],
                                   persistent_workers=args['persistent_workers'], prefetch_factor=args['prefetch_factor'],
                                   worker_init_fn=_seed_worker, generator=self.data_loader_generator)

        self.emotions = train_dataset.get_emotions()
        args['n_labels'] = len(self.emotions)
        modality_dims = train_dataset.get_modality_dims()
        args['modality_dims'] = modality_dims
        self.idx_to_emotion = train_dataset.get_idx_to_emotion_map()
        if test_dataset is not None:
            assert self.idx_to_emotion == test_dataset.get_idx_to_emotion_map(), "Train and test datasets have different emotion label mappings, which can lead to incorrect evaluation. Please ensure they are consistent."
        args['idx_to_emotion'] = self.idx_to_emotion

        if not args['debug'] and not args['test_only']:
            torch.save(args, args['work_dir'] / 'config.pth')
        
        if args['fusion_method'] == 'attention':
            self.model = AttentionClassifier(
                modality_dims=modality_dims,
                d_model=args['d_model'],
                n_classes=len(self.emotions),
                n_heads=args['n_heads'],
                dropout=args['dropout'],
                nonlinearity=args['nonlinearity'],
                whisper_embedding_len=args['whisper_embedding_len'] if 'whisper' in args['audio_model'].lower() else -1,   
            ).to(self.device)
        elif args['fusion_method'] == 'average':
            self.model = AveragingClassifier(
                modality_dims=modality_dims,
                d_model=args['d_model'],
                n_classes=len(self.emotions),
                dropout=args['dropout'],
            ).to(self.device)

        self.case_evaluator = None
        if args['case_mapping']:
            self.case_evaluator = CASEEvaluator(
                classifier=self.model,
                model_config=self.model.model_config,
                config=args,
                debug=args['debug'],
                training=True, 
                use_extracted_features=args['use_preextracted_case_features'],
                use_whisper_text=args['use_whisper_text'],
                max_samples=args['max_test_step']
            )

        if not args['test_only']:
            self.model.train()
        
        if test_dataset is not None:
            self.files_to_utterances = test_dataset.get_files_to_utterances()
        else:
            self.files_to_utterances = dev_dataset.get_files_to_utterances()

        if not args['loss_weighting']:
            loss_weights = None
        else:
            label_frequencies = get_label_frequencies(metadata_path)
            # Build weights for each class in Emotions
            weights_list = []
            for emotion in self.emotions:
                freq = label_frequencies.get(emotion, 1.0 / len(self.emotions))
                weights_list.append(1.0 / freq if freq > 0 else 1.0)
            loss_weights = torch.tensor(weights_list, dtype=torch.float32)
            loss_weights = loss_weights / loss_weights.mean()  # normalize


        self.criterion = torch.nn.CrossEntropyLoss(weight=loss_weights, reduction='none').to(self.device)

        t0_loaders = time.time()

        if not args['test_only']:
            most_frequent_class, self.naive_micro_baseline = train_dataset.get_naive_micro_baseline()
            self.naive_macro_baseline = 1 / len(self.emotions)
            print(f"Naive Micro Baseline: Class {most_frequent_class} with accuracy {self.naive_micro_baseline:.4f}")
            print(f"Naive Macro Baseline: {self.naive_macro_baseline:.4f}")
            self.train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=False, 
                                       num_workers=args['num_workers'], collate_fn=collate_fn, pin_memory=args['pin_memory'],
                                       persistent_workers=args['persistent_workers'], prefetch_factor=args['prefetch_factor'],
                                       worker_init_fn=_seed_worker, generator=self.data_loader_generator
                                           )

            self.dev_loader = DataLoader(dev_dataset, batch_size=args['batch_size'], shuffle=False, drop_last=False, 
                                 num_workers=args['num_workers'], collate_fn=collate_fn, pin_memory=args['pin_memory'],
                                   persistent_workers=args['persistent_workers'], prefetch_factor=args['prefetch_factor'],
                                   worker_init_fn=_seed_worker, generator=self.data_loader_generator)

            start_lr = args['max_lr'] * args['start_lr_factor']
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=start_lr)

            self.lr_scheduler = OneCycleLR(
                self.optimizer,
                max_lr=args['max_lr'],
                total_steps=args['max_train_step']+1,
                pct_start=args['warmup_pct'],
                div_factor=1.0 / args['start_lr_factor'],
                final_div_factor=1.0 / args['final_lr_factor'],
                anneal_strategy='cos'
            )

            # Select the correct GradScaler depending on PyTorch version
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                GradScaler = torch.amp.GradScaler
            else:
                GradScaler = torch.cuda.amp.GradScaler
            self.scaler = GradScaler(enabled=False)

            n_trainable = n_trainable_parameters(self.model)
            n_total = n_parameters(self.model)

            args['n_parameters'] = n_total
            args['n_trainable_parameters'] = n_trainable

            csv_fields = [
                "epoch", "step", "hour", "lr", "train_loss", "dev_loss", 
                "test_loss", "test2_loss",
                'dev_micro_accuracy', 'test_micro_accuracy', 'test2_micro_accuracy', 
                'dev_macro_accuracy', 'test_macro_accuracy', 'test2_macro_accuracy',
                'dev_macro_f1', 'test_macro_f1', 'test2_macro_f1',
                'case_micro_accuracy_audio', 'case_micro_accuracy_text',
                'case_macro_f1_audio', 'case_macro_f1_text',
                'case_runtime_rate',
                'case_micro_accuracy_fusion_vs_audio', 'case_micro_accuracy_fusion_vs_text',
                'case_macro_f1_fusion_vs_audio', 'case_macro_f1_fusion_vs_text'
            ]
            if not args['dont_plot_modality_losses']:
                csv_fields += [
                    'train_audio_loss', 'train_text_loss', 'train_fusion_loss',
                    'dev_audio_loss', 'dev_text_loss', 'dev_fusion_loss',
                    'test_audio_loss', 'test_text_loss', 'test_fusion_loss',
                    'test2_audio_loss', 'test2_text_loss', 'test2_fusion_loss'
                ]
            self.csv_writer = u.CsvWriter(
                os.path.join(args['work_dir'], "performance.csv"),
                csv_fields,
                in_path=self.csv_in,
                debug=args['debug'] or args['test_only']
            )
            
            for k, v in args.items():
                self.logging(f"{k}: {v}")

            self.logging(f"Dataset lengths\nTrain: {len(train_dataset):,}")
            self.logging(f"Dev: {len(dev_dataset):,}")

            self.logging(f"Model has {n_trainable:,} trainable parameters out of {n_total:,} total parameters")
            self.logging(f"Device: {self.device}")

            if self.device == "cuda":
                self.logging(f"GPU: {u.get_gpu_name()}")

        
    def forward_pass(self, features_dict, target_batch=None, eval=False):
        # Move tensor features to device
        device_features = {}
        for key, value in features_dict.items():
            if isinstance(value, torch.Tensor):
                device_features[key] = value.to(self.device, non_blocking=True)
            else:
                device_features[key] = value  # Keep strings/None as is

        if target_batch is not None:
            target_batch = target_batch.to(self.device, non_blocking=True)

        with torch.amp.autocast(self.device, enabled=False):
            output = self.model(device_features)
            
            if target_batch is None:
                losses = None
                loss_components = None
            else:
                loss = 0
                loss_components = {
                    'audio': {'loss': float('nan'), 'count': 0},
                    'text': {'loss': float('nan'), 'count': 0},
                    'fusion': {'loss': float('nan'), 'count': 0},
                }

                # compute per-modality loss only on non-NaN logits
                audio_present = device_features['has_audio']
                if audio_present.any():
                    loss_audio = self.criterion(output['audio'][audio_present], target_batch[audio_present]).mean()
                    loss += args['modality_weights'][0] * loss_audio
                    loss_components['audio'] = {
                        'loss': loss_audio.detach().item(),
                        'count': int(audio_present.sum().item())
                    }

                text_present = device_features['has_text']
                if text_present.any():
                    loss_text = self.criterion(output['text'][text_present], target_batch[text_present]).mean()
                    loss += args['modality_weights'][1] * loss_text
                    loss_components['text'] = {
                        'loss': loss_text.detach().item(),
                        'count': int(text_present.sum().item())
                    }

                fusion_present = device_features['has_audio'] & device_features['has_text']
                if fusion_present.any():
                    loss_fusion = self.criterion(output['fusion'][fusion_present], target_batch[fusion_present]).mean()
                    loss += args['modality_weights'][2] * loss_fusion
                    loss_components['fusion'] = {
                        'loss': loss_fusion.detach().item(),
                        'count': int(fusion_present.sum().item())
                    }

        return output, loss, loss_components

    def train(self):
        total_loss = 0.0
        modality_loss_sums = {'audio': 0.0, 'text': 0.0, 'fusion': 0.0}
        modality_loss_counts = {'audio': 0, 'text': 0, 'fusion': 0}
        best_test_result_str = ""
        train_interval_start_time = time.time()
        test_loss = -1
        lr_finder_logger = None

        # Dummy context manager and step function when profiling is disabled
        prof = contextlib.nullcontext()
        prof_step = lambda: None # A function that does nothing
        record_fn = lambda name: contextlib.nullcontext()

        if args['lr_finder_steps'] > 0:
            print("Starting LR Finder...")
            lr_finder_logger = u.LRFinderLogger(total_steps=args['lr_finder_steps'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_finder_logger.init_lr


        train_iter = iter(self.train_loader)
        with prof:
            while True:
                with record_fn("DataLoader/IO"):
                    try:
                        features_batch, labels_dict, file_stem, dataset_name = next(train_iter)
                        label_batch = labels_dict['label']
                    except StopIteration:
                        # Handle end of epoch: re-initialize the iterator and get the first batch
                        train_iter = iter(self.train_loader) 
                        self.epoch += 1
                        features_batch, labels_dict, file_stem, dataset_name = next(train_iter)
                        label_batch = labels_dict['label']

                with record_fn("Forward/Backward/Optim"):
                    self.optimizer.zero_grad()

                    output, loss, loss_components = self.forward_pass(features_batch, label_batch)

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)

                    if args['clip'] > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args['clip'])

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.lr_scheduler.step()
                    
                    loss_value = loss.item()
                    total_loss += loss_value
                    if not args['dont_plot_modality_losses'] and loss_components is not None:
                        for key in ('audio', 'text', 'fusion'):
                            comp = loss_components.get(key, None)
                            if comp is not None and comp['count'] > 0 and not np.isnan(comp['loss']):
                                modality_loss_sums[key] += comp['loss'] * comp['count']
                                modality_loss_counts[key] += comp['count']

                    if args['lr_finder_steps'] > 0 and lr_finder_logger is not None:
                        lr = lr_finder_logger.compute_lr(self.train_step)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                        lr_finder_logger.log(lr, loss_value)
                        if (self.train_step + 1) >= args['lr_finder_steps']:
                            lr_finder_logger.plot(args['work_dir'])
                            self.logging(f"LR Finder complete. Plot saved to {args['work_dir'] / 'lr_find.png'}")
                            return test_loss

                    if (self.train_step % args['log_step'] == 0 and self.train_step > 0) or self.train_step == args['max_train_step']:
                        if self.train_step == args['log_step'] and self.train_step > 0:
                            self.logging(u.memory())
                        train_loss = total_loss / args['log_step']
                        elapsed = (time.time() - self.init_time) / 3600
                        elapsed_interval = time.time() - train_interval_start_time
                        ms_per_batch = (elapsed_interval * 1000) / args['log_step']
                        ms_per_sample = ms_per_batch / args['batch_size']
                        train_interval_start_time = time.time()
                        cur_lr = self.optimizer.param_groups[0]['lr']
                        self.logging(f"Step {self.train_step}, Epoch {self.epoch+1}, Train Loss: {train_loss:.4f}, Elapsed Hours: {elapsed:.3f}, MS/Batch: {ms_per_batch:.2f}, MS/Sample: {ms_per_sample:.2f}, LR: {cur_lr:.2e}")
                        if not args['debug']:
                            perf_update = {
                                "epoch": self.epoch + 1,
                                "step": self.train_step,
                                "hour": elapsed,
                                "lr": cur_lr,
                                "train_loss": train_loss,
                                "dev_loss": np.nan,
                                "dev_micro_accuracy": np.nan,
                                "dev_macro_accuracy": np.nan,
                                "dev_macro_f1": np.nan,
                                "test_loss": np.nan,
                                "test_micro_accuracy": np.nan,
                                "test_macro_accuracy": np.nan,
                                "test_macro_f1": np.nan,
                                "test2_loss": np.nan,
                                "test2_micro_accuracy": np.nan,
                                "test2_macro_accuracy": np.nan,
                                "test2_macro_f1": np.nan,
                                "case_micro_accuracy_audio": np.nan,
                                "case_micro_accuracy_text": np.nan,
                                "case_macro_f1_audio": np.nan,
                                "case_macro_f1_text": np.nan,
                                "case_micro_accuracy_fusion_vs_audio": np.nan,
                                "case_micro_accuracy_fusion_vs_text": np.nan,
                                "case_macro_f1_fusion_vs_audio": np.nan,
                                "case_macro_f1_fusion_vs_text": np.nan
                            }
                            if not args['dont_plot_modality_losses']:
                                for key in ('audio', 'text', 'fusion'):
                                    if modality_loss_counts[key] > 0:
                                        perf_update[f'train_{key}_loss'] = modality_loss_sums[key] / modality_loss_counts[key]
                                    else:
                                        perf_update[f'train_{key}_loss'] = np.nan
                                perf_update['dev_audio_loss'] = np.nan
                                perf_update['dev_text_loss'] = np.nan
                                perf_update['dev_fusion_loss'] = np.nan
                                perf_update['test_audio_loss'] = np.nan
                                perf_update['test_text_loss'] = np.nan
                                perf_update['test_fusion_loss'] = np.nan
                                perf_update['test2_audio_loss'] = np.nan
                                perf_update['test2_text_loss'] = np.nan
                                perf_update['test2_fusion_loss'] = np.nan
                            self.csv_writer.update(perf_update)
                            u.plot_performance(
                                os.path.join(args['work_dir'], "performance.csv"),
                                save=not args['debug'],
                                n_labels=len(self.emotions),
                                title=args['note'],
                                start_step=0 if args['plot_step_zero'] else 1,
                                plot_modality_losses=not args['dont_plot_modality_losses']
                            )

                        total_loss = 0.0
                        modality_loss_sums = {'audio': 0.0, 'text': 0.0, 'fusion': 0.0}
                        modality_loss_counts = {'audio': 0, 'text': 0, 'fusion': 0}

                        if train_loss < self.best_train_loss:
                            self.best_train_loss = train_loss


                    if (
                        self.train_step % args['test_step'] == 0
                        and args['lr_finder_steps'] <= 0
                    ):
                        test_interval_start_time = time.time()
                        results = self.test()
                        test_interval_end_time = time.time()
                        ms_per_batch = (test_interval_end_time - test_interval_start_time) * 1000 / results['num_batches']
                        del results['num_batches']
                        available_splits = results.pop('_available_splits', ['dev', 'test', 'test2'])
                        ms_per_sample = ms_per_batch / args['batch_size']
                        dev_loss = results.get('dev', {}).get('loss', np.nan)
                        dev_micro_accuracy = results.get('dev', {}).get('micro_accuracy', np.nan)
                        dev_macro_accuracy = results.get('dev', {}).get('macro_accuracy', np.nan)
                        dev_macro_f1 = results.get('dev', {}).get('macro_f1', np.nan)
                        test_loss = results.get('test', {}).get('loss', np.nan)
                        test_micro_accuracy = results.get('test', {}).get('micro_accuracy', np.nan)
                        test_macro_accuracy = results.get('test', {}).get('macro_accuracy', np.nan)
                        test_macro_f1 = results.get('test', {}).get('macro_f1', np.nan)

                        elapsed = (time.time() - self.init_time) / 3600

                        log_str = f"Step {self.train_step}, Epoch {self.epoch+1}, "
                        split_results = {k: results[k] for k in available_splits if k in results}

                        for split in split_results:
                            macro_f1 = split_results[split].get('macro_f1', np.nan)
                            log_str += (
                                f"{split.capitalize()} Loss: {split_results[split]['loss']:.4f}, "
                                f"{split.capitalize()} Micro Accuracy: {split_results[split]['micro_accuracy']:.4f}, "
                                f"{split.capitalize()} Macro Accuracy: {split_results[split]['macro_accuracy']:.4f}, "
                                f"{split.capitalize()} Macro F1: {macro_f1:.4f}, "
                            )
                        log_str += f"Elapsed Hours: {elapsed:.3f}, MS/Batch: {ms_per_batch:.2f}, MS/Sample: {ms_per_sample:.2f}"
                        self.logging(log_str)

                        performance_dict = {
                            "epoch": self.epoch + 1,
                            "step": self.train_step,
                            "hour": elapsed,
                            "lr": self.optimizer.param_groups[0]['lr'],
                            "train_loss": np.nan,
                        }

                        if not args['dont_plot_modality_losses']:
                            performance_dict['train_audio_loss'] = np.nan
                            performance_dict['train_text_loss'] = np.nan
                            performance_dict['train_fusion_loss'] = np.nan

                        for split in sorted(list(split_results.keys())):
                            for key in split_results[split]:
                                performance_dict[f"{split}_{key}"] = split_results[split][key]

                        if 'case' in results:
                            performance_dict['case_micro_accuracy_audio'] = results['case'].get('micro_accuracy_audio', np.nan)
                            performance_dict['case_micro_accuracy_text'] = results['case'].get('micro_accuracy_text', np.nan)
                            performance_dict['case_macro_f1_audio'] = results['case'].get('macro_f1_audio', np.nan)
                            performance_dict['case_macro_f1_text'] = results['case'].get('macro_f1_text', np.nan)
                            performance_dict['case_runtime_rate'] = results['case'].get('runtime_rate', np.nan)
                            performance_dict['case_micro_accuracy_fusion_vs_audio'] = results['case'].get('micro_accuracy_fusion_vs_audio', np.nan)
                            performance_dict['case_micro_accuracy_fusion_vs_text'] = results['case'].get('micro_accuracy_fusion_vs_text', np.nan)
                            performance_dict['case_macro_f1_fusion_vs_audio'] = results['case'].get('macro_f1_fusion_vs_audio', np.nan)
                            performance_dict['case_macro_f1_fusion_vs_text'] = results['case'].get('macro_f1_fusion_vs_text', np.nan)
                        else:
                            performance_dict['case_micro_accuracy_audio'] = np.nan
                            performance_dict['case_micro_accuracy_text'] = np.nan
                            performance_dict['case_macro_f1_audio'] = np.nan
                            performance_dict['case_macro_f1_text'] = np.nan
                            performance_dict['case_runtime_rate'] = np.nan
                            performance_dict['case_micro_accuracy_fusion_vs_audio'] = np.nan
                            performance_dict['case_micro_accuracy_fusion_vs_text'] = np.nan
                            performance_dict['case_macro_f1_fusion_vs_audio'] = np.nan
                            performance_dict['case_macro_f1_fusion_vs_text'] = np.nan

                        self.csv_writer.update(performance_dict)
                        if not args['debug']:
                            u.plot_performance(os.path.join(args['work_dir'], "performance.csv"), 
                                save=True, 
                                n_labels=len(self.emotions),
                                title=args['note'],
                                start_step=0 if args['plot_step_zero'] else 1,
                                plot_modality_losses=not args['dont_plot_modality_losses']
                                )
                        
                        if args['accuracy_averaging'] == 'micro':
                            dev_accuracy = dev_micro_accuracy
                            test_accuracy = test_micro_accuracy
                        else:
                            dev_accuracy = dev_macro_f1
                            test_accuracy = test_macro_f1

                        if args['skip_test']:
                            test_accuracy = dev_accuracy

                        if dev_accuracy > self.best_dev_accuracy:
                            self.best_dev_accuracy = dev_accuracy
                            
                            # Save best result in a file name
                            if best_test_result_str:
                                # delete previous
                                old_path = args['work_dir'] / best_test_result_str
                                if old_path.exists():
                                    old_path.unlink()

                            # Save new result
                            if args['case_mapping']:
                                best_test_result_str = f"CASE_{self.train_step}_{round(results['case']['micro_accuracy_text'], 4)}_{round(results['case']['micro_accuracy_audio'], 4)}"
                            elif args['skip_test']:
                                best_test_result_str = f"DEV_{args['accuracy_averaging']}_{self.train_step}_{round(dev_accuracy, 4)}"
                            else:
                                best_test_result_str = f"TST_{args['accuracy_averaging']}_{self.train_step}_{round(test_accuracy, 4)}"
                            new_path = args['work_dir'] / best_test_result_str
                            if not args['debug']:
                                open(new_path, 'x').close()
                                # Save the best model
                                self.model.save_model(args['work_dir'] / "model.pth")
                                
                        train_interval_start_time = time.time()
                                    
                    if self.train_step >= args['max_train_step'] and args['max_train_step'] > 0:
                        self.logging("Reached maximum training steps.")

                        return test_loss
                    self.train_step += 1
                    
                    
    def test(self):
        self.model.eval()

        with torch.no_grad():
            loaders = {}
            results = {}
            num_batches_all = 0
            if not args['skip_test']:
                loaders['test'] = self.test_loader
                results['test'] = {}
            if self.test2_loader is None:
                results['test2'] = {'loss': np.nan, 'micro_accuracy': np.nan, 'macro_accuracy': np.nan, 'macro_f1': np.nan}
                if not args['dont_plot_modality_losses']:
                    results['test2'].update({'audio_loss': np.nan, 'text_loss': np.nan, 'fusion_loss': np.nan})
            else:
                loaders['test2'] = self.test2_loader
                results['test2'] = {}
            if not args['test_only']:
                loaders['dev'] = self.dev_loader
                results['dev'] = {}

            for split, loader in loaders.items():
                all_labels = []
                all_probs = []
                all_files = []
                all_datasets = []

                total_loss = 0.0
                modality_loss_sums = {'audio': 0.0, 'text': 0.0, 'fusion': 0.0}
                modality_loss_counts = {'audio': 0, 'text': 0, 'fusion': 0}
                test_losses = []
                num_batches = 0

                if args['test_only']:
                    loader = tqdm(loader, total=len(loader), desc="Testing")
                for i, (features_batch, labels_dict, file_path, dataset_name) in enumerate(loader):
                    label_batch = labels_dict['label']
                    if i == args['max_test_step'] and args['max_test_step'] > 0:
                        break
                    output, loss, loss_components = self.forward_pass(features_batch, label_batch)

                    fusion_output = output['fusion']
                    audio_output = output['audio']
                    # When text is missing, fusion output is Nan. Replace it with audio output
                    text_mask = features_batch['has_text'].to(self.device)
                    fusion_output[~text_mask] = audio_output[~text_mask]

                    probs = torch.softmax(fusion_output, dim=1)
                    
                    # Collect predictions and labels
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(label_batch.cpu().numpy())
                    all_files.extend(file_path)
                    all_datasets.extend(dataset_name)

                    total_loss += loss.item()
                    if not args['dont_plot_modality_losses'] and loss_components is not None:
                        for key in ('audio', 'text', 'fusion'):
                            comp = loss_components.get(key, None)
                            if comp is not None and comp['count'] > 0 and not np.isnan(comp['loss']):
                                modality_loss_sums[key] += comp['loss'] * comp['count']
                                modality_loss_counts[key] += comp['count']
                    test_losses.append((loss.item(), file_path))
                    num_batches += 1

                avg_loss = total_loss / num_batches
                all_probs = np.array(all_probs)
                num_batches_all += num_batches

                # Average probabilities across files
                unique_preds = []
                unique_probs = []
                unique_labels = []

                unique_files = np.unique(np.array(all_files))
                if len(unique_files) == 0 or len(all_labels) == 0:
                    print(f"Warning: No samples found for {split} split. Setting metrics to nan.")
                    results[split]['loss'] = float('nan')
                    results[split]['micro_accuracy'] = float('nan')
                for file in unique_files:
                    file_mask = np.array(all_files) == file
                    file_probs = all_probs[file_mask]
                    file_labels = np.array(all_labels)[file_mask]


                    if len(file_labels) == 0:
                        continue
                    assert len(np.unique(file_labels)) == 1, "All labels for the same file should be identical"
                    mean_probs = np.mean(file_probs, axis=0)
                    prediction = np.argmax(mean_probs)
                    # print('prediction is', prediction)
                    unique_probs.append(mean_probs[prediction])
                    unique_preds.append(prediction)
                    unique_labels.append(file_labels[0])  # All labels for the same file are identical
                
                unique_preds = np.array(unique_preds)
                unique_labels = np.array(unique_labels).squeeze()
                unique_probs = np.array(unique_probs).squeeze()

                micro_accuracy = np.mean(unique_preds == unique_labels)
                classes = np.unique(unique_labels)
                per_class_acc = []
                per_class_f1 = []
                for class_ in classes:
                    cls_mask = (unique_labels == class_)
                    if np.sum(cls_mask) > 0:
                        acc = np.mean(unique_preds[cls_mask] == unique_labels[cls_mask])
                        per_class_acc.append(acc)
                        if args['test_only']:
                            self.logging(f"Emotion {self.idx_to_emotion[class_]:<8s}: Accuracy {acc:.4f} ({np.sum(cls_mask)} samples)")
                    tp = np.sum((unique_preds == class_) & (unique_labels == class_))
                    fp = np.sum((unique_preds == class_) & (unique_labels != class_))
                    fn = np.sum((unique_preds != class_) & (unique_labels == class_))
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    per_class_f1.append(f1)
                macro_accuracy = np.mean(per_class_acc) if per_class_acc else float('nan')
                macro_f1 = np.mean(per_class_f1) if per_class_f1 else float('nan')
                results[split]['loss'] = avg_loss
                results[split]['micro_accuracy'] = micro_accuracy
                results[split]['macro_accuracy'] = macro_accuracy
                results[split]['macro_f1'] = macro_f1
                if not args['dont_plot_modality_losses']:
                    for key in ('audio', 'text', 'fusion'):
                        if modality_loss_counts[key] > 0:
                            results[split][f"{key}_loss"] = modality_loss_sums[key] / modality_loss_counts[key]
                        else:
                            results[split][f"{key}_loss"] = np.nan

                # Calculate and log metrics for each dataset within a split
                if (args['test_only'] or (self.train_step + args['test_step'] > args['max_train_step'])) and split == 'test':
                    for dataset in np.unique(all_datasets):
                        dataset_mask = np.array(all_datasets) == dataset
                        dataset_labels = np.array(all_labels)[dataset_mask]
                        dataset_probs = np.array(all_probs)[dataset_mask]

                        unique_preds = []
                        unique_labels = []
                        unique_probs = []

                        for file in np.unique(np.array(all_files)[dataset_mask]):
                            file_mask = np.array(all_files)[dataset_mask] == file
                            file_probs = dataset_probs[file_mask]
                            file_labels = dataset_labels[file_mask]

                            if len(file_labels) == 0:
                                continue
                            assert len(np.unique(file_labels)) == 1, "All labels for the same file should be identical"
                            mean_probs = np.mean(file_probs, axis=0)
                            prediction = np.argmax(mean_probs)
                            unique_probs.append(mean_probs[prediction])
                            unique_preds.append(prediction)
                            unique_labels.append(file_labels[0])

                        unique_preds = np.array(unique_preds)
                        unique_labels = np.array(unique_labels).squeeze()
                        unique_probs = np.array(unique_probs).squeeze()

                        micro_accuracy = np.mean(unique_preds == unique_labels)
                        classes = np.unique(unique_labels)
                        per_class_acc = []
                        per_class_f1 = []
                        for class_ in classes:
                            cls_mask = (unique_labels == class_)
                            if np.sum(cls_mask) > 0:
                                acc = np.mean(unique_preds[cls_mask] == unique_labels[cls_mask])
                                per_class_acc.append(acc)
                            tp = np.sum((unique_preds == class_) & (unique_labels == class_))
                            fp = np.sum((unique_preds == class_) & (unique_labels != class_))
                            fn = np.sum((unique_preds != class_) & (unique_labels == class_))
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                            per_class_f1.append(f1)
                        macro_accuracy = np.mean(per_class_acc) if per_class_acc else float('nan')
                        macro_f1 = np.mean(per_class_f1) if per_class_f1 else float('nan')

                        self.logging(
                            f"Dataset: {dataset}, Micro Accuracy: {micro_accuracy:.4f}, Macro Accuracy: {macro_accuracy:.4f}, Macro F1: {macro_f1:.4f}"
                        )

            if self.case_evaluator is not None:
                case_results = self.case_evaluator.evaluate()
                results['case'] = {}
                results['case']['runtime_rate'] = case_results['results']['runtime']['forward_rate_sec_per_sec_audio']
                results['case']['micro_accuracy_audio'] = case_results['results']['audio']['micro_accuracy']
                results['case']['macro_f1_audio'] = case_results['results']['audio']['macro_f1']

                

                if 'text' in case_results['results']:
                    results['case']['micro_accuracy_text'] = case_results['results']['text']['micro_accuracy']
                    results['case']['macro_f1_text'] = case_results['results']['text']['macro_f1']
                else:
                    results['case']['micro_accuracy_text'] = float('nan')
                    results['case']['macro_f1_text'] = float('nan')

                if 'fusion' in case_results['results']:
                    results['case']['micro_accuracy_fusion_vs_audio'] = case_results['results']['fusion']['micro_accuracy_audio']
                    results['case']['micro_accuracy_fusion_vs_text'] = case_results['results']['fusion']['micro_accuracy_text']
                    results['case']['macro_f1_fusion_vs_audio'] = case_results['results']['fusion']['macro_f1_audio']
                    results['case']['macro_f1_fusion_vs_text'] = case_results['results']['fusion']['macro_f1_text']
                else:
                    results['case']['micro_accuracy_fusion_vs_audio'] = float('nan')
                    results['case']['micro_accuracy_fusion_vs_text'] = float('nan')
                    results['case']['macro_f1_fusion_vs_audio'] = float('nan')
                    results['case']['macro_f1_fusion_vs_text'] = float('nan')

                self.logging(
                    f"\nStep {self.train_step}. CASE results: "
                    f"Audio micro {results['case']['micro_accuracy_audio']:.4f}, "
                    f"Text micro {results['case']['micro_accuracy_text']:.4f}, "
                    f"Audio macro F1 {results['case']['macro_f1_audio']:.4f}, "
                    f"Text macro F1 {results['case']['macro_f1_text']:.4f}, "
                    f"Fusion vs Audio micro {results['case']['micro_accuracy_fusion_vs_audio']:.4f}, "
                    f"Fusion vs Text micro {results['case']['micro_accuracy_fusion_vs_text']:.4f}, "
                    f"Fusion vs Audio macro F1 {results['case']['macro_f1_fusion_vs_audio']:.4f}, "
                    f"Fusion vs Text macro F1 {results['case']['macro_f1_fusion_vs_text']:.4f}"
                )

            if args['skip_test']:
                results['test'] = {'loss': np.nan, 'micro_accuracy': np.nan, 'macro_accuracy': np.nan, 'macro_f1': np.nan}
                if not args['dont_plot_modality_losses']:
                    results['test'].update({'audio_loss': np.nan, 'text_loss': np.nan, 'fusion_loss': np.nan})

            results['_available_splits'] = list(loaders.keys())
            results['num_batches'] = num_batches_all

        self.model.train()
        return results

if __name__ == "__main__":

    if args['debug']:
        print("Torch version:", torch.__version__)
        try:
            import torchaudio
            print("Torchaudio version:", torchaudio.__version__)
        except ImportError:
            print("Torchaudio not installed")
        try:
            import torchvision
            print("Torchvision version:", torchvision.__version__)
        except ImportError:
            print("Torchvision not installed")
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device:", torch.cuda.get_device_name(0))

    trainer = SERTrainer()

    if args['test_only']:
        test_results = trainer.test()
        print(
            f"Test Results - Loss: {test_results['test']['loss']:.4f}, "
            f"Micro Accuracy: {test_results['test']['micro_accuracy']:.4f}, "
            f"Macro Accuracy: {test_results['test']['macro_accuracy']:.4f}, "
            f"Macro F1: {test_results['test'].get('macro_f1', float('nan')):.4f}"
        )
    else:
        trainer.train()