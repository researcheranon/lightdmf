import os
import csv
from pathlib import Path
import shutil
import functools
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
try:
    import pynvml
except ImportError:
    pynvml = None
from time import time
import torchaudio
from tqdm import tqdm
import concurrent.futures
import getpass
import librosa
import math
import sys


def run_lr_finder(distiller, train_loader, loss_fn, save_dir, steps=500):
    print(f"Running LR Finder for {steps} steps...")
    distiller.student.train()
    start_lr, end_lr = 1e-7, 1e-2
    optimizer = torch.optim.AdamW(distiller.student.parameters(), lr=start_lr)
    lr_factor = (end_lr / start_lr) ** (1 / steps)
    
    lrs, losses = [], []
    current_lr = start_lr

    for i, batch in enumerate(tqdm(train_loader, total=steps, desc="LR Sweep")):
        if i >= steps: break
        
        s_hid, t_hid = distiller.forward_batch(batch)
        loss = loss_fn(s_hid, t_hid)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        lrs.append(current_lr)
        losses.append(loss.item())
        
        current_lr *= lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

    # Print LRs and losses
    for lr, loss in zip(lrs, losses):
        print(f"LR: {lr:.2e}, Loss: {loss:.4f}")

    # Analysis
    smooth_loss = np.convolve(losses, np.ones(10)/10, mode='valid')
    valid_lrs = lrs[len(lrs)-len(smooth_loss):]
    div_lr = valid_lrs[np.argmin(smooth_loss)]

    plt.figure(figsize=(8, 5))
    plt.plot(lrs, losses, alpha=0.2, color='gray')
    plt.plot(valid_lrs, smooth_loss, color='blue')
    plt.xscale('log')
    plt.axvline(x=div_lr, color='red', linestyle='--')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "lr_find.png"))
    print("LR Finder complete. Plot saved to lr_find.png")
    
    print(f"\nDivergence: {div_lr:.2e} | Suggested Peak: {div_lr * 0.5:.2e}")


def parse_debug():
    return '--debug' in sys.argv


def run_parallel(func, my_iter, type, timer=False, mininterval=120, max_workers=4, first_n=None):

    if first_n is not None:
        my_iter = list(my_iter)[:first_n]

    assert type in ('process', 'thread', "sequential"), 'Type can be process, thread or sequential'
    if timer:
        t0 = time()

    if type == 'process':
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(func, my_iter), total=len(my_iter), mininterval=mininterval))
    elif type == 'thread':
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(func, my_iter), total=len(my_iter), mininterval=mininterval))
    elif type == "sequential":
        # Not parallel
        results = [func(item) for item in tqdm(my_iter)]

    if timer:
        print(f'{type} took {time()-t0} seconds')

    return results


def samples_to_frames(input_length, frame_length=400, hop_length=160, center=False):
    """
    Calculate number of spectrogram frames from input length.
    """
    if center:
        padded_length = input_length + 2 * (frame_length // 2)
    else:
        padded_length = input_length

    n_frames = (padded_length - frame_length) // hop_length + 1
    return max(n_frames, 0)


def frames_to_samples(n_frames, frame_length=400, hop_length=160, center=False):
    """
    Calculate minimum input length required to produce n_frames.
    """
    if n_frames <= 0:
        return 0

    total = (n_frames - 1) * hop_length + frame_length
    if center:
        total -= frame_length // 2 * 2  # remove the padding added in forward
    return total


def create_exp_dir(dir_path, debug=False, print_=True):
    # Create experiment directory
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False, print_=print_)
    else:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        print('Experiment dir : {}'.format(dir_path))

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))

class CsvWriter:
    # Save performance as a csv file
    def __init__(self, out_path, fieldnames, in_path=None, debug=False):

        self.out_path = out_path
        self.fieldnames = fieldnames
        self.debug = debug

        if not debug:
            if in_path is None:
                with open(out_path, "w") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
            else:
                try:
                    shutil.copy(in_path, out_path)
                except:
                    with open(out_path, "w") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()


    def update(self, performance_dict):
        if not self.debug:
            with open(self.out_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(performance_dict)


def memory():
    if torch.cuda.is_available() and pynvml is not None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        ret = 'Total: {:.2f} | Free: {:.2f} | Used: {:.2f}'.format(
            info.total / 1000000000,
            info.free / 1000000000,
            info.used / 1000000000,)
        pynvml.nvmlShutdown()
    elif torch.cuda.is_available():
        ret = "CUDA available (install pynvml for detailed memory stats)"
    else:
        ret = "CUDA not available"
    # print(ret)
    return ret

def logging(s, log_path, print_=True, log_=True):
    # Prints log
    if print_:
        print(s, flush=True)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


class LRFinderLogger:
    def __init__(self, init_lr=1e-7, final_lr=100.0, total_steps=0):
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        # self.explode_factor = explode_factor
        self.lrs = []
        self.losses = []
        self.best_loss = float('inf')
        self.step_count = 0

    def compute_lr(self, step):
        if self.total_steps <= 1:
            return self.final_lr
        return self.init_lr * (self.final_lr / self.init_lr) ** (step / (self.total_steps - 1))

    def log(self, lr, loss):
        self.lrs.append(lr)
        self.losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss

        if self.step_count % 10 == 0:
            print(f"Step: {self.step_count:3d}  LR: {lr:.2e}, Loss: {loss:6.4f}", flush=True)

        self.step_count += 1

        

    # def should_stop(self, loss):
    #     if not np.isfinite(loss):
    #         print("Loss is NaN or Inf, stopping LR finder.")
    #         return True
    #     if self.best_loss < float('inf') and loss > self.best_loss * self.explode_factor:
    #         print("Loss has exploded, stopping LR finder.")
    #         return True
    #     return False

    def plot(self, out_dir):
        if len(self.lrs) == 0:
            return
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(self.lrs, self.losses, color='blue')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('LR Finder')
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "lr_find.png")
        plt.close()

def read_csv(input_file_path, delimiter=",", numeric=False):
    with open(input_file_path, "r") as f_in:
        reader = csv.DictReader(f_in, delimiter=delimiter)
        if numeric:
            def _to_float(value):
                try:
                    if value is None:
                        return float('nan')
                    return float(value)
                except (ValueError, TypeError):
                    return float('nan')
            data = [{key: _to_float(value) for key, value in row.items()} for row in reader]
        else:
            data = [{key: value for key, value in row.items()} for row in reader]
    return data

def plot_performance(csv_path, start_step=1, title=None, save=True, n_labels=None, plot_lr_changes=False, plot_modality_losses=False):

    keys = [
        "train_loss", 
        "test_loss", 
        "dev_loss",
        f"dev_micro_accuracy",
        f"test_micro_accuracy",
        f"dev_macro_accuracy",
        f"test_macro_accuracy",
        f"dev_macro_f1",
        f"test_macro_f1",
        "case_audio_macro_accuracy",
        "case_text_macro_accuracy",
    ]
    if plot_modality_losses:
        keys += [
            "train_audio_loss", "train_text_loss", "train_fusion_loss",
            "dev_audio_loss", "dev_text_loss", "dev_fusion_loss",
            "test_audio_loss", "test_text_loss", "test_fusion_loss",
            "test2_audio_loss", "test2_text_loss", "test2_fusion_loss",
        ]
    data = read_csv(csv_path, numeric=True)
    if len(data) == 0:
        return
    available_keys = set(data[0].keys())
    keys = [k for k in keys if k in available_keys]

    # only plot test2 metrics if they are not nan
    present_test2_keys = [k for k in available_keys if 'test2' in k]
    if present_test2_keys:
        has_any_test2 = False
        for row in data:
            for k in present_test2_keys:
                val = row.get(k, float('nan'))
                if not np.isnan(val):
                    has_any_test2 = True
                    break
            if has_any_test2:
                break
        if not has_any_test2:
            keys = [k for k in keys if 'test2' not in k]

    x_lr_changes = []
    vals = {key: {"x":[], "y":[]} for key in keys}
    old_lr = data[0]["lr"]
    for item in data:
        step = item["step"]
        if step >= start_step:
            new_lr = item["lr"]
            if new_lr < old_lr:
                x_lr_changes.append(step)
                old_lr = new_lr

            for key in keys:
                val = item[key]
                if not np.isnan(val):
                    vals[key]["x"].append(step)
                    vals[key]["y"].append(val)
    
    for key, points in vals.items():
        plt.plot(points["x"], points["y"], label=key)

    if plot_lr_changes:
        label = f"LR changes (x{len(x_lr_changes)})"
        for x in x_lr_changes:
            plt.axvline(x=x, color="black", linestyle="--", linewidth=1, label=label)
            label = None
    
    plt.legend()
    plt.grid()
    plt.xlabel("Steps")
    # plt.ylabel("Loss")
    if title == None:
        title = csv_path.split("/")[-2]
    # Add random baseline lines if n_labels is provided
    if n_labels is not None:
        random_loss = np.log(n_labels)
        random_accuracy = 1.0 / n_labels
        title += f"\nRandom Baseline: Loss={random_loss:.2f}, Acc={random_accuracy:.2f}"

    plt.title(title)
    png_path = csv_path.replace(".csv", ".pdf")
    if save:
        plt.savefig(png_path)
    else:
        plt.show()
    plt.close()

def get_gpu_name():
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Index of the GPU
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        capacity = info.total // (1024 ** 3)
        pynvml.nvmlShutdown()
        return gpu_name + ' ' + str(capacity) + ' GB'
    else:
        return "CPU"


def get_mfcc(audio, sr=16000, n_mfcc=40, n_mels=23, hop_length_ms=10, win_length_ms=25, delta1=False, delta2=False, **kwargs):
    """
    Extract MFCC features from audio.

    Returns:
        np.ndarray of shape (n_mfcc, time)
    """
    hop_length = int(sr * hop_length_ms / 1000)
    win_length = int(sr * win_length_ms / 1000)
    n_fft = win_length

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels
    )
    delta_mfcc = librosa.feature.delta(mfcc)          # first derivative
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)  # second derivative

    # Stack them: (40 + 40 + 40) = 120 features
    return {'mfcc': mfcc,
            'delta1': delta_mfcc,
            'delta2': delta2_mfcc}


def seconds_to_mfcc_frames(n_seconds, sr=16000, hop_length_ms=10, win_length_ms=25):
    hop_length = int(sr * hop_length_ms / 1000)
    win_length = int(sr * win_length_ms / 1000)
    n_fft = win_length

    n_samples = n_seconds * sr
    n_time = math.floor((n_samples - n_fft) / hop_length) + 1
    return n_time


def get_melspectrogram(audio, sr=16000, n_mels=128, hop_length_ms=10, win_length_ms=25, **kwargs):
    """
    Extract Mel-scaled spectrogram from audio, converted to dB.
    
    Returns:
        np.ndarray of shape (n_mels, time)
    """
    hop_length = int(sr * hop_length_ms / 1000)
    win_length = int(sr * win_length_ms / 1000)
    n_fft = win_length

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length
    )
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram


def get_chroma(audio, sr=16000, n_chroma=12, hop_length_ms=10, win_length_ms=25):
    """
    Extract chroma features from audio.

    Returns:
        np.ndarray of shape (n_chroma, time)
    """
    hop_length = int(sr * hop_length_ms / 1000)
    win_length = int(sr * win_length_ms / 1000)
    n_fft = win_length

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_chroma=n_chroma,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length
    )
    return chroma


if __name__ == "__main__":
    plt.ion()
    csv_path = '../output/20260212-155722-850/performance.csv'
    plot_performance(csv_path, start_step=1, title=None, save=False, n_labels=None, plot_lr_changes=False, plot_modality_losses=False)

