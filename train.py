import gc      
import os
import json
import numpy as np
import math
import torch
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader

from utils.checkpoint_utils import save_model_checkpoint_peft
from utils.metrics_utils import save_metrics_to_json

from tqdm import tqdm
from contextlib import nullcontext
from collections import defaultdict

from torch.amp import autocast
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_mean(values):
    cleaned = []
    for v in values:
        if v is None:
            continue
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        cleaned.append(v)
    return np.mean(cleaned) if cleaned else None

class EarlyStopper:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def load_last_epoch_and_step(log_file):
    if not os.path.exists(log_file):
        return 1, 0
    with open(log_file, 'r') as f:
        logs = json.load(f)
    if not logs:
        return 1, 0
    last_log = logs[-1]
    return last_log['epoch'], last_log['step']

def is_iterable_dataset(dataloader):
    """Check if the dataloader uses an IterableDataset"""
    try:
        len(dataloader)
        return False
    except TypeError:
        return True

def move_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def run_validation(model, eval_dataloader, train_config, step_count, writer, device):
    model.eval()
    with torch.no_grad():
        eval_ppl, eval_loss, eval_acc = evaluation(model, device, train_config, eval_dataloader)
    writer.add_scalar("Val/Loss", eval_loss, step_count)
    writer.add_scalar("Val/Accuracy", eval_acc, step_count)
    writer.add_scalar("Val/Perplexity", eval_ppl, step_count)
    return eval_loss, eval_acc, eval_ppl

def save_checkpoint(model, tokenizer, optimizer, lr_scheduler, scaler, train_config, epoch, step_count, loss, acc, val_metrics=None):
    if train_config.save_model:
        save_model_checkpoint_peft(model, tokenizer, optimizer, lr_scheduler, scaler, train_config, epoch, step_count)
        logger.info(f"Model checkpoint saved at step {step_count}")
    log_dict = {
        "epoch": epoch,
        "step": step_count,
        "train_loss": loss,
        "train_acc": acc,
        "train_ppl": math.exp(loss),
    }
    if val_metrics:
        log_dict.update({
            "val_loss": val_metrics[0],
            "val_acc": val_metrics[1],
            "val_ppl": val_metrics[2],
            "best_val_loss": val_metrics[0],
        })
    else:
        log_dict.update({"val_loss": None, "val_acc": None, "val_ppl": None, "best_val_loss": None})

    save_metrics_to_json(log_dict, filepath=os.path.join(train_config.output_dir, "train_log.json"))


def train(model, tokenizer, train_dataloader, eval_dataloader, optimizer, lr_scheduler, scaler, train_config, start_epoch=1, start_step=0, device=None):
    writer = SummaryWriter(log_dir=os.path.join(train_config.output_dir, "runs"))
    use_fp16 = train_config.use_fp16
    autocast_context = autocast(device_type=device.type, enabled=use_fp16) if use_fp16 else nullcontext()

    model.to(device)
    model.train()
    
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_training_loss = float("inf")
    best_training_acc = 0.0

    if not any(p.requires_grad for p in model.encoder_projector.parameters()):
        logger.error("No projector parameters require gradients!")
        raise RuntimeError("Projector parameters are frozen")

    early_stopping = EarlyStopper(train_config.patience) if train_config.patience else None
    step_count = start_step
    is_iterable = is_iterable_dataset(train_dataloader)

    if is_iterable:
        logger.info("Using IterableDataset - no total in progress bar")
        pbar = tqdm(desc="Training", dynamic_ncols=True, colour='blue')
    else:
        total_batches = (train_config.num_epochs - start_epoch + 1) * len(train_dataloader)
        pbar = tqdm(total=total_batches, initial=(start_epoch - 1) * len(train_dataloader) + start_step,
                    desc="Training", dynamic_ncols=True, colour='blue')

    try:
        for epoch in range(start_epoch, train_config.num_epochs + 1):
            if epoch != start_epoch:
                start_step = 0

            model.train()
            epoch_loss, epoch_acc, steps_this_epoch = 0.0, 0.0, 0
            optimizer.zero_grad()
            batch_iterator = enumerate(train_dataloader)

            for step, batch in batch_iterator:
                if epoch == start_epoch and step < start_step:
                    continue

                step_count += 1
                steps_this_epoch += 1
                batch = move_to_device(batch, device)

                try:
                    with autocast_context:
                        outputs, *rest = model(**batch)
                        loss = outputs.loss / train_config.gradient_accumulation_steps
                        acc = rest[0] / train_config.gradient_accumulation_steps if rest else 0.0

                    epoch_loss += loss.item()
                    epoch_acc += acc

                    if use_fp16 and scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (step + 1) % train_config.gradient_accumulation_steps == 0:
                        params = model.encoder_projector.parameters() if train_config.train_projector_only else model.parameters()
                        if use_fp16 and scaler:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(params, train_config.gradient_clip_val)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(params, train_config.gradient_clip_val)
                            optimizer.step()
                        optimizer.zero_grad()
                        if lr_scheduler:
                            lr_scheduler.step()

                except Exception as e:
                    logger.error(f"Training error at step {step_count}: {e}")
                    raise

                avg_loss = epoch_loss / steps_this_epoch
                avg_acc = epoch_acc / steps_this_epoch
                writer.add_scalar("Train/Loss", avg_loss, step_count)
                writer.add_scalar("Train/Accuracy", avg_acc, step_count)

                pbar.set_description(f"Epoch {epoch}/{train_config.num_epochs} | Step {step_count} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
                pbar.set_postfix({'mem': f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB"})
                pbar.update(1)

                # Validation or checkpointing
                if step_count % train_config.runing_steps == 0:
                    if train_config.run_validation and eval_dataloader:
                        eval_loss, eval_acc, eval_ppl = run_validation(model, eval_dataloader, train_config, step_count, writer, device)
                        if eval_loss < best_val_loss:
                            best_val_loss = eval_loss
                            best_val_acc = max(best_val_acc, eval_acc)
                            save_checkpoint(model, tokenizer, optimizer, lr_scheduler, scaler, train_config, epoch, step_count, avg_loss, avg_acc, (eval_loss, eval_acc, eval_ppl))
                        val_metrics["loss"].append(eval_loss)
                        val_metrics["acc"].append(eval_acc)
                        val_metrics["ppl"].append(eval_ppl)
                        
                        if early_stopping and early_stopping(eval_loss):
                            logger.info(f"Early stopping triggered at epoch {epoch}")
                            return summarize_results(train_metrics, val_metrics)
                    else:
                        save_checkpoint(model, tokenizer, optimizer, lr_scheduler, scaler, train_config, epoch, step_count, avg_loss, avg_acc)

                if hasattr(train_config, 'max_steps_per_epoch') and steps_this_epoch >= train_config.max_steps_per_epoch:
                    logger.info(f"Reached max steps per epoch ({train_config.max_steps_per_epoch}) at epoch {epoch}")
                    break

            if steps_this_epoch > 0:
                train_metrics["loss"].append(avg_loss)
                train_metrics["acc"].append(avg_acc)
                train_metrics["ppl"].append(math.exp(avg_loss))
                writer.add_scalar("Train/EpochLoss", avg_loss, epoch)
                writer.add_scalar("Train/EpochAccuracy", avg_acc, epoch)
                save_checkpoint(model, tokenizer, optimizer, lr_scheduler, scaler, train_config, epoch, step_count, avg_loss, avg_acc)
            else:
                logger.warning(f"No steps run in epoch {epoch}, skipping logging.")

    finally:
        pbar.close()
        writer.close()

    return summarize_results(train_metrics, val_metrics)


def summarize_results(train_metrics, val_metrics):
    return {
        "train_loss": safe_mean(train_metrics["loss"]),
        "train_acc": safe_mean(train_metrics["acc"]),
        "train_ppl": safe_mean(train_metrics["ppl"]),
        "val_loss": safe_mean(val_metrics["loss"]),
        "val_acc": safe_mean(val_metrics["acc"]),
        "val_ppl": safe_mean(val_metrics["ppl"]),
    }

@torch.no_grad()
def evaluation(model, device, train_config, eval_dataloader):
    """
    Memory-optimized evaluation - handles both regular and iterable datasets
    """
    model.eval()
    use_fp16 = train_config.use_fp16
    autocast_context = autocast(device_type=device.type, enabled=use_fp16) if use_fp16 else nullcontext()

    eval_loss, eval_acc = 0.0 , 0.0
    acc_steps = 0
    total_steps = 0

    # Check if evaluation dataset is iterable
    is_eval_iterable = is_iterable_dataset(eval_dataloader)
    
    if is_eval_iterable:
        pbar = tqdm(desc="Evaluating", colour='green')
    else:
        pbar = tqdm(eval_dataloader, desc="Evaluating", colour='green')

    try:
        for step, batch in enumerate(eval_dataloader):
            total_steps += 1
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            with autocast_context:
                outputs, *rest = model(**batch)

            loss = outputs.loss.detach().cpu().item()
            eval_loss += loss

            if rest:
                acc = rest[0].detach().cpu().item()
                eval_acc += acc
                acc_steps += 1
            else:
                acc = None

            del batch, outputs, rest
            torch.cuda.empty_cache()

            if is_eval_iterable:
                pbar.set_description(f"Evaluating - Step {total_steps}")
                pbar.update(1)
            
            pbar.set_postfix({
                'eval_loss': f"{eval_loss/total_steps:.4f}",
                'eval_acc': f"{(eval_acc/acc_steps):.4f}" if acc is not None else "N/A",
                'eval_ppl': f"{torch.exp(torch.tensor(eval_loss/total_steps)):.4f}",
            })

            # Optional: limit evaluation steps for very large iterable datasets
            if hasattr(train_config, 'max_eval_steps') and total_steps >= train_config.max_eval_steps:
                logger.info(f"Reached max evaluation steps ({train_config.max_eval_steps})")
                break

    finally:
        pbar.close()

    avg_loss = eval_loss / total_steps if total_steps > 0 else float('inf')
    avg_acc = eval_acc / acc_steps if acc_steps > 0 else None
    eval_ppl = torch.exp(torch.tensor(avg_loss))

    return eval_ppl.item(), avg_loss, avg_acc