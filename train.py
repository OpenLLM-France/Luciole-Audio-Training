import gc      
import os
import json
import numpy as np

import torch
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader

from utils.checkpoint_utils import save_model_checkpoint_peft, save_optimizer_scheduler_scaler
from utils.metrics_utils import save_metrics_to_json

from tqdm import tqdm
from contextlib import nullcontext

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_mean(values):
    return np.mean([v for v in values if v is not None]) if values else None

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

def train(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, scaler, train_config, device=None):
    """
    Trains the model with memory optimizations and supports resuming from a specific epoch and step.
    """
    use_fp16 = train_config.use_fp16
    autocast = torch.amp.autocast if use_fp16 else nullcontext

    train_metrics = {"loss": [], "acc": [], "ppl": []}
    val_metrics = {"loss": [], "acc": [], "ppl": []}

    best_val_loss = float("inf")
    best_val_acc = 0.0

    model.to(device)
    
    # If using fp16, ensure model parameters are in float16 (but optimizer works with float32)
    if use_fp16:
        model.to(dtype=torch.float16)
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        
    early_stopping = EarlyStopper(train_config.patience) if train_config.patience else None

    # Load the last epoch and step from the training log
    start_epoch, start_step = load_last_epoch_and_step(os.path.join(train_config.output_dir, "train_log.json"))
    logger.info(f"Resuming training from epoch {start_epoch}, step {start_step}")

    # Overall step counter across epochs
    step_count = start_step  
    # Total number of batches for progress reporting (might be approximate when resuming)
    total_batches = (train_config.num_epochs - start_epoch + 1) * len(train_dataloader)
    initial_progress = (start_epoch - 1) * len(train_dataloader) + start_step

    with tqdm(total=total_batches, initial=initial_progress, desc="Training", dynamic_ncols=True, colour='blue') as pbar:
        for epoch in range(start_epoch, train_config.num_epochs + 1):
            model.train()
            epoch_loss, epoch_acc = 0.0, 0.0
            optimizer.zero_grad(set_to_none=True)
            total_steps = 0
            
            # For the first resumed epoch, skip batches that were already processed.
            for step, batch in enumerate(train_dataloader):
                # Skip already processed steps in the resumed epoch.
                if epoch == start_epoch and step < start_step:
                    continue

                step_count += 1
                total_steps += 1

                # Move batch to device
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                with autocast(device_type=device.type, enabled=use_fp16):
                    outputs, *rest = model(**batch)
                    # Scale loss if using gradient accumulation.
                    loss = outputs.loss / train_config.gradient_accumulation_steps
                    loss_item = loss.detach().item()

                # Optionally, calculate accuracy (assuming rest[0] holds accuracy info)
                acc = rest[0].cpu().item() / train_config.gradient_accumulation_steps if rest else None

                epoch_loss += loss_item
                if acc is not None:
                    epoch_acc += acc

                # Backward pass (using AMP if enabled)
                if use_fp16 and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update parameters when gradient accumulation steps are met.
                if (step + 1) % train_config.gradient_accumulation_steps == 0 or (step == len(train_dataloader) - 1):
                    if use_fp16 and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    optimizer.zero_grad(set_to_none=True)
               
                # Update progress bar and metrics per batch
                current_loss = epoch_loss / total_steps
                current_acc = epoch_acc / total_steps if total_steps > 0 else 0
               
                pbar.set_description(
                    f"Epoch {epoch}/{train_config.num_epochs} | "
                    f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f}" +
                    (f" | LR: {optimizer.param_groups[0]['lr']:.2e}" if optimizer else "")
                )
                pbar.set_postfix({'mem': f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB"})
                pbar.update(1)

                # Evaluation logic at intervals (or at end of epoch)
                if step_count % train_config.validation_step == 0 or step == len(train_dataloader) - 1:
                    current_progress = pbar.n
                    model.eval()
                    with torch.no_grad():
                        eval_ppl, eval_loss, eval_acc = evaluation(model, device, train_config, eval_dataloader)

                    if eval_loss < best_val_loss:
                        best_val_loss = eval_loss
                        best_val_acc = max(best_val_acc, eval_acc)

                        if train_config.save_model:
                            checkpoint_name = f"{train_config.model_name}_step_{step_count}"
                            save_model_checkpoint_peft(
                                model, train_config, epoch, step_count, checkpoint_name=checkpoint_name, save_trainable_only=True
                            )
                            save_optimizer_scheduler_scaler(
                                optimizer, lr_scheduler, scaler, train_config.output_dir
                            )
                            logger.info(f"Model saved: {checkpoint_name}")

                    val_metrics["loss"].append(eval_loss)
                    val_metrics["acc"].append(eval_acc)
                    val_metrics["ppl"].append(eval_ppl)

                    model.train()
                    pbar.n = current_progress
                    pbar.last_print_n = current_progress
                    pbar.refresh()

                    steps_metrics = {
                        "epoch": epoch,
                        "step": step_count,
                        "train_loss": current_loss,
                        "train_acc": current_acc,
                        "train_ppl": torch.exp(torch.tensor(current_loss)).item(),
                        "val_loss": eval_loss,
                        "val_acc": eval_acc,
                        "val_ppl": eval_ppl if eval_ppl else None,
                        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
                    }
                    save_metrics_to_json(steps_metrics, filepath=os.path.join(train_config.output_dir, "train_log.json"))
                    # Reset evaluation trigger if needed

            # Compute and record epoch metrics
            train_epoch_loss = epoch_loss / total_steps if total_steps else float('inf')
            train_epoch_acc = epoch_acc / total_steps if total_steps and acc is not None else None
            train_epoch_ppl = torch.exp(torch.tensor(train_epoch_loss)).item() if total_steps else None

            train_metrics["loss"].append(train_epoch_loss)
            train_metrics["acc"].append(train_epoch_acc)
            train_metrics["ppl"].append(train_epoch_ppl)

            # Evaluate at the end of the epoch if evaluation loader is provided
            if train_config.run_validation and eval_dataloader:
                eval_ppl, eval_loss, eval_acc = evaluation(model, device, train_config, eval_dataloader)
                    
                if eval_loss < best_val_loss:
                    best_val_loss = eval_loss
                    best_val_acc = max(best_val_acc, eval_acc)

                    if train_config.save_model:
                        checkpoint_name = f"{train_config.model_name}_step_{step_count}"
                        save_model_checkpoint_peft(
                            model, train_config, epoch, step_count, checkpoint_name=checkpoint_name, save_trainable_only=True
                        )
                        save_optimizer_scheduler_scaler(
                            optimizer, lr_scheduler, scaler, train_config.output_dir
                        )
                        logger.info(f"Model saved: {checkpoint_name}")

                val_metrics["loss"].append(eval_loss)
                val_metrics["acc"].append(eval_acc)
                val_metrics["ppl"].append(eval_ppl)
            
            # Check for early stopping if enabled
            if early_stopping and early_stopping(eval_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    results = {
        "train_loss": safe_mean(train_metrics["loss"]),
        "train_acc": safe_mean(train_metrics["acc"]),
        "train_ppl": safe_mean(train_metrics["ppl"]),
        "val_loss": safe_mean(val_metrics["loss"]),
        "val_acc": safe_mean(val_metrics["acc"]),
        "val_ppl": safe_mean(val_metrics["ppl"]),
    }

    return results


def evaluation(model, device, train_config, eval_dataloader):
    """
    Memory-optimized evaluation
    """
    model.eval()
    eval_loss, eval_acc = 0.0 , 0.0
    autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=train_config.use_fp16)   

    with torch.no_grad(), tqdm(eval_dataloader, desc="Evaluating", colour='green') as pbar:
        for step, batch in enumerate(pbar):
            # FIXED: Batch cleanup
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            with autocast:
                outputs, *rest = model(**batch)
            
            # FIXED: Immediate tensor conversion
            loss = outputs.loss.detach().cpu().item()
            acc = rest[0].detach().cpu().item() if rest else -1

            eval_loss += loss
            eval_acc += acc

            # FIXED: Clear intermediates
            del batch, outputs, rest

            pbar.set_postfix({
                'eval_loss': f"{eval_loss/(step + 1):.4f}",
                'eval_acc': f"{eval_acc/(step + 1):.4f}" if acc else "N/A",
                'eval_ppl': f"{torch.exp(torch.tensor(eval_loss/(step + 1))):.4f}" if acc else "N/A",
                
            })
            
    avg_loss = eval_loss / len(eval_dataloader)
    avg_acc = eval_acc / len(eval_dataloader)
    eval_ppl = torch.exp(torch.tensor(avg_loss))

    # logger.info(f"Evaluation - Perplexity: {eval_ppl:.4f}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    return eval_ppl.item(), avg_loss, avg_acc

