import gc      
import os
import json
import numpy as np

import torch
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader

from utils.checkpoint_utils import save_model_checkpoint_peft
from utils.metrics_utils import save_metrics_to_json

from tqdm import tqdm
from contextlib import nullcontext
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

def train(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, scaler, train_config, start_epoch=1, start_step=0, device=None):

    writer = SummaryWriter(log_dir=os.path.join(train_config.output_dir, "runs"))
    use_fp16 = train_config.use_fp16
    autocast_context = autocast(device_type=device.type, enabled=use_fp16) if use_fp16 else nullcontext()

    model.to(device)
    model.train()

    train_metrics = {"loss": [], "acc": [], "ppl": []}
    val_metrics = {"loss": [], "acc": [], "ppl": []}

    best_val_loss = float("inf")
    best_val_acc = 0.0

    if not any(p.requires_grad for p in model.encoder_projector.parameters()):
        logger.error("No projector parameters require gradients!")
        raise RuntimeError("Projector parameters are frozen")

    early_stopping = EarlyStopper(train_config.patience) if train_config.patience else None

    step_count = start_step
    total_batches = (train_config.num_epochs - start_epoch + 1) * len(train_dataloader)

    with tqdm(total=total_batches, initial=(start_epoch - 1) * len(train_dataloader) + start_step, desc="Training", dynamic_ncols=True, colour='blue') as pbar:
        for epoch in range(start_epoch, train_config.num_epochs + 1):
            if epoch != start_epoch:
                start_step = 0

            model.train()
            epoch_loss, epoch_acc, steps_this_epoch = 0.0, 0.0, 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_dataloader):
                if epoch == start_epoch and step < start_step:
                    continue

                step_count += 1
                steps_this_epoch += 1

                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                try:
                    with autocast_context:
                        outputs, *rest = model(**batch)
                    loss = outputs.loss / train_config.gradient_accumulation_steps
                    acc = rest[0] / train_config.gradient_accumulation_steps if rest else 0.0

                    epoch_loss += loss.item()
                    epoch_acc += acc

                except Exception as e:
                    logger.error(f"Error in forward pass at step {step_count}: {str(e)}")
                    raise

                try:
                    if use_fp16 and scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (step + 1) % train_config.gradient_accumulation_steps == 0 or (step == len(train_dataloader) - 1):
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
                    logger.error(f"Error in backward pass at step {step_count}: {str(e)}")
                    raise

                del loss, outputs, rest
                torch.cuda.empty_cache()


                avg_loss = epoch_loss / steps_this_epoch
                avg_acc = epoch_acc / steps_this_epoch if steps_this_epoch else 0.0
                
                # save on TensorBoard
                writer.add_scalar("Train/Loss", avg_loss, step_count)
                writer.add_scalar("Train/Accuracy", avg_acc, step_count)
                
                pbar.set_description(f"Epoch {epoch}/{train_config.num_epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
                pbar.set_postfix({'mem': f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB"})
                pbar.update(1)

                # Run validation
                if step_count % train_config.validation_step == 0 or step == len(train_dataloader) - 1:
                    model.eval()
                    with torch.no_grad():
                        eval_ppl, eval_loss, eval_acc = evaluation(model, device, train_config, eval_dataloader)
                        
                    writer.add_scalar("Val/Loss", eval_loss, step_count)
                    writer.add_scalar("Val/Accuracy", eval_acc, step_count)
                    writer.add_scalar("Val/Perplexity", eval_ppl, step_count)
                        
                    if eval_loss < best_val_loss:
                        best_val_loss = eval_loss
                        best_val_acc = max(best_val_acc, eval_acc)
                        if train_config.save_model:
                            save_model_checkpoint_peft(model, optimizer, lr_scheduler, scaler, train_config, epoch, step_count)
                            logger.info(f"Model checkpoint saved at step {step_count}")
                        torch.cuda.reset_peak_memory_stats()


                    val_metrics["loss"].append(eval_loss)
                    val_metrics["acc"].append(eval_acc)
                    val_metrics["ppl"].append(eval_ppl)

                    save_metrics_to_json({
                        "epoch": epoch,
                        "step": step_count,
                        "train_loss": avg_loss if steps_this_epoch else None,
                        "train_acc": avg_acc if steps_this_epoch else None,
                        "train_ppl": avg_loss if steps_this_epoch else None,
                        "val_loss": eval_loss,
                        "val_acc": eval_acc,
                        "val_ppl": eval_ppl,
                        "best_val_loss": best_val_loss
                    }, filepath=os.path.join(train_config.output_dir, "train_log.json"))

                    if early_stopping and early_stopping(eval_loss):
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        return summarize_results(train_metrics, val_metrics)
            del batch
            torch.cuda.empty_cache()
            # End of epoch
            if steps_this_epoch > 0:
                train_metrics["loss"].append(avg_loss)
                train_metrics["acc"].append(avg_acc)
                train_metrics["ppl"].append(avg_loss)
            else:
                logger.warning(f"Epoch {epoch} had no training steps, skipping metric logging.")
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
    Memory-optimized evaluation
    """
    model.eval()
    use_fp16 = train_config.use_fp16
    autocast_context = autocast(device_type=device.type, enabled=use_fp16) if use_fp16 else nullcontext()

    eval_loss, eval_acc = 0.0 , 0.0
    acc_steps = 0

    with tqdm(eval_dataloader, desc="Evaluating", colour='green') as pbar:
        for step, batch in enumerate(pbar):
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

            pbar.set_postfix({
                'eval_loss': f"{eval_loss/(step + 1):.4f}",
                'eval_acc': f"{(eval_acc/acc_steps):.4f}" if acc is not None else "N/A",
                'eval_ppl': f"{torch.exp(torch.tensor(eval_loss/(step + 1))):.4f}",
            })

    avg_loss = eval_loss / len(eval_dataloader)
    avg_acc = eval_acc / acc_steps if acc_steps > 0 else None
    eval_ppl = torch.exp(torch.tensor(avg_loss))

    return eval_ppl.item(), avg_loss, avg_acc



