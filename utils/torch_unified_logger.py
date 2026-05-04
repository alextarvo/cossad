import torch
import wandb
import datetime

from omegaconf import OmegaConf
from collections.abc import Mapping

def to_wandb_config(cfg):
    """Converts the given Hydra config to the WanDB plain format.
    TODO(alexta): this supposed to remove duplicates in WanDB; it doesn't seem to help much.
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    flat = {}
    def rec(prefix, obj):
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                rec(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, (list, tuple)):
            flat[prefix] = list(obj)
        elif isinstance(obj, (int, float, bool, str)) or obj is None:
            flat[prefix] = obj
        else:
            flat[prefix] = str(obj)
    rec("", raw)
    return flat

class UnifiedLogger:
    """Unified logging for both TensorBoard and Weights & Biases"""

    def __init__(self, tensorboard_writer, use_wandb=True):
        self.tb_writer = tensorboard_writer
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.define_metric("GradientsNorm/*", hidden=True)
            wandb.define_metric("GradientsHist/*", hidden=True)
            wandb.define_metric("MiningYield/*", hidden=True)

    @staticmethod
    def get_timestamp():
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return timestamp

    def log_scalar(self, tag, value, step):
        """Log scalar value to both TensorBoard and W&B"""
        self.tb_writer.add_scalar(tag, value, global_step=step)
        if self.use_wandb:
            wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag, tag_scalar_dict, step, log_together=True, log_separate=False):
        """Log multiple scalars to both TensorBoard and W&B.
        Args:
            log_together: if true, all the scalars will be logged together, into the same figure
            log_together: if true, all the scalars will be logged into the separate figures
        """

        def get_tag(main_tag, sec_tag):
            return f"{main_tag}/{sec_tag}"

        if log_together:
            self.tb_writer.add_scalars(main_tag, tag_scalar_dict, global_step=step)
            if self.use_wandb:
                wandb_dict = {get_tag(main_tag, tag): value for tag, value in tag_scalar_dict.items()}
                wandb.log(wandb_dict, step=step)
        if log_separate:
            for tag, value in tag_scalar_dict.items():
                self.log_scalar(get_tag(main_tag, tag), value, step=step)

    def log_histogram(self, tag, values, step):
        """Log histogram to both TensorBoard and W&B"""
        self.tb_writer.add_histogram(tag, values, global_step=step)
        if self.use_wandb:
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            wandb.log({tag: wandb.Histogram(values)}, step=step)

    def log_text(self, tag, text_string, step):
        """Log text to TensorBoard (W&B doesn't have direct equivalent)"""
        self.tb_writer.add_text(tag, text_string, global_step=step)
