import logging
import os
import numpy as np


class SyncFileHandler(logging.FileHandler):
    """Logging handleer that forces sync'ing every message to the disk. Slow but sure."""

    def emit(self, record):
        super().emit(record)
        if self.stream is not None:
            self.flush()
            os.fsync(self.stream.fileno())


def setup_logging(log_filename, do_sync=False):
    """Sets up two loggers: the console logger (info and higher log levels) and file logger (for debug)"""
    # Root logger (global)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # master level

    # --- Console handler ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    # --- File handler ---
    if not do_sync:
        fh = logging.FileHandler(log_filename, mode='w')
    else:
        fh = SyncFileHandler(log_filename, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Clear any default handlers to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def log_nans_nparray_error(str_prefix, np_arr, max_percent_nan):
    total_nans = np.isnan(np_arr).sum()
    nan_perc = (total_nans / np_arr.size())*100
    if nan_perc > max_percent_nan:
        logging.warning(f'{str_prefix} {total_nans} out of {np_arr.size} ({nan_perc:.2f}%) entries are NaN')

def log_nans_tensor_error(str_prefix, tensor, max_percent_nan):
    total_nans = np.isnan(tensor).sum().item()
    nan_perc = (total_nans / tensor.numel())*100
    if nan_perc > max_percent_nan:
        logging.warning(f'{str_prefix} {total_nans} out of {tensor.numel()} ({nan_perc:.2f}%) entries are NaN')
