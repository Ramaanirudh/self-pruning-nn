import os
import random
import numpy as np  # type: ignore
import torch  # type: ignore
import logging
from typing import Optional

try:
    from layers import PrunableLinear  # type: ignore
except ImportError:
    from src.layers import PrunableLinear  # type: ignore

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducible results across Python, 
    NumPy, and PyTorch (including CUDA deterministic behaviors).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_all_gates(model: torch.nn.Module) -> torch.Tensor:
    """
    Helper function to collect and concatenate all gate values 
    (sigmoid outputs) from all PrunableLinear layers within the model.
    """
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                # Apply sigmoid to calculate the actual gate
                gate_scores = getattr(module, 'gate_scores')
                gates = torch.sigmoid(gate_scores)
                all_gates.append(gates.cpu().view(-1))
                
    if not all_gates:
        return torch.tensor([])
        
    return torch.cat(all_gates)


def get_logger(name: str = "NN_Logger", log_file: Optional[str] = None) -> logging.Logger:
    """
    Simple logging helper function to return a configured Python logger.
    If log_file is provided, it writes to the file as well as stdout.
    """
    logger = logging.getLogger(name)
    
    # Avoid attaching multiple handlers if called repeatedly
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Stream Handler (stdout)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Optional File Handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

