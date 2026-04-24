import os
import torch  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from typing import List, Dict, Optional

# Assumes layers.py is in the same directory (src/)
from layers import PrunableLinear  # type: ignore

def compute_sparsity(model: torch.nn.Module, threshold: float = 1e-2) -> float:
    """
    Collects all gate values from PrunableLinear layers and computes the 
    percentage of gates that have a value < threshold (1e-2).
    """
    pruned_count = 0
    total_count = 0
    
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gate_scores = getattr(module, 'gate_scores')
                gates = torch.sigmoid(gate_scores)
                pruned_count += int(torch.sum(gates < threshold).item())  # type: ignore
                total_count += int(gates.numel())  # type: ignore
                
    if total_count == 0:
        return 0.0
        
    sparsity_percentage = 100.0 * pruned_count / total_count  # type: ignore
    return sparsity_percentage


def test_accuracy(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device = None) -> float:
    """
    Tests the accuracy of the model on the provided dataloader testset.
    """
    if device is None:
        # Infer device from model parameters
        device = next(model.parameters()).device

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += int(labels.size(0))  # type: ignore
            correct += int(torch.sum(predicted == labels).item())  # type: ignore
            
    if total == 0:
        return 0.0
        
    accuracy_percentage = 100.0 * correct / total
    return accuracy_percentage


def plot_gate_distribution(model: torch.nn.Module, lambda_val: Optional[float] = None, save_path: Optional[str] = None) -> None:
    """
    Collects all gate values from PrunableLinear layers, plots a full 
    distribution histogram using matplotlib, and saves the plot.
    """
    # Use default relative outputs/plots folder if no save_path provided
    if save_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plot_dir = os.path.join(project_root, 'outputs', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        filename = 'gate_distribution.png' if lambda_val is None else f'gate_distribution_lambda_{lambda_val}.png'
        save_path = os.path.join(plot_dir, filename)
    else:
        # Create user-specified folder if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gate_scores = getattr(module, 'gate_scores')
                gates = torch.sigmoid(gate_scores)
                all_gates.append(gates.cpu().view(-1))
                
    if not all_gates:
        print("No PrunableLinear layers found in the model.")
        return
        
    # Concatenate and convert nicely to numpy for matplotlib plotting
    all_gates = torch.cat(all_gates).numpy()
    
    plt.figure(figsize=(8, 6))
    plt.hist(all_gates, bins=50, range=(0, 1), alpha=0.75, color='#4c72b0', edgecolor='black')
    
    title = 'Gate Value Distribution'
    if lambda_val is not None:
        title += f' ($\\lambda$ = {lambda_val})'
        
    plt.title(title, fontsize=14)
    plt.xlabel('Gate Output Value (0.0 to 1.0)', fontsize=12)
    plt.ylabel('Frequency (Number of Weights)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Gate distribution plot successfully saved to: {save_path}")


def generate_results_table(results: List[Dict[str, float]]) -> str:
    """
    Easily generates an ascii/markdown formatted table correlating Lambda, 
    Accuracy, and Sparsity.
    
    Expects input `results` as: 
    [{'lambda': 1e-5, 'accuracy': 95.2, 'sparsity': 12.3}, ...]
    """
    # Define exact table headers
    header  = f"{'Lambda':>10} | {'Accuracy (%)':>14} | {'Sparsity (%)':>14}"
    divider = "-" * len(header)
    
    lines = [header, divider]
    for res in results:
        l_val = res.get('lambda', 0)
        acc = res.get('accuracy', 0)
        spar = res.get('sparsity', 0)
        
        # Elegant scientific formatting for lambda
        l_str = f"{l_val:.1e}" if l_val != 0 else "0.0"
            
        line = f"{l_str:>10} | {acc:>14.2f} | {spar:>14.2f}"
        lines.append(line)
        
    table_str = "\n".join(lines)
    return table_str
