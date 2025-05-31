import time
import torch
import pandas as pd
from lstm_gru_models import LSTMModel, GRUModel
from basic_tcn import TCNModel

def benchmark_inference_speed(model, model_name, device, num_runs=100):
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 30, 63).to(device)  # (batch_size, seq_len, input_size)

    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)

    # Measure
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    elapsed = time.time() - start
    return elapsed / num_runs  # Average time per inference

inference_results = []
results = [{'name': 'LSTM', 'save_path': 'best_lstm_gesture_model.pth' },
           {'name': 'GRU', 'save_path': 'best_gru_gesture_model.pth'},
           {'name': 'TCN', 'save_path': 'best_tcn_optuna_model.pth'},]
for result in results:
    model_class = eval(result['name'].upper() + 'Model')  # Get class by name
    model = model_class()
    model.load_state_dict(torch.load(result['save_path']))
    
    gpu_time = benchmark_inference_speed(model, result['name'], device='cuda' if torch.cuda.is_available() else 'cpu')
    cpu_time = benchmark_inference_speed(model, result['name'], device='cpu')
    
    inference_results.append({
        'model': result['name'],
        'gpu_time': gpu_time,
        'cpu_time': cpu_time
    })


# Collect metrics into a DataFrame
comparison_data = []
for res in results:
    for inf_res in inference_results:
        if inf_res['model'] == res['name']:
            comparison_data.append({
                'Model': res['name'],
                # 'Best Val Acc (%)': round(res['best_val_acc'] * 100, 2),
                # 'Best Val Loss': round(res['best_val_loss'], 4),
                # 'Avg Epoch Time (s)': round(res['avg_epoch_time'], 2),
                'Inference GPU (ms)': round(inf_res['gpu_time'] * 1000, 2),
                'Inference CPU (ms)': round(inf_res['cpu_time'] * 1000, 2)
            })

# Create DataFrame
df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))