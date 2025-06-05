import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from data_set import GestureDataset
from basic_tcn import TCNModel
import os
import time
from tqdm import tqdm

def compare_outputs(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5):
    """Compare PyTorch and ONNX model outputs"""
    # Convert PyTorch tensor to numpy for comparison
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.detach().cpu().numpy()
    
    # Check if outputs are close
    is_close = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    
    # Print comparison results
    if not is_close:
        print("❌ PyTorch and ONNX model outputs differ!")
        print(f"Max absolute difference: {np.max(np.abs(pytorch_output - onnx_output))}")
        try:
            print(f"Max relative difference: {np.max(np.abs((pytorch_output - onnx_output) / (pytorch_output + 1e-10)))}")
        except:
            print("Could not calculate relative difference (division issue)")
    
    return is_close

def test_all_files():
    # Define classes and correctly point to the test directory
    classes = ['swipe_up', 'swipe_down', 'thumbs_up', 'idle_gestures']
    
    # Load models
    print("Loading models...")
    try:
        # Load the PyTorch model
        model = TCNModel()
        model.load_state_dict(torch.load('best_tcn_optuna_model.pth'))
        model.eval()
        print("✅ PyTorch model loaded successfully")
        
        # Load the ONNX model
        onnx_session = ort.InferenceSession("tcn_model.onnx", providers=['CPUExecutionProvider'])
        print("✅ ONNX model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None
    
    # Verify directories exist
    test_dir = os.path.join('model', 'test')
    print(f"Looking for test data in: {os.path.abspath(test_dir)}")
    
    file_count = 0
    for cls in classes:
        cls_path = os.path.join(test_dir, cls)
        if os.path.exists(cls_path):
            files = os.listdir(cls_path)
            file_count += len(files)
            print(f"Found {len(files)} files in {cls}")
        else:
            print(f"Directory not found: {cls_path}")
    
    if file_count == 0:
        print("No test files found!")
        return None
    
    try:
        dataset = GestureDataset(test_dir, classes)
        print(f"Dataset loaded successfully with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Initialize results tracking
    results = {
        'match_count': 0,
        'total_count': 0,
        'pytorch_time': 0,
        'onnx_time': 0,
        'correct_predictions': 0,
        'pytorch_correct': 0,
        'onnx_correct': 0,
    }
    
    # Test each file
    print(f"\nTesting {len(dataset)} files...")
    for idx in tqdm(range(len(dataset))):
        sample_data, sample_label = dataset[idx]
        sample_input = sample_data.unsqueeze(0)  # Add batch dimension
        
        # PyTorch inference
        with torch.no_grad():
            start_time = time.time()
            pytorch_output = model(sample_input)
            results['pytorch_time'] += time.time() - start_time
        
        # ONNX inference
        try:
            onnx_inputs = {onnx_session.get_inputs()[0].name: sample_input.numpy()}
            start_time = time.time()
            onnx_output = onnx_session.run(None, onnx_inputs)[0]
            results['onnx_time'] += time.time() - start_time
            
            # Compare outputs
            is_match = compare_outputs(pytorch_output, onnx_output)
            if is_match:
                results['match_count'] += 1
            results['total_count'] += 1
            
            # Check if predictions are correct
            pytorch_pred = torch.argmax(pytorch_output, dim=1).item()
            onnx_pred = np.argmax(onnx_output, axis=1).item()
            
            if pytorch_pred == sample_label:
                results['pytorch_correct'] += 1
            if onnx_pred == sample_label:
                results['onnx_correct'] += 1
            if pytorch_pred == sample_label and onnx_pred == sample_label:
                results['correct_predictions'] += 1
                
        except Exception as e:
            print(f"Error during test for sample {idx}: {e}")
    
    # Print summary
    print("\n===== SUMMARY =====")
    print(f"Total files tested: {results['total_count']}")
    print(f"Output match rate: {results['match_count'] / results['total_count'] * 100:.2f}%")
    print(f"PyTorch accuracy: {results['pytorch_correct'] / results['total_count'] * 100:.2f}%")
    print(f"ONNX accuracy: {results['onnx_correct'] / results['total_count'] * 100:.2f}%")
    print(f"Both correct rate: {results['correct_predictions'] / results['total_count'] * 100:.2f}%")
    
    # Compare average inference times
    avg_pytorch_time = results['pytorch_time'] / results['total_count'] * 1000  # ms
    avg_onnx_time = results['onnx_time'] / results['total_count'] * 1000  # ms
    
    print("\n===== PERFORMANCE =====")
    print(f"Average PyTorch inference time: {avg_pytorch_time:.2f} ms")
    print(f"Average ONNX inference time: {avg_onnx_time:.2f} ms")
    print(f"Speed improvement: {avg_pytorch_time / avg_onnx_time:.2f}x")
    
    # Create visualization for a random sample
    print("\nCreating visualization for a random sample...")
    random_idx = np.random.randint(0, len(dataset))
    random_data, random_label = dataset[random_idx]
    random_input = random_data.unsqueeze(0)
    
    with torch.no_grad():
        random_pytorch_output = model(random_input)
    
    random_onnx_inputs = {onnx_session.get_inputs()[0].name: random_input.numpy()}
    random_onnx_output = onnx_session.run(None, random_onnx_inputs)[0]
    
    # Visualize predictions
    pytorch_probs = torch.nn.functional.softmax(random_pytorch_output, dim=1).numpy()
    onnx_probs = torch.nn.functional.softmax(torch.tensor(random_onnx_output), dim=1).numpy()
    
    plt.figure(figsize=(12, 5))
    
    # PyTorch predictions
    plt.subplot(1, 2, 1)
    plt.bar(classes, pytorch_probs[0])
    plt.title(f"PyTorch Prediction\nTrue label: {classes[random_label]}")
    plt.ylabel("Probability")
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    
    # ONNX predictions
    plt.subplot(1, 2, 2)
    plt.bar(classes, onnx_probs[0])
    plt.title(f"ONNX Prediction\nTrue label: {classes[random_label]}")
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    
    return results

if __name__ == "__main__":
    test_all_files()

