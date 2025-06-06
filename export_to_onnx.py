from model.basic_tcn import TCNModel
import torch

model = TCNModel()
model.load_state_dict(torch.load('best_tcn_optuna_model.pth'))
model.eval()

dummy_input = torch.randn(1, 30, 63)


traced_model = torch.jit.trace(model, dummy_input)

traced_model.save("tcn_model.pt")
# torch.onnx.export(
#     traced_model,               # PyTorch model
#     dummy_input,        # Input tensor or tuple of tensors
#     "tcn_model.onnx",           # Output file path
#     export_params=True,  # Store model weights
#     opset_version=12,    # ONNX version
#     do_constant_folding=True,  # Optimize constants
#     input_names=['input'],     # Input names
#     output_names=['output'],   # Output names
#     dynamic_axes={
#         'input': {0: 'batch_size', 1: 'sequence_length'},
#         'output': {0: 'batch_size'}
#     }
# )

