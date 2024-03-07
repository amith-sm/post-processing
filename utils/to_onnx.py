import onnx
import os
import sys
import torch

sys.path.append(r"C:\Users\transponster\Documents\anshul\postProcessing")
from model.resnet import ResidualClassifier

model_dir = r"D:\anshul\notera\model\post_processing"
model_path = r"D:\anshul\notera\model\post_processing\classifier.pth"

model = ResidualClassifier(num_classes=22)
model.load_state_dict(torch.load(model_path))
# model.eval()
torch_input = torch.randn(1, 3, 64, 64)

torch.onnx.export(model,               # model being run
                  torch_input,                         # model input (or a tuple for multiple inputs)
                  os.path.join(model_dir, "classifier.onnx"),  # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})
# onnx_program = torch.onnx.dynamo_export(model, torch_input)
#
# onnx_program.save(os.path.join(model_dir, "classifier.onnx"))

# onnx_model = onnx.load(os.path.join(model_dir, "classifier.onnx"))
# onnx.checker.check_model(onnx_model)
