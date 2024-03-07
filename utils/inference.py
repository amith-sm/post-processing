import numpy as np
import onnxruntime
import os
import torch


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


model_dir = r"D:\anshul\notera\model\post_processing"
ort_session = onnxruntime.InferenceSession(os.path.join(model_dir, "classifier.onnx"),
                                           providers=["CPUExecutionProvider"])

import cv2
import torchvision.transforms as transforms

img = cv2.imread(r"D:\anshul\notera\data\class\1\20231128_150923_1.png")
img = cv2.resize(img, (64, 64))
# img = Image.open(r"D:\anshul\notera\data\class\1\20231128_150923_1.png")
#
# resize = transforms.Resize([64, 64])
# img = resize(img)

# img_ycbcr = img.convert('YCbCr')
# img_y, img_cb, img_cr = img_ycbcr.split()

# to_tensor = transforms.ToTensor()
# img_y = to_tensor(img)
img_tensor = torch.from_numpy((img/255).astype(np.float32))
img_tensor = img_tensor.permute(2, 0, 1)
img_tensor.unsqueeze_(0)
print(img_tensor.shape)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_tensor)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]
print(np.argmax(img_out_y))
