import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'tf_efficientnetv2_s.in21k_ft_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()
model.global_pool = nn.Identity()
print(model)
# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
data_config['input_size'] = (3, 256, 256)
print(data_config)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 1280, 10, 10) shaped tensor

output = model.forward_head(output, pre_logits=True)
print(output.shape)
# output is a (1, num_features) shaped tensor

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
