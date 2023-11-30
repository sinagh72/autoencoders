import timm

model = timm.create_model(
    'tf_efficientnetv2_s.in21k_ft_in1k',
    pretrained=True,
    num_classes=2,  # remove classifier nn.Linear
    drop_rate=0.1,
    drop_path_rate=0.1,
)
print(model)