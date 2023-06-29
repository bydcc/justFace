import torch

model_dir = (
    './src/pretrained_ckpts/face_parsing/segnext.small.best_mIoU_iter_140000.pth'
)
dict = torch.load(model_dir, map_location=torch.device('cpu'))

for i in dict['state_dict']:
    print(i)


old_names = [
    "backbone.block1.0.mlp.dwconv.dwconv.weight",
    "backbone.block1.0.mlp.dwconv.dwconv.bias",
    "backbone.block1.1.mlp.dwconv.dwconv.weight",
    "backbone.block1.1.mlp.dwconv.dwconv.bias",
    "backbone.block2.0.mlp.dwconv.dwconv.weight",
    "backbone.block2.0.mlp.dwconv.dwconv.bias",
    "backbone.block2.1.mlp.dwconv.dwconv.weight",
    "backbone.block2.1.mlp.dwconv.dwconv.bias",
    "backbone.block3.0.mlp.dwconv.dwconv.weight",
    "backbone.block3.0.mlp.dwconv.dwconv.bias",
    "backbone.block3.1.mlp.dwconv.dwconv.weight",
    "backbone.block3.1.mlp.dwconv.dwconv.bias",
    "backbone.block3.2.mlp.dwconv.dwconv.weight",
    "backbone.block3.2.mlp.dwconv.dwconv.bias",
    "backbone.block3.3.mlp.dwconv.dwconv.weight",
    "backbone.block3.3.mlp.dwconv.dwconv.bias",
    "backbone.block4.0.mlp.dwconv.dwconv.weight",
    "backbone.block4.0.mlp.dwconv.dwconv.bias",
    "backbone.block4.1.mlp.dwconv.dwconv.weight",
    "backbone.block4.1.mlp.dwconv.dwconv.bias",
]

new_names = [
    "backbone.block1.0.mlp.dwconv.weight",
    "backbone.block1.0.mlp.dwconv.bias",
    "backbone.block1.1.mlp.dwconv.weight",
    "backbone.block1.1.mlp.dwconv.bias",
    "backbone.block2.0.mlp.dwconv.weight",
    "backbone.block2.0.mlp.dwconv.bias",
    "backbone.block2.1.mlp.dwconv.weight",
    "backbone.block2.1.mlp.dwconv.bias",
    "backbone.block3.0.mlp.dwconv.weight",
    "backbone.block3.0.mlp.dwconv.bias",
    "backbone.block3.1.mlp.dwconv.weight",
    "backbone.block3.1.mlp.dwconv.bias",
    "backbone.block3.2.mlp.dwconv.weight",
    "backbone.block3.2.mlp.dwconv.bias",
    "backbone.block3.3.mlp.dwconv.weight",
    "backbone.block3.3.mlp.dwconv.bias",
    "backbone.block4.0.mlp.dwconv.weight",
    "backbone.block4.0.mlp.dwconv.bias",
    "backbone.block4.1.mlp.dwconv.weight",
    "backbone.block4.1.mlp.dwconv.bias",
]


for index, old_name in enumerate(old_names):
    # 修改参数名
    dict['state_dict'][new_names[index]] = dict['state_dict'].pop(old_name)
torch.save(dict, './model_changed.pth')
