import torch
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18(pretrained=True).eval()

# 1. Build dependency graph for resnet18  这行代码创建了一个DependencyGraph对象，并使用.build_dependency()方法构建了模型的依赖图，其中model是ResNet-18模型，example_inputs是输入的示例张量。
DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, 224, 224))

# 2. Group coupled layers for model.conv1 从依赖图中获取了与model.conv1相关联的剪枝组。tp.prune_conv_out_channels是一个函数，用于指定剪枝操作的方式，idxs=[2, 6, 9]是一个参数用于指定要剪枝的具体索引或位置。
group = DG.get_pruning_group(model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9])

# 3. Prune grouped layers altogether  这段代码首先检查剪枝组是否有效，避免剪枝导致通道数为0的情况。如果剪枝组有效，就调用group.prune()方法对剪枝组中的层进行剪枝操作。
if DG.check_pruning_group(group):  # avoid full pruning, i.e., channels=0.
    group.prune()

# 4. Save & Load
model.zero_grad()  # clear gradients
torch.save(model, 'model.pth')  # We can not use .state_dict as the model structure is changed.
model = torch.load('model.pth')  # load the pruned model