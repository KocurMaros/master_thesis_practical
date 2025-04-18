import torch
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
from approach.ResEmoteNet import ResEmoteNet
from PIL import Image

# Load the model
model = ResEmoteNet()
model.eval()

# First, let's inspect the model structure to find available layers
print("Model structure:")
for name, module in model.named_modules():
    print(f"{name}: {module.__class__.__name__}")

# Based on the printed structure, select an interesting layer to visualize
# Let's use one of the convolutional layers from a residual block
layer_to_visualize = model.res_block2.conv1  # Using the first conv layer from res_block2

# Load image
image_path = "train_00103_aligned.jpg"
image = Image.open(image_path).convert('RGB')

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_image = transform(image).unsqueeze(0)

# Function to get output from layer
activation = {}

def hook_fn(module, input, output):
    activation['feature_maps'] = output

# Attach hook to selected layer
hook = layer_to_visualize.register_forward_hook(hook_fn)

# Forward pass
with torch.no_grad():
    _ = model(input_image)

# Remove the hook
hook.remove()
feature_maps = activation['feature_maps'].squeeze(0)
num_channels_to_show = min(16, feature_maps.size(0))  # Show up to 16 channels

plt.figure(figsize=(15, 15))
for i in range(num_channels_to_show):
    plt.subplot(4, 4, i + 1)
    plt.imshow(feature_maps[i].cpu().numpy(), cmap='viridis')
    plt.axis('off')

plt.tight_layout()
plt.savefig('feature_maps.png')
plt.show()