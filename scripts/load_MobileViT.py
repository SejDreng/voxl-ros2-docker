import torch
import timm

# --- Step 1: Load the pre-trained MobileViT model ---
model_name = "mobilevit_s"  # Choose your preferred variant
model = timm.create_model(model_name, pretrained=True)
model.eval()  # Set to evaluation mode

print(f"Model '{model_name}' loaded successfully!")

# --- Step 2: Save the model's state_dict as a .pt file ---
output_path = f"{model_name}_pretrained.pt"
torch.save(model.state_dict(), output_path)  # Save only the state_dict
print(f"Model saved as: {output_path}")

# --- Step 3: Load the model back from the .pt file ---
# Initialize a new model with the same architecture
loaded_model = timm.create_model(model_name, pretrained=False)
loaded_model.load_state_dict(torch.load(output_path))  # Load the state_dict
loaded_model.eval()  # Set to evaluation mode

print("Model loaded back from .pt file successfully!")

