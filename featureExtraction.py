# Required libraries
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"  # Model Stable Diffusion
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipeline.to("cpu")

noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)


def preprocess_image(image_path):
    """Preprocesses the image to fit the model's expected input size."""
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # Stable Diffusion uses 512x512 images
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to match model input
    ])
    return preprocess(image).unsqueeze(0).to("cuda")  # Add batch dimension and move to GPU


def add_noise(image_tensor, timestep, noise_scheduler):
    """Adds noise to the image at a given timestep using the noise scheduler."""
    noise = torch.randn_like(image_tensor).to("cuda")
    noisy_image = noise_scheduler.add_noise(image_tensor, noise, timestep)
    return noisy_image, noise


def get_unet_decoder_features(noisy_image, timestep, unet_model):
    """Feeds the noisy image through the UNet and captures intermediate decoder features."""
    with torch.no_grad():
        latents = noisy_image
        # UNet forward pass, capture feature maps from the decoder
        features = []
        for i, layer in enumerate(unet_model.down_blocks):
            latents = layer(latents, timestep)
            if i >= len(unet_model.down_blocks) - len(unet_model.up_blocks):  # Only decoder features
                features.append(latents)
    return features


def main(image_path, timesteps):
    # Load and preprocess the image
    image_tensor = preprocess_image(image_path)

    # Dictionary to store features for each timestep
    all_features = {}

    for timestep in timesteps:
        # Add noise to the image at the specified timestep
        noisy_image, noise = add_noise(image_tensor, timestep, noise_scheduler)

        # Extract decoder features using the UNet model from Stable Diffusion
        unet_model = pipeline.unet
        diffusion_features = get_unet_decoder_features(noisy_image, timestep, unet_model)

        # Store the features for the current timestep
        all_features[timestep] = diffusion_features

        # Visualize the noisy image
        noisy_image_np = noisy_image.squeeze().permute(1, 2, 0).cpu().numpy()
        plt.imshow((noisy_image_np * 0.5 + 0.5).clip(0, 1))  # Denormalize for visualization
        plt.title(f"Noisy Image at Timestep {timestep}")
        plt.axis("off")
        plt.show()

    return all_features  # Dictionary with timesteps as keys and features as values


# Example usage
timesteps = [0, 2, 4, 6]  # List of timesteps for which to extract features
image_path = "data/Query_image/test.jpg"  # Replace with path to your image
features = main(image_path, timesteps)
print("Extracted Diffusion Features for each timestep:", features)
