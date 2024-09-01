from PIL import Image
import torch

from .dataloader import norm_RGB

def load_image(image_path: str) -> torch.tensor:
    """load image from path to torch.tensor

    Args:
        image_path (str): image file path

    Returns:
        torch.tensor: image tensor
    """
    img = Image.open(image_path).convert('RGB')
    W, H = img.size
    img = norm_RGB(img)[None] 
    return img
