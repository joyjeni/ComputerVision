import os
import PIL
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def grad_cam(img, model, layer):
	configs = [dict(model_type='resnet', arch=model, layer_name=layer)]
	
	for config in configs:
		config['arch'].to(device).eval()
	
	torch_img = transforms.Compose([transforms.ToTensor()])(img).to(device)
	normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
	
	cams = [[cls.from_config(**config) for cls in (GradCAM, GradCAMpp)] for config in configs]	

	images = []
	for gradcam, gradcam_pp in cams:
		mask, _ = gradcam(normed_torch_img)
		heatmap, result = visualize_cam(mask, torch_img)

		mask_pp, _ = gradcam_pp(normed_torch_img)
		heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

		images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])

	return images
