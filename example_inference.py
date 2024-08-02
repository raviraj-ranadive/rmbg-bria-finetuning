from skimage import io
import torch, os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from skimage.color import rgba2rgb, gray2rgb
from torchvision.transforms.functional import normalize


def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


def model_inference(image_path, net):
    image = np.array(Image.open(image_path))
    orig_image = Image.fromarray(image)
    w,h = orig_im_size = orig_image.size
    image = resize_image(orig_image)
    im_np = np.array(image)
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
    im_tensor = torch.unsqueeze(im_tensor,0)
    im_tensor = torch.divide(im_tensor,255.0)
    im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
    if torch.cuda.is_available():
        im_tensor=im_tensor.cuda()

    result=net(im_tensor)
    # post process
    result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)    
    # image to pil
    im_array = (result*255).cpu().data.numpy().astype(np.uint8)
    pil_im = Image.fromarray(np.squeeze(im_array))
    # paste the mask on the original image
    new_im = Image.new("RGBA", pil_im.size, (0,0,0,0))
    new_im.paste(orig_image, mask=pil_im)
    # new_orig_image = orig_image.convert('RGBA')

    return new_im, pil_im


