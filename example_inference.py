from skimage import io
import torch, os
from PIL import Image
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
from huggingface_hub import hf_hub_download
from skimage.color import rgba2rgb, gray2rgb


def example_inference(image_path, net):
    im_path = image_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.to(device)  
    # prepare input
    model_input_size = [1024,1024]
    orig_im = io.imread(im_path)
    # Convert to RGB if image is grayscale
    if len(orig_im.shape) == 2 or orig_im.shape[2] == 1:
        orig_im = gray2rgb(orig_im)

    # Convert to RGB if image is RGBA
    if orig_im.shape[2] == 4:
        orig_im = rgba2rgb(orig_im)


    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)
    # print(image.shape)
    # inference 
    result=net(image)
    del net

    # print(result[0][0].shape)
    # post process
    result_image = postprocess_image(result[0][0], orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)
    # no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
    # orig_image = Image.open(im_path)
    # no_bg_image.paste(orig_image, mask=pil_im)
    # no_bg_image.save("example_image_no_bg.png")
    return pil_im


# if __name__ == "__main__":
#     example_inference()