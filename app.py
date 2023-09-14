"""
statble diffusion main module excute each of the module 
"""
import sys
from subprocess import call
import os
import cv2
from random import randint
from codeformer import CodeFormer
import random
from PIL import Image
import torch
import gradio as gr
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
# SwinIR
os.system('wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth -P experiments/pretrained_models')
def swinir_inference(img: Image)-> Image:
    """
    swinir_inference module take an pillow image as input and return a high
    quality restores image as output.

    Parameters
    ----------
    img: Image
        Pillow image recieve as input.
    Return
    image path: str
        return the image path in string form stored on loacal system.
    """
    os.system('mkdir input_swinir')
    basewidth = 256
    width_percent = (basewidth/float(img.size[0]))
    height_size = int((float(img.size[1])*float(width_percent)))
    img = img.resize((basewidth,height_size), Image.ANTIALIAS)
    img.save("input_swinir/1.jpg", "JPEG")
    os.system('python main_test_swinir.py --task real_sr --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq input_swinir --scale 4')
    return 'results/swinir_real_sr_x4/1_SwinIR.png'

# RealEserGAn
def run_cmd(command: str)-> None:
    """
    run_cmd method execute the cmd command create, delete directories.

    Parameters
    ----------
    command: str
        cmd command user want to execute.
    Return
    ------
    None
    """
    try:
        call(command, shell = True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)
def reslesergan_inference(img: Image, mode: str)-> str:
    """
    reslesergan_inference take Pillow image and mode for image enhancement
    as input.

    Parameters
    ----------
    img: Image
        Pillow image recieve as input.
    mode: str
        image enhancement mode in Real-EsrGan tool.
    
    Return
    ------
    iamge path: str
        enhanced image path as string.
    """
    random_id = randint(1, 10000)
    input_dir = "input_image" + str(random_id) + "/"
    output_dir = "output_image" + str(random_id) + "/"
    run_cmd("rm -rf " + input_dir)
    run_cmd("rm -rf " + output_dir)
    run_cmd("mkdir " + input_dir)
    run_cmd("mkdir " + output_dir)
    basewidth = 256
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.LANCZOS)
    img.save(input_dir + "1.jpg", "JPEG")
    if mode == "base":
        run_cmd("python inference_realesrgan.py -n RealESRGAN_x4plus -i "+ input_dir + " -o " + output_dir)
    else:
        os.system("python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i "+ input_dir + " -o " + output_dir)
    return os.path.join(output_dir, "1_out.jpg")
run_cmd("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P .")
run_cmd("pip install basicsr")
run_cmd("pip freeze")
os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P .")
# GFPGAN
os.system("pip freeze")
if not os.path.exists('realesr-general-x4v3.pth'):
    os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")
if not os.path.exists('GFPGANv1.2.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth -P .")
if not os.path.exists('GFPGANv1.3.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P .")
if not os.path.exists('GFPGANv1.4.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")
if not os.path.exists('RestoreFormer.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth -P .")
if not os.path.exists('CodeFormer.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/CodeFormer.pth -P .")
torch.hub.download_url_to_file(
    'https://user-images.githubusercontent.com/17445847/187401133-8a3bf269-5b4d-4432-b2f0-6d26ee1d3307.png',
    '10045.png')
model = SRVGGNetCompact(num_in_ch = 3, num_out_ch = 3, num_feat = 64,
                        num_conv = 32, upscale = 4, act_type = 'prelu')
MODEL_PATH = 'realesr-general-x4v3.pth'
HALF = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale = 4, model_path = MODEL_PATH, 
                         model = model, tile = 0, tile_pad = 10, 
                         pre_pad = 0, half = HALF)
os.makedirs('output_image_gfpgan', exist_ok = True)
def gfpgan_inference(img, version, scale):
    """
    gfpgan_inference image filepath model version rescalling factor as input
    and return enhnaced face image and path.

    Parameters
    ----------
    img: str
        image path recieve as input.
    version: float
        gfpgan model version as float.
    scale: int
        image rescale factor as int.
    
    Return
    ------
    output: ndarray
        output image as ndarray.
    save_path: str
        output image save path as string.
    """
    print(img, version, scale)
    if scale > 4:
        scale = 4
    try:
        extension = os.path.splitext(os.path.basename(str(img)))[1]
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:  # for gray inputs
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None
        h, w = img.shape[0:2]
        if h > 3500 or w > 3500:
            print('too large size')
            return None, None
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation = cv2.INTER_LANCZOS4)
        if version == 'v1.2':
            face_enhancer = GFPGANer(
            model_path = 'GFPGANv1.2.pth', upscale = 2,
            arch = 'clean', channel_multiplier = 2,
            bg_upsampler = upsampler)
        elif version == 'v1.3':
            face_enhancer = GFPGANer(
            model_path = 'GFPGANv1.3.pth', upscale = 2, arch = 'clean',
            channel_multiplier = 2,
            bg_upsampler = upsampler)
        elif version == 'v1.4':
            face_enhancer = GFPGANer(
            model_path = 'GFPGANv1.4.pth', upscale = 2,
            arch = 'clean', channel_multiplier = 2,
            bg_upsampler = upsampler)
        elif version == 'RestoreFormer':
            face_enhancer = GFPGANer(
            model_path='RestoreFormer.pth', upscale = 2,
            arch = 'RestoreFormer', channel_multiplier = 2,
            bg_upsampler = upsampler)
        try:
            _, _, output = face_enhancer.enhance(img, has_aligned = False,
                                                 only_center_face = False,
                                                 paste_back = True)
        except RuntimeError as error:
            print('Error', error)
        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)),
                                    interpolation = interpolation)
        except Exception as error:
            print('wrong scale input.', error)
        if img_mode == 'RGBA':
            extension = 'png'
        else:
            extension = 'jpg'
        save_path = f'output_image_gfpgan/out.{extension}'
        cv2.imwrite(save_path, output)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output, save_path
    except Exception as error:
        print("global exception", error)
        return None, None
# Code Former
model = CodeFormer()
def codeformer_inference(img):
    """
    codeformer_inference take Pillow image as input and return 
    enhnaced face Pillow image as output.
    
    Parameters
    ----------
    img: Image
        Pillow image recieve as input.
    
    Return
    ------
        Pillow enhanced face image.
    """
    restored_image = model(img)
    return restored_image
with gr.Blocks() as demo:
    gr.Markdown("Image Enhancement task")
    with gr.Tab("Real_EserGAn"):
        gr.Interface(
        reslesergan_inference,
        [gr.inputs.Image(type = "pil",
                         label = "Input"),gr.inputs.Radio(["base","anime"],
                                                        type = "value",
                                                        default = "base",
                                                        label = "model type")],
        gr.outputs.Image(type = "pil", label = "Output"),
        )
    with gr.Tab("GFPGAN"):
        gr.Interface(
        gfpgan_inference,[
        gr.Image(type = "filepath", label = "Input"),
        gr.Radio(['v1.2', 'v1.3', 'v1.4', 'RestoreFormer'],
                 type = "value", value = 'v1.4', label = 'version'),
        gr.Number(label = "Rescaling factor", value = 2),
        ],
        [
        gr.Image(type = "numpy", label = "Output"),
        gr.File(label = "Download the output image")
        ]
        )
    with gr.Tab("Code Former"):
        gr.Interface(
        codeformer_inference,[
        gr.Image(type = "pil", label = "Input")],
        gr.Image(type = "pil", label = "Output"),
        examples=[['images/AI-generate.jpg', 'v1.4', 2, 50],
                  ['images/lincoln.jpg', 'v1.4', 2, 50]]
        )
    with gr.Tab("SwinIR"):
        gr.Interface(
        swinir_inference,
        [gr.inputs.Image(type = "pil", label = "Input")],
        gr.outputs.Image(type = "pil", label = "Output"),
        enable_queue=True
        )
if __name__ == "__main__":
    demo.launch(share = True, debug = True)
