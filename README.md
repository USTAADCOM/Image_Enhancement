# Image_Enhancement
 Image_Enhancement project contains the tools for low-quality Whole Images and Faces Restoration in high quality. It contains the following tools.
* Real EserGain (Image Restoration Tool)
* GFPGain (Face Restoration Tool)
* Code Former (Face Restoration Tool)
* SwinIR (Image Restoration Tool)

## Setup
  ```code
  conda create -n <env_name>
  conda activate <env_name>
  git clone https://github.com/USTAADCOM/Image_Enhancement.git
  cd Image_Enhancement
  pip install -r requirements.txt -q
  ```
## Project Structure
```bash
Image_Enhancement
│   app.py
│   inference_realesrgan.py
│   main_test_swinir.py
│   network_swinir.py
│   README.md
│   requirements.txt
│   srvgg_arch.py
│   util_calculate_psnr_ssim.py
└───images
        AI-generate.jpg
        lincoln.jpg

```
## Run Gradio Demo
```code
python3 app.py 
```
