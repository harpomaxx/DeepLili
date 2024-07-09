PATH = 'harpomaxx/deeplili' #stable diffusion 1.5
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import tomesd
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
#from diffusers import LMSDiscreteScheduler, DDIMScheduler
from tqdm.auto import tqdm
import random
import gradio as gr
import os
import logging
import datetime
from oauth_dropbox import *

torch.backends.cudnn.benchmark = True

# Function to generate and save image and prompt
def generate_image(prompt, guidance_scale, num_inference_steps):
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    
    # Create a timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    full_path = f"deeplili/{timestamp}_{seed}"
    
    os.makedirs(full_path, exist_ok=True)
    with open(f'{full_path}/image.png', 'wb') as f:
        image.save(f, format='PNG')
    with open(f'{full_path}/prompt.txt', 'w') as f:
        f.write(prompt)
   # Initial setup for dropbox
    access_token, refresh_token = load_tokens()
    if not access_token or not refresh_token:
        logging.error("No tokens found, authorization required.")
    else:
        upload_file(f'{full_path}/image.png',f'/{full_path}/image.png')
        upload_file(f'{full_path}/prompt.txt',f'/{full_path}/prompt.txt')
        os.remove(f'{full_path}/image.png')
        os.remove(f'{full_path}/prompt.txt')
        logging.info("Temporary files successfully removed.")
        os.removedirs(full_path)
        logging.info("Temporary dir successfully removed.")
    return image

def gr_generate_images(prompt: str, num_inference = 20, guidance_scale = 8):
    prompt = prompt + " sks style"
    image = generate_image(prompt, guidance_scale, num_inference)
    return image

with gr.Blocks() as demo:
    examples = [
    [
        'A black and white cute character on top of a hill',
        1,
        30
    ],
    [
        'burbujas y montañas en el cielo',
        1,
        20
    ],
    [
        'A tree with multiple eyes and a small flower muted colors',
        1,
        20
    ],
    [
        "Un personaje en 3D en la cima de una colina",
        1,
        20
    ],
    [
        "a poster of a large forest with black and white characters",
        1,
        20
    ],
    ]

    gr.Markdown(
    """
    <img src="https://github.com/harpomaxx/DeepLili/raw/main/images/lilifiallo/660.png" width="150" height="150">

    # #Deeplili MMAMM 2024 

    ## 1. Inspírate: Pensá en una idea para una obra.

    ## 2. Creá: Escribí tu idea y presioná "generar obra".

    ## 3. Descubrí: Observá cómo la IA crea una imagen inspirada en la serie Toy Art de [@fiallolili](https://www.instagram.com/fiallolili/).
    
    ## 4. Sé Parte: Las obras generadas se usaran para componer el mural en esta exhibición.

    ### Podés escribir en español o inglés (aunque el inglés funciona mejor)

    ### Recordá que también podes usarla desde tu celu visitando http://deeplili.co

    """
    )

    with gr.Column(variant="panel"):
        with gr.Row(variant="compact"):
            text = gr.Textbox(
                    label="Escribí acá tu idea:",
                show_label=True,
                max_lines=2,
                placeholder="un personaje simpatico peleando con una maquina, en blanco y negro, kawai"
            )

        with gr.Row(variant="compact"):
         
            btn = gr.Button("Generar obra")

        gallery = gr.Image(
            label="Imagen Generada")  # Use the full view height
    btn.click(gr_generate_images, [text], gallery)
    gr.Examples(examples, inputs=[text])
    gr.HTML(
    """
    <h3 style="text-align: center;">
    <a href="https://www.dropbox.com/scl/fi/zph9qr3sduqd7c6m7nebp/deeplili_stats.png?rlkey=3fwu8498kbkhw0zlvjota2wia&dl=0"> | stats | </a>
    <a href="https://github.com/harpomaxx/DeepLili/" > repository| </a>
    <a href="https://www.dropbox.com/scl/fo/1vh6vx16t2wl9sr9w1v8s/AD4BfWI5H42Jh5CaMUKxoOg?rlkey=vc6l8ppbcuqam4nwhvkvg6ls1&st=lgo2zy24&dl=0"> gallery|</a>
    </h3>
    """
    )
    gr.HTML(
    """
    <h6 style="text-align:left;" ><a href="https://harpomaxx.github.io/">Fine tuned by Harpo MAxx </a></h6>
    """
    )

if __name__ == "__main__":
   # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # Use float16 for GPU to save memory
    else:
        device = "cpu"
        dtype = torch.float32  # CPU does not support float16, use float32 instead


    sched = DPMSolverMultistepScheduler.from_pretrained(PATH, subfolder="scheduler")
    #sched = LMSDiscreteScheduler.from_pretrained(PATH, subfolder="scheduler")
    #sched = DDIMScheduler.from_pretrained(PATH, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(PATH,torch_dtype=dtype, 
            scheduler=sched).to(device)
    tomesd.apply_patch(pipe, ratio=0.5)
    
    
    demo.queue(concurrency_count=2,
            ).launch(server_name="0.0.0.0",server_port=7777)  
