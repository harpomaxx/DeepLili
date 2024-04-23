PATH = 'harpomaxx/deeplili' #stable diffusion 1.5
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import tomesd

#from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


from tqdm.auto import tqdm
import random
import gradio as gr

def generate_image(prompt, guidance_scale, num_inference_steps):
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    image = pipe(prompt, num_inference_steps=num_inference_steps,guidance_scale=guidance_scale).images[0]
    return image

def gr_generate_images(prompt: str, num_inference = 20, guidance_scale = 8 ):
    prompt = prompt + "sks style"
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
        'Bubbles and mountains in the sky',
        1,
        20
    ],
    [
        'A tree with multiple eyes and a small flower muted colors',
        1,
        20
    ],
    [
        "3d character on top of a hill",
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

    # #DeepLili v0.5b

    ## Enter your prompt and generate a work of art in the style of Lili's Toy Art paintings.

    ## (English, Spanish)
    """
    )

    with gr.Column(variant="panel"):
        with gr.Row(variant="compact"):
            text = gr.Textbox(
                label="Enter your prompt",
                show_label=False,
                max_lines=2,
                placeholder="a white and black drawing of  a cute character on top of a house with a little animal"
            )

        with gr.Row(variant="compact"):
         
            btn = gr.Button("Generate image")

        gallery = gr.Image(
            label="Generated image")  # Use the full view height
    btn.click(gr_generate_images, [text], gallery)
    gr.Examples(examples, inputs=[text])
    gr.HTML(
    """
    <h6><a href="https://harpomaxx.github.io/"> harpomaxx </a></h6>
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


    dpm = DPMSolverMultistepScheduler.from_pretrained(PATH, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(PATH,torch_dtype=dtype, scheduler=dpm).to(device)
    tomesd.apply_patch(pipe, ratio=0.5)
    
    demo.queue(concurrency_count=2,
            ).launch(server_name="0.0.0.0",server_port=7777)  
