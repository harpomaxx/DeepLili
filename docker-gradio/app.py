#PATH = '/home/harpo/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors'
#PATH = 'harpomaxx/deeplili' #stable diffusion 1.5
PATH = '/home/harpo/stable-diffusion-webui/models/Stable-diffusion/deeplili.safetensors'
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from tqdm.auto import tqdm
import random
import gradio as gr

#pipe = StableDiffusionPipeline.from_single_file(PATH,torch_dtype=torch.float16).to("cuda:0")
#pipe = StableDiffusionPipeline.from_single_file(PATH,local_files_only=False, use_safetensors= True ).to("cuda")
#guidance_scale = 8.5

def generate_images(prompt, guidance_scale, n_samples, num_inference_steps):
    seeds = [random.randint(1, 10000) for _ in range(n_samples)]
    images = [] 
    for seed in tqdm(seeds):
        torch.manual_seed(seed)
        image = pipe(prompt, num_inference_steps=num_inference_steps,guidance_scale=guidance_scale).images[0]   
        images.append(image)
    return images

def gr_generate_images(prompt: str, num_images = 1, num_inference = 20, guidance_scale = 8 ):
    prompt = prompt + "sks style"
    images = generate_images(prompt, guidance_scale, num_images, num_inference)
    return images

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
            ).style(
                container=False,
            )
          
        with gr.Row(variant="compact"):
          #  num_images_slider = gr.Slider(
          #      minimum=1,
          #      maximum=10,
          #      step=1,
          #      value=1,
          #      label="Number of Images",
          #  )
   
          #  num_inference_steps_slider = gr.Slider(
          #      minimum=1,
          #      maximum=25,
          #      step=1,
          #      value=20,
          #      label="Inference Steps",
          #  )

          #  guidance_slider = gr.Slider(
          #      minimum=1,
          #      maximum=14,
          #      step=1,
          #      value=8,
          #      label="Guidance Scale",
          #  )



            btn = gr.Button("Generate image").style(full_width=False)
      
        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(columns=[1], rows=[1], object_fit="contain", height="512px", width="512px")

    num_images_slider = 1
    num_inference_steps_slider = 20
    guidance_slider = 8

    btn.click(gr_generate_images, [text], gallery)
    gr.Examples(examples, inputs=[text])
    gr.HTML(
    """
    <h6><a href="https://harpomaxx.github.io/"> harpomaxx </a></h6>
    """
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ip",
        default="0.0.0.0",
        help="The IP address to of the server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="The port used"
    )
    parser.add_argument(
        "--gpuid",
        default="0",
        help="The gpu id"
    )

    args = parser.parse_args()
    pipe = StableDiffusionPipeline.from_single_file(PATH,torch_dtype=torch.float16).to(f"cuda:{args.gpuid}")
    #demo.launch(share=True)
    demo.queue(concurrency_count=2,
            ).launch(server_name = args.ip, server_port = args.port)
