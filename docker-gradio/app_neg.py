PATH = 'harpomaxx/deeplili' #stable diffusion 1.5
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
#from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
#from diffusers import UniPCMultistepScheduler
#from diffusers import StableDiffusionPipeline
#from PIL import Image
from tqdm.auto import tqdm
import random
import gradio as gr
import tomesd

pipe = StableDiffusionPipeline.from_pretrained(PATH,local_files_only=False ).to("cuda")
tomesd.apply_patch(pipe, ratio=0.5)

#guidance_scale = 8.5

def generate_images(prompt, guidance_scale, n_samples, num_inference_steps, nprompt):
    seeds = [random.randint(1, 10000) for _ in range(n_samples)]
    images = []
    for seed in tqdm(seeds):
        torch.manual_seed(seed)
        image = pipe(prompt, negative_prompt=nprompt,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale).images[0]
        images.append(image)
    return images

def gr_generate_images(prompt: str, num_images: int, num_inference: int, guidance_scale: float, nprompt ):
    prompt = prompt + "sks style"
    images = generate_images(prompt, guidance_scale, num_images, num_inference,nprompt)
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

    # #DeepLili v0.6

    ## Enter your prompt and generate a work of art in the style of Lili's Toy Art paintings.
    ## (English, Spanish)
    """
    )

    with gr.Column(variant="panel"):
        with gr.Row(variant="compact"):
            text = gr.Textbox(
                label="Enter your prompt",
                show_label=True,
                max_lines=2,
                placeholder="a white and black drawing of  a cute character on top of a house with a little animal"
            ).style(
                container=False,
            )
           
        with gr.Row(variant="compact"):
            nprompt = gr.Textbox(
                label="Negative prompt",
                show_label=True,
                max_lines=1
            ).style(
                container=False,
            )


        with gr.Row(variant="compact"):
            num_images_slider = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=1,
                label="Number of Images",
            )

            num_inference_steps_slider = gr.Slider(
                minimum=1,
                maximum=25,
                step=1,
                value=20,
                label="Inference Steps",
            )

            guidance_slider = gr.Slider(
                minimum=1,
                maximum=14,
                step=1,
                value=8,
                label="Guidance Scale",
            )



            btn = gr.Button("Generate image").style(full_width=False)

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(columns=[5], rows=[1], object_fit="contain", height="250px", width="250px")

    btn.click(gr_generate_images, [text, num_images_slider,num_inference_steps_slider,guidance_slider, nprompt], gallery)
    gr.Examples(examples, inputs=[text])
    gr.HTML(
    """
    <h6><a href="https://harpomaxx.github.io/"> harpomaxx </a></h6>
    """
    )

if __name__ == "__main__":
    #demo.launch(share=True)
    demo.queue().launch(share=True,server_name="0.0.0.0")


