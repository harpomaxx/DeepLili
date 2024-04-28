# Use the official PyTorch Docker image as the base
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

# Install Jupyter Notebook and JupyterLab
RUN pip install jupyter

# Install hugging face libraries
RUN pip install diffusers[torch] huggingface_hub
RUN pip install transformers
RUN pip install xformers==0.0.16
RUN pip install bitsandbytes
RUN pip install accelerate
RUN pip install dropbox
RUN pip install tomesd

RUN pip install torchvision ftfy tensorboard Jinja2

# Install Gradio
RUN pip install gradio==3.45

# Copy the app.py file into the container
COPY app.py /workspace
COPY oauth_dropbox.py /workspace
COPY dropbox_tokens.json /workspace
COPY config.json /workspace

# Set the entrypoint to run the Gradio app 
ENTRYPOINT ["python", "/workspace/app.py"]

