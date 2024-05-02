# DeepLili Image Generator
## Overview
DeepLili is a Gradio web application that uses the [harpomaxx/deeplili](https://huggingface.co/harpomaxx/deeplili)_ model based on Stable Diffusion 1.4 to generate images in a unique style inspired by Lili Fiallo's Toy Art series. This tool allows users to input a creative prompt and see it transformed into a distinct visual interpretation using AI. The generated images and their corresponding prompts are automatically saved to Dropbox, ensuring easy access and organization.

## Features
- Generate images based on textual prompts.
- Support for both English and Spanish inputs.
- Results inspired by the Toy Art series of Lili Fiallo.
- Automated saving of images and prompts to Dropbox.
- Cleanup of temporary files and directories post-upload.

## Installation
Before you can run the application, make sure you have Python 3.8+ and pip installed. Then, follow these steps to set up your environment:

### Clone the repository:
```
git clone https://github.com/harpomaxx/DeepLili.git
```

### Install the required Python libraries:
```
pip install -r requirements.txt
```

### Dropbox Integration
The application uses Dropbox for file management. To use this feature, follow these steps:

1. Obtain an access token from Dropbox App Console.
2. Place your access token in the oauth_dropbox.py script.
3. Ensure the script handles token refresh and storage securely.
4. The app automatically uploads generated images and their corresponding prompts to your Dropbox account, then cleans up the local temporary files to free up space.

## Usage
To start the application, navigate to the project directory and run:

```
python app.py
``` 
This will start the Gradio interface on http://localhost:7777 by default.

## How to Use the Interface
- Enter your artistic idea into the textbox.
- Click the "Generar obra" button to generate an image based on your input.
- View the generated image directly in the web interface.

## Contributing
Contributions to DeepLili are welcome! Please read the CONTRIBUTING.md file for guidelines on how to make contributions.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
