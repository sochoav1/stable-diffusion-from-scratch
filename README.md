# Stable Diffusion From Scratch

## Sample Images

<table>
  <tr>
    <td align="center"><img src="./samples/tmp_t07zp3u.PNG" alt="Sample Image 1" width="400"/></td>
    <td align="center"><img src="./samples/tmp75tyt6fu.PNG" alt="Sample Image 2" width="400"/></td>
  </tr>
</table>

## Description
This project implements a text-to-image generation pipeline using pre-trained diffusion models. It allows users to generate images based on text prompts and offers flexibility in choosing different fine-tuned models.

## Project Structure
```
.
├── data
│   ├── inkpunk-diffusion-v1.ckpt
│   ├── merges.txt
│   ├── v1-5-pruned-emaonly.ckpt
│   └── vocab.json
├── images
|── samples
│   └── dog.jpeg
├── sd
│   ├── __pycache__
│   ├── attention.py
│   ├── clip.py
│   ├── ddpm.py
│   ├── decoder.py
│   ├── demo.ipynb
│   ├── diffusion.py
│   ├── encoder.py
│   ├── model_converter.py
│   ├── model_loader.py
│   └── pipeline.py
├── .gitignore
├── LICENSE
├── makefile
└── README.md
```

## Prerequisites
- Python 3.x
- pip (Python package installer)

## Installation
1. Clone the repository:
   ```
   git clone [https://github.com/sochoav1/Stable-Diffusion-from-scratch]
   cd [project directory]
   ```
2. Install the required dependencies:
   ```
   make install
   ```

## Usage
To run the main script (assuming it's in the root directory):
```
python sd/demo.ipynb
```

Note: The main script is a Jupyter notebook (`demo.ipynb`). You'll need to use Jupyter to run it interactively.

## Build and Formatting
This project uses a Makefile for build automation and code formatting:

- To install dependencies (including Black for formatting):
  ```
  make install
  ```

- To format the code:
  ```
  make format
  ```

- To run both installation and formatting:
  ```
  make all
  ```

## Configuration
The project uses several configuration files:
- `data/inkpunk-diffusion-v1.ckpt`: InkPunk Diffusion model checkpoint
- `data/v1-5-pruned-emaonly.ckpt`: Additional model checkpoint
- `data/vocab.json` and `data/merges.txt`: Tokenizer files

## Customization
1. Refer to `sd/demo.ipynb` for customization options, including:
   - Modifying the prompt for image generation
   - Modifying the unconditonal prompt to tell the model what NOT do
   - Adjusting inference steps
   - Changing the random seed

2. Changing Models:
   Users can change the model by downloading fine-tuned versions from Hugging Face. Here are some options:
   - [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer)
   - [Inkpunk Diffusion](https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main)
   - [Illustration Diffusion](https://huggingface.co/ogkalu/Illustration-Diffusion/tree/main)

   To use a different model:
   a. Download the desired model checkpoint from one of these repositories.
   b. Place the downloaded `.ckpt` file in the `data/` directory.
   c. Update the `model_file` variable in your script to point to the new checkpoint file.

   Example:
   ```python
   model_file = "../data/new-model-checkpoint.ckpt"
   ```

   Note: Ensure that the tokenizer files (`vocab.json` and `merges.txt`) are compatible with the new model. You may need to download these from the same repository as the model checkpoint.

## License
CC0 1.0 Universal

## Acknowledgements
- This project uses various pre-trained diffusion models
- CLIP tokenizer and related components from the Transformers library
- Hugging Face for hosting model repositories
