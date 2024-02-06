## What do you need?

The orca-mini-3b-gguf2-q4_0.gguf model file

A computer with a CPU or GPU

Instructions
Download the orca-mini-3b-gguf2-q4_0.gguf model file from here: https://gpt4all.io/index.html?ref=dataphoenix.info.

Place the model file in the root directory of your project.
Open a terminal and navigate to the directory of your project.
Run the following command to start the model:
```
python3 GPT4ALL.py orca-mini-3b-gguf2-q4_0.gguf **"URL of PDF"** **"your question"** **"cpu or gpu"**
```
- Replace **"URL of PDF"** with the URL of the PDF file you want to process.

- Replace **"your question"** with the question you want to ask the model.

- Replace **"cpu or gpu"** with cpu if you want to use the CPU or gpu if you want to use the GPU.

The model will generate a response to your question.

### Installing PyTorch for CUDA
To use PyTorch with CUDA, you need to install it with the following command:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### Note:

You need to replace cu118 with the version of CUDA that is installed on your computer.
