# TDDepthAnything with Thread Manager and TDPyEnvManager support

This repository is designed to support the following article: https://derivative.ca/community-post/custom-integration-thread-manager-support-third-party-python-library/72023

TDDepthAnything uses Depth Anything V2 to generate a depth map from a
TouchDesigner TOP. Model loading and inference run through Thread Manager so
the TouchDesigner interface remains responsive.

## Dependency setup

The included `TDPyEnvManagerContext.yaml` configures TDPyEnvManager to create a
Python 3.11 virtual environment at `.venv` and install `requirements.txt`
automatically. The initial PyTorch installation is blocking and may make the
first project startup take several minutes.

Requirements are installed automatically when the environment is first
created. They are not reinstalled when `.venv` already exists. After changing
`requirements.txt`, either recreate `.venv` or install the updated requirements
manually through TDPyEnvManager.

## Usage

### Use in sample project

1. Open `TDDepthAnything.toe`.
2. Allow TDPyEnvManager to create `.venv` and install the dependencies. Restart
   TouchDesigner after the initial setup completes.
3. On TDDepthAnything, click **Load Model**. The model is downloaded from
   Hugging Face the first time and then read from the local `checkpoints` cache.
4. When the model is loaded, click **Inference**.

### Use in standalone project

1. Copy `TDPyEnvManagerContext.yaml` and `requirements.txt` into the folder
   containing your `.toe` project.
2. From the palette, add TDPyEnvManager to the project and activate it.
3. Reopen the project and allow TDPyEnvManager to create `.venv` and install the
   dependencies. Restart TouchDesigner after the initial setup completes.
4. Add `release/TDDepthAnything.tox` to the project.
5. On TDDepthAnything, click **Load Model**.
6. When the model is loaded, click **Inference**.

## Acknowledgment and licensing

[Depth Anything V2 models accessed via Hugging Face Transformers library](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) under [Apache 2.0 license](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
