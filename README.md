# TDDepthAnything with Thread Manager and TDPyEnvManager support

This repository is designed to support the following article: https://derivative.ca/community-post/custom-integration-thread-manager-support-third-party-python-library/72023

## Usage

### Use in sample project

* Open TDDepthAnything.toe
* Select TDPyEnvManager
* Click on the pulse parameter `Create From requirements.txt` to create vEnv
* On TDDepthAnything, click on Load Model
* When the model is loaded, you can use the Inference pulse

### Use in standalone project

* From palette, drag n drop TDPyEnvManger
* Select TDPyEnvManager, Activate
* Click on the pulse parameter `Create From requirements.txt` to create vEnv
* Add the TDDepthAnything release .tox (release folder) to your project
* On TDDepthAnything, click on Load Model
* When the model is loaded, you can use the Inference pulse

## Acknowledgment and licensing

[Depth Anything V2 models accessed via Hugging Face Transformers library](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) under [Apache 2.0 license](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)