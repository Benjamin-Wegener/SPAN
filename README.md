# 🚀 Swift Parameterfree Attention Network (SPAN) for Super-Resolution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-F16923?logo=tensorflow)](https://www.tensorflow.org/js)
[![WebGPU Ready](https://img.shields.io/badge/WebGPU-ready-brightgreen)](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-webgpu)

A **TensorFlow.js** implementation of the **Swift Parameterfree Attention Network (SPAN)** for **Single Image Super-Resolution (SISR)**.
Inspired by the paper:
["Swift Parameter-free Attention Network for Efficient Super-Resolution"](https://openaccess.thecvf.com/content/CVPR2024/html/Wan_Swift_Parameter-Free_Attention_Network_for_Efficient_Super-Resolution_CVPR_2024_paper.html)

> 🔍 *Upscale low-resolution images directly in your browser using WebGPU-accelerated models.*

🔗 **GitHub**: <https://github.com/Benjamin-Wegener/SPAN>

## 🔧 Features

* ✅ SPAN Model implemented in TensorFlow.js

* ✅ Hybrid loss: MAE + lightweight VGG perceptual loss

* ✅ WebGPU-first backend (fallback to WebGL/CPU)

* ✅ Train & infer entirely in-browser

* ✅ Save/load models via IndexedDB or local files

* ✅ Real-time training visualization

* ✅ Lightweight and fast for edge deployment

## 📸 Details of Implementation

Here are some points where the code might differ from the explicit details or focus of the paper:

### Loss Function Specifics:

Paper: The paper states, "We train the SPAN with L1 loss, which is widely adopted in SR tasks, combined with a perceptual loss. This hybrid loss function ensures that both pixel-wise accuracy and perceptual quality are optimized." It doesn't specify the exact perceptual loss (e.g., which VGG layer features are used) or the weighting.

Code: The main.js and utils.js files explicitly implement a hybrid loss of MAE (Mean Absolute Error, equivalent to L1) and a lightweight VGG perceptual loss, with a specific VGG_LOSS_WEIGHT = 0.001. While consistent with the paper's general statement, the "lightweight" VGG and the specific weight are implementation choices not detailed in the paper.

### Symmetric Activation Functions:

Paper: The paper describes using "symmetric activation functions," providing tanh as an example for the attention module and noting that "we adopt LeakyReLU in the convolutional layers as the non-linear activation function for its efficiency."

Code: The spanBlock in main.js uses tf.tanh for the symmetric attention, and LeakyReLU for other convolutional layers, which is consistent with the paper's description.

### Specific Hyperparameters for Training:

Paper: The paper mentions training details but might not list every hyperparameter with the exact value (e.g., initial learning rate, weight decay, specific learning rate decay schedule).

Code: main.js defines precise hyperparameters such as EPOCHS = 500, BATCH_SIZE = 4, INITIAL_LEARNING_RATE = 1e-4, LEARNING_RATE_DECAY_EPOCHS = [200, 300], LEARNING_RATE_DECAY_FACTOR = 0.5, and WEIGHT_DECAY = 1e-4. These are concrete values chosen for this implementation, which might be specific to the TensorFlow.js environment or a particular experimental setup not fully detailed in the paper's general methodology.

### Implementation Environment and Optimizations:

Paper: The paper focuses on the model's theoretical design and performance on standard benchmarks, without specifying the underlying deep learning framework (e.g., TensorFlow, PyTorch) or specific hardware/software optimizations for deployment.

Code: The readme.md and main.js clearly state this is a TensorFlow.js implementation with a "WebGPU-first backend (fallback to WebGL/CPU)." This is a significant technical detail related to the deployment and performance in a web environment, which is outside the scope of the core research presented in the paper.

## ▶️ Usage

### 📦 1. Clone or download the repo

You can download the repository as a ZIP file from the GitHub page:

Download ZIP


### 🌐 2. Open in Browser

Open `index.html` in a modern browser that supports **WebGPU** (Chrome, Edge recommended).

The app will:

* Automatically initialize the best available backend (WebGPU > WebGL > CPU)

* Show memory usage and training progress

## ▶️ Controls

| Button             | Functionality                                   |
| :----------------- | :---------------------------------------------- |
| **Start Training** | Begin or resume training                        |
| **Stop Training** | Pause the current epoch                         |
| **Save Model** | Download trained model to disk                  |
| **Load Model** | Upload a saved `.json` model file               |
| **Delete Model** | Clear stored model from IndexedDB               |

## 📈 Real-Time Feedback

* Loss curve updates dynamically using Chart.js

* Sample image upscaling shown during training

* Logs also appear in the browser console (`F12`)

## ⚙️ Dependencies

All dependencies are loaded via CDN:

* [`@tensorflow/tfjs`](https://www.tensorflow.org/js)

* [`@tensorflow/tfjs-backend-webgpu`](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-webgpu)

* [`Chart.js`](https://www.chartjs.org/) – for visualizing training loss

No build tools or bundlers needed!

## 💾 Model Management

Models can be:

* **Saved** as `.json` + binary weights files

* **Loaded** back into the browser

* **Stored** temporarily in IndexedDB

Useful for resuming training or exporting for other apps.

## 📄 Citation

If you use this work, please cite the original paper:

```
@inproceedings{wan2024swift,
  title={Swift Parameter-free Attention Network for Efficient Super-Resolution},
  author={Wan, Cheng and Yu, Hongyuan and Li, Zhiqi and Chen, Yihang and Zou, Yajun and Liu, Yuqing and Yin, Xuanwu and Zuo, Kunlong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6246--6256},
  year={2024}
}
```
