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

## 📸 Demo

Trains a model to upscale images **4x**, improving perceptual quality using a custom perceptual loss.

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
