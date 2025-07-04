// Training Schedule - SPAN specific settings
const EPOCHS = 1000000; // niter: 1000000 (from paper)
const BATCH_SIZE = 4; // batch_size: min(64, floor(64 / 14)) = 4 (updated based on dataset size)

// Learning Rate Schedule - SPAN specific
const LR = 5e-4; // lr_G: 5e-4 (from paper)
const LR_STEPS = [200000, 400000, 600000, 800000]; // lr_steps: halving every 2e5 iterations (from paper)
const LR_RATE = 0.5; // lr_rate: 0.5 (from paper)

// Adam Optimizer Betas (for Generator)
const ADAM_BETA1 = 0.9; // adam_beta1_G
const ADAM_BETA2 = 0.99; // adam_beta2_G

// Image Configuration - SPAN specific
const INPUT_SIZE = 64; // input_size: 256 (HR) / 4 (upscaling factor) = 64 (LR) - Used for TRAINING ONLY
const GT_SIZE = 256; // gt_size: 256 (HR) (from paper)
const UPSCALING_FACTOR = 4; // scale: 4
const CH_SIZE = 3; // ch_size: 3

// Paths for browser-based operations
const IMAGE_DATA_URL_PREFIX = './Set14/'; // Using Set14 as per existing script, paper uses DF2K
const MODEL_NAME_G = 'span_3xc'; // Renamed generator model
const MODEL_LOAD_PATH_G = `indexeddb://${MODEL_NAME_G}`; // Path for loading generator model
const MODEL_SAVE_PATH_G = `indexeddb://${MODEL_NAME_G}`; // Path for saving generator model


// UI elements
let statusElement;
let epochStatusElement;
let lossStatusElement;
let sampleContainer;
let saveModelBtn;
let loadModelInput;
let startTrainingBtn;
let pauseResumeTrainingBtn; // Changed to a single button
let deleteModelBtn;
let stopTrainingBtn;
let trainingTimeElement; // Renamed to epochTimingElement
let epochTimingElement; // This will now be iteration time
let etaTimeElement; // New element for ETA to end
let enhanceImageInput;
let enhanceImageBtn;
let enhanceResultContainer;
let enhanceProcessingOverlay;
let enhanceEtaElement;
let closeOverlayBtn; // New button for overlay
let spinnerElement; // Reference to the spinner element

// Chart.js related elements
let lossChartCanvas;
let lossChart;
let generatorLossData = [];
let epochLabels = [];
let lrData = []; // To store generator learning rate for charting

// Flag to control training interruption
let stopTrainingFlag = false;
let pauseTrainingFlag = false; // New flag for pausing

// Backend selection is now handled internally
let currentBackend = '';

let generatorModel;


// --- Start of utils.js functions integrated into main.js ---

/**
 * Generates a timestamp string for logging.
 * @returns {string} The current time in HH:MM:SS format.
 */
function getTimestamp() {
    return new Date().toTimeString().split(' ')[0];
}

/**
 * Logs current tensorFlow.js memory usage.
 * @param {object} tf - The TensorFlow.js library object.
 */
function logMemoryUsage(tf) {
    if (tf.getBackend() && tf.memory) {
        const mem = tf.memory();
        console.log(`[${getTimestamp()}] TF Memory: ${mem.numBytes / 1024 / 1024} MB (${mem.numTensors} tensors)`);
    }
}

/**
 * Augments an image tensor with random horizontal flip.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} tensor - The input image tensor (e.g., [H, W, C]).
 * @returns {tf.Tensor} The augmented image tensor.
 */
function augmentImage(tf, tensor) {
    return tf.tidy(() => {
        let aug = tensor.expandDims(0); // Add batch dimension for tf.image.flipLeftRight
        if (Math.random() > 0.5) {
            aug = tf.image.flipLeftRight(aug);
        }
        const result = aug.squeeze(0); // Remove batch dimension
        return result;
    });
}

/**
 * Randomly adjusts brightness of an image tensor
 * @param {tf.Tensor} tensor - Input image tensor [H, W, C] normalized to [0, 1]
 * @param {number} maxDelta - Maximum brightness change (default: 0.1)
 * @returns {tf.Tensor} Brightness-adjusted tensor
 */
function randomBrightness(tensor, maxDelta = 0.1) {
    return tf.tidy(() => {
        const delta = (Math.random() - 0.5) * 2 * maxDelta; // Range: [-maxDelta, maxDelta]
        return tensor.add(delta).clipByValue(0, 1);
    });
}

/**
 * Randomly adjusts contrast of an image tensor
 * @param {tf.Tensor} tensor - Input image tensor [H, W, C] normalized to [0, 1]
 * @param {number} maxFactor - Maximum contrast factor (default: 0.2)
 * @returns {tf.Tensor} Contrast-adjusted tensor

 */
function randomContrast(tensor, maxFactor = 0.2) {
    return tf.tidy(() => {
        const factor = 1 + (Math.random() - 0.5) * 2 * maxFactor; // Range: [1-maxFactor, 1+maxFactor]
        const mean = tf.mean(tensor);
        return tensor.sub(mean).mul(factor).add(mean).clipByValue(0, 1);
    });
}

/**
 * Randomly rotates image by 90, 180, or 270 degrees
 * @param {tf.Tensor} tensor - Input image tensor [H, W, C]
 * @returns {tf.Tensor} Rotated tensor
 */
function randomRotate90(tensor) {
    return tf.tidy(() => {
        const rotation = Math.floor(Math.random() * 4); // 0, 1, 2, or 3
        let result = tensor;
        
        for (let i = 0; i < rotation; i++) {
            // Rotate 90 degrees clockwise: transpose + reverse along width axis
            result = result.transpose([1, 0, 2]).reverse(1);
        }
        
        return result;
    });
}

/**
 * Randomly flips image vertically
 * @param {tf.Tensor} tensor - Input image tensor [H, W, C]
 * @returns {tf.Tensor} Potentially flipped tensor
 */
function randomVerticalFlip(tensor) {
    // Safety check: ensure tensor is a valid TF.js tensor
    if (!(tensor instanceof tf.Tensor)) {
        console.error(`[${getTimestamp()}] Input to randomVerticalFlip is not a tensor, but:`, tensor);
        return tf.zeros(tensor.shape || [1, 1, 3]); // Return a dummy tensor with expected shape if possible
    }

    return tf.tidy(() => {
        if (Math.random() > 0.5) {
            return tensor.reverse(0); // Reverse along height axis
        }
        return tensor.clone();
    });
}

/**
 * Adds random noise to the image
 * @param {tf.Tensor} tensor - Input image tensor [H, W, C] normalized to [0, 1]
 * @param {number} stddev - Standard deviation of noise (default: 0.02)
 * @returns {tf.Tensor} Noisy tensor
 */
function randomNoise(tensor, stddev = 0.02) {
    return tf.tidy(() => {
        const noise = tf.randomNormal(tensor.shape, 0, stddev);
        return tensor.add(noise).clipByValue(0, 1);
    });
}

/**
 * Randomly adjusts saturation (works on RGB images)
 * @param {tf.Tensor} tensor - Input RGB image tensor [H, W, 3] normalized to [0, 1]
 * @param {number} maxFactor - Maximum saturation factor (default: 0.3)
 * @returns {tf.Tensor} Saturation-adjusted tensor
 */
function randomSaturation(tensor, maxFactor = 0.3) {
    return tf.tidy(() => {
        if (tensor.shape[2] !== 3) return tensor.clone(); // Only works on RGB
        
        const factor = 1 + (Math.random() - 0.5) * 2 * maxFactor;
        
        // Convert to grayscale using standard weights
        const gray = tensor.mul([0.299, 0.587, 0.114]).sum(2, true);
        
        // Interpolate between grayscale and original
        return gray.mul(1 - factor).add(tensor.mul(factor)).clipByValue(0, 1);
    });
}

/**
 * Randomly zooms into an image tensor.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} tensor - The input image tensor [H, W, C] normalized to [0, 1].
 * @param {number} maxZoomFactor - The maximum zoom factor (e.g., 2.0 for 2x zoom).
 * @returns {tf.Tensor} The zoomed tensor, which will be larger than the input.
 */
function randomZoom(tf, tensor, maxZoomFactor = 2.0) {
    return tf.tidy(() => {
        const [h, w, c] = tensor.shape;
        const zoomFactor = 1.0 + Math.random() * (maxZoomFactor - 1.0); // Factor between 1.0 and maxZoomFactor
        const zoomedHeight = Math.floor(h * zoomFactor);
        const zoomedWidth = Math.floor(w * zoomFactor);

        // Resize the image. This will create a new tensor.
        return tf.image.resizeBilinear(tensor, [zoomedHeight, zoomedWidth]);
    });
}

/**
 * Crops a tensor from its center to a target size.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} tensor - The input tensor [H, W, C].
 * @param {number} targetSize - The desired square output size (e.g., 64).
 * @returns {tf.Tensor} The centrally cropped tensor.
 */
function centerCrop(tf, tensor, targetSize) {
    return tf.tidy(() => {
        const [h, w, c] = tensor.shape;
        if (h < targetSize || w < targetSize) {
            console.warn(`[${getTimestamp()}] Tensor (${h}x${w}) is smaller than target crop size (${targetSize}). Resizing to target.`);
            // If the tensor is smaller, resize it up to the target size
            return tf.image.resizeBilinear(tensor, [targetSize, targetSize]);
        }
        const startY = Math.floor((h - targetSize) / 2);
        const startX = Math.floor((w - targetSize) / 2);
        return tensor.slice([startY, startX, 0], [targetSize, targetSize, c]);
    });
}

/**
 * Calculates the necessary padding for a square image to avoid black borders after rotation.
 * @param {number} originalSize - The side length of the original square image.
 * @param {number} maxRotationDegrees - The maximum rotation angle in degrees.
 * @returns {number} The amount of padding needed on each side.
 */
function calculateRotationPadding(originalSize, maxRotationDegrees) {
    const maxRotationRadians = maxRotationDegrees * Math.PI / 180;
    // The side length of the smallest square that can contain the rotated original square
    const requiredSize = originalSize * (Math.abs(Math.cos(maxRotationRadians)) + Math.abs(Math.sin(maxRotationRadians)));
    // Padding needed on one side
    const padding = (requiredSize - originalSize) / 2;
    return Math.ceil(padding);
}


/**
 * Enhanced augmentation function that replaces your existing augmentImage
 * Applies random horizontal flip and additional augmentations, then crops to targetOutputSize.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} tensor - Input image tensor [H, W, C] normalized to [0, 1]. This tensor might be larger due to initial padding/zoom.
 * @param {number} targetOutputSize - The desired final output size of the patch.
 * @param {object} options - Augmentation options.
 * @returns {tf.Tensor} Augmented and centrally cropped tensor.
 */
function enhancedAugmentImage(tf, tensor, targetOutputSize, options = {}) {
    return tf.tidy(() => {
        let result = tensor; // This 'tensor' is the initial slice from the image, potentially already zoomed.

        // Apply padding and rotation first
        if (options.randomAngleRotation !== false && Math.random() > 0.5) {
            const angleDegrees = Math.random() * 45; // Random angle between 0 and 45 degrees
            const angleRadians = angleDegrees * Math.PI / 180; // Convert to radians
            
            const paddingAmount = calculateRotationPadding(result.shape[0], 45); // Calculate padding based on current tensor size
            // Corrected: Use tf.mirrorPad instead of tf.image.pad
            let paddedResult = tf.mirrorPad(result, [[paddingAmount, paddingAmount], [paddingAmount, paddingAmount], [0, 0]], 'reflect');
            
            // tf.image.rotateWithOffset expects a 4D tensor [batch, H, W, C]
            let rotatedResult = tf.image.rotateWithOffset(paddedResult.expandDims(0), angleRadians).squeeze(0);
            
            paddedResult.dispose(); // Dispose the intermediate padded tensor
            result = rotatedResult; // Update result to the rotated tensor
        }
        
        // After potential rotation and padding, crop back to targetOutputSize
        result = centerCrop(tf, result, targetOutputSize);

        // Apply other augmentations to the now correctly sized tensor
        if (Math.random() > 0.5) { // Original horizontal flip
            result = result.expandDims(0);
            result = tf.image.flipLeftRight(result);
            result = result.squeeze(0);
        }
        
        if (options.verticalFlip !== false && Math.random() > 0.7) {
            result = randomVerticalFlip(result);
        }
        
        if (options.brightness !== false && Math.random() > 0.6) {
            result = randomBrightness(result, options.brightnessMax || 0.1);
        }
        
        if (options.contrast !== false && Math.random() > 0.6) {
            result = randomContrast(result, options.contrastMax || 0.2);
        }
        
        if (options.rotation !== false && Math.random() > 0.8) {
            result = randomRotate90(result);
        }
        
        if (options.noise !== false && Math.random() > 0.7) {
            result = randomNoise(result, options.noiseStddev || 0.02);
        }
        
        if (options.saturation !== false && Math.random() > 0.7) {
            result = randomSaturation(result, options.saturationMax || 0.3);
        }
        
        return result;
    });
}


/**
 * Extracts random patches of a given size from a given image tensor.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} imgTensor - The full image tensor (e.g., [H, W, C]).
 * @param {number} patchCount - The number of random patches to extract.
 * @param {number} patchSize - The side length of the square patches to extract. This is the *final* desired size.
 * @param {function} getTimestampFn - Function to get a timestamp for logging.
 * @returns {Promise<tf.Tensor>} A stacked tensor of patches (e.g., [N, patchSize, patchSize, C]).
 */
async function extractRandomPatches(tf, imgTensor, patchCount, patchSize, getTimestampFn) {
    const [h, w, c] = imgTensor.shape;
    const patches = [];

    // The initial slice size is just the patchSize, as zoom and rotation padding/cropping
    // will be handled within enhancedAugmentImage.
    const initialExtractSize = patchSize; 

    if (h < initialExtractSize || w < initialExtractSize) {
        console.warn(`[${getTimestampFn()}] Skipping image due to small size for patch extraction: ${h}x${w}. Required: ${initialExtractSize}x${initialExtractSize}`);
        return tf.zeros([0, patchSize, patchSize, 3]); // Return empty tensor if too small
    }

    for (let i = 0; i < patchCount; i++) {
        const patch = tf.tidy(() => {
            const top = Math.floor(Math.random() * (h - initialExtractSize + 1));
            const left = Math.floor(Math.random() * (w - initialExtractSize + 1));
            const initialSlice = imgTensor.slice([top, left, 0], [initialExtractSize, initialExtractSize, c]);
            
            // Apply random zoom to this initially extracted slice.
            // This will return a tensor that is `initialExtractSize * zoomFactor` in size.
            const zoomedPatch = randomZoom(tf, initialSlice, 2.0); // Max 2x zoom factor

            // Apply other enhanced augmentations, and finally crop to patchSize.
            // enhancedAugmentImage now takes care of the final cropping to `patchSize`.
            return enhancedAugmentImage(tf, zoomedPatch, patchSize);
        });
        patches.push(patch);
    }

    let stackedPatches;
    if (patches.length > 0) {
        stackedPatches = tf.stack(patches);
    } else {
        stackedPatches = tf.zeros([0, patchSize, patchSize, 3]);
        console.warn(`[${getTimestampFn()}] No patches were extracted from this image. Returning empty tensor.`);
    }

    patches.forEach(t => t.dispose()); // Dispose the patches in the array after stacking

    return stackedPatches;
}

/**
 * Loads and preprocesses an image from a URL into a TensorFlow tensor.
 * The tensor is normalized to [0, 1].
 * @param {object} tf - The TensorFlow.js library object.
 * @param {string} url - The URL of the image.
 * @param {function} updateStatusFn - Function to update UI status.
 * @param {function} getTimestampFn - Function to get a timestamp for logging.
 * @returns {Promise<tf.Tensor3D>} The image tensor normalized to [0, 1].
 */
async function loadImageAsTensor(tf, url, updateStatusFn, getTimestampFn) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            resolve(tf.browser.fromPixels(img).div(255));
        };
        img.onerror = (err) => {
            console.error(`[${getTimestampFn()}] Failed to load image from URL: ${url}`, err);
            updateStatusFn(`Error: Could not load image from ${url}. Please ensure 'Set14' folder exists with images.`);
            reject(err);
        };
        img.src = url;
    });
}

/**
 * Loads training patches from a single image and returns them immediately.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {string} imageUrl - URL of the image to process.
 * @param {number} patchCount - Number of patches to extract.
 * @param {number} patchSize - The target size for extracted HR patches.
 * @param {function} updateStatusFn - Function to update UI status.
 * @param {function} getTimestampFn - Function to get a timestamp for logging.
 * @returns {Promise<tf.Tensor>} High-resolution patches for this image.
 */
async function loadPatchesFromSingleImage(tf, imageUrl, patchCount, patchSize, updateStatusFn, getTimestampFn) {
    let fullImageTensor = null;
    try {
        // Load image at its native resolution
        fullImageTensor = await loadImageAsTensor(tf, imageUrl, updateStatusFn, getTimestampFn);

        if (fullImageTensor && fullImageTensor.shape && fullImageTensor.shape.length === 3) {
            return await extractRandomPatches(tf, fullImageTensor, patchCount, patchSize, getTimestampFn);
        } else {
            return tf.zeros([0, patchSize, patchSize, 3]);
        }
    } catch (error) {
        console.error(`[${getTimestampFn()}] Error processing image ${imageUrl} for patches:`, error);
        return tf.zeros([0, patchSize, patchSize, 3]);
    } finally {
        if (fullImageTensor) {
            fullImageTensor.dispose();
        }
    }
}


/**
 * Displays a tensor as an image on the webpage using a canvas.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} tensor - The tensor to display (expected to be [0, 255] int32).
 * @param {HTMLElement} parentElement - The DOM element to append the image to.
 * @param {string} title - A title for the image.
 */
async function displayTensorAsImage(tf, tensor, parentElement, title) {
    const displayTensor = tf.tidy(() => tensor.squeeze());
    const canvas = document.createElement('canvas');
    canvas.width = displayTensor.shape[1];
    canvas.height = displayTensor.shape[0];
    await tf.browser.toPixels(displayTensor, canvas);
    const container = document.createElement('div');
    container.style.display = 'inline-block';
    container.style.margin = '10px';
    const h4 = document.createElement('h4');
    h4.textContent = title;
    container.appendChild(h4);
    container.appendChild(canvas);
    parentElement.appendChild(container);
    displayTensor.dispose();
}

/**
 * Downscales an image tensor by a given scale factor.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} tensor - The input image tensor.
 * @param {number} scaleFactor - The factor by which to downscale.
 * @returns {tf.Tensor} The downscaled image tensor.
*/
function downscaleImageTensor(tf, tensor, scaleFactor) {
    return tf.tidy(() => {
        const [batch, height, width, channels] = tensor.shape;
        const newHeight = Math.floor(height / scaleFactor);
        const newWidth = Math.floor(width / scaleFactor);
        // Using resizeBilinear for better quality downscaling
        return tf.image.resizeBilinear(tensor, [newHeight, newWidth]);
    });
}

/**
 * Prepares an original high-resolution image tensor for display.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} originalTensor - The original high-resolution image tensor.
 * @returns {tf.Tensor} The prepared tensor for display.
 */
function visualizeOriginal(tf, originalTensor) {
    return tf.tidy(() => originalTensor.clipByValue(0, 1).mul(255).cast('int32'));
}

/**
 * Prepares a generated high-resolution image tensor for display.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} generatedTensor - The generated high-resolution image tensor.
 * @returns {tf.Tensor} The prepared tensor for display.
 */
function visualizeGenerated(tf, generatedTensor) {
    return tf.tidy(() => {
        return generatedTensor.clipByValue(0, 1).mul(255).cast('int32');
    });
}

/**
 * Displays three images side-by-side.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} originalHRTensor - The original high-resolution image tensor.
 * @param {tf.Tensor} downscaledLRTensor - The downscaled low-resolution image tensor.
 * @param {tf.Tensor} generatedHRTensor - The model's generated high-resolution output tensor.
 * @param {string} titlePrefix - A prefix for the titles of the displayed images.
 * @param {HTMLElement} sampleContainerElement - The DOM element to append the image containers to.
 * @param {function} getTimestampFn - Function to get a timestamp for logging.
 */
async function saveSideBySideImage(tf, originalHRTensor, downscaledLRTensor, generatedHRTensor, titlePrefix, sampleContainerElement, getTimestampFn) {
    tf.engine().startScope();
    try {
        await displayTensorAsImage(tf, visualizeOriginal(tf, downscaledLRTensor), sampleContainerElement, `${titlePrefix} Downscaled Image`);
        await displayTensorAsImage(tf, visualizeOriginal(tf, originalHRTensor), sampleContainerElement, `${titlePrefix} Original Image`);
        await displayTensorAsImage(tf, visualizeGenerated(tf, generatedHRTensor), sampleContainerElement, `${titlePrefix} Generated Image`);
    } catch (error) {
        console.error(`[${getTimestampFn()}] Error during sample visualization:`, error);
        if (sampleContainerElement) sampleContainerElement.innerHTML = '<h3>Sample Visualizations:</h3><p>Error displaying samples.</p>';
    } finally {
        tf.engine().endScope();
    }
}
// --- End of utils.js functions integrated into main.js ---


// Helper to update UI status
function updateStatus(message) {
    if (statusElement) {
        statusElement.textContent = `Status: ${message}`;
    }
}

/**
 * Formats seconds into HH:MM:SS.
 * @param {number} totalSeconds - The total number of seconds.
 * @returns {string} Formatted time string.
 */
function formatTime(totalSeconds) {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = Math.floor(totalSeconds % 60);

    const pad = (num) => num.toString().padStart(2, '0');
    return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
}


// ---------------------------------------------------------------------------
// PIXELSHUFFLE (DepthToSpace) IMPLEMENTATION (FROM DEBUG.JS)
// ---------------------------------------------------------------------------

/**
 * Manual implementation of spaceToDepth, used for the custom gradient.
 */
function spaceToDepthManual(x, blockSize, dataFormat = 'NHWC') {
    return tf.tidy(() => {
        const inputShape = x.shape;
        if (dataFormat === 'NHWC') {
            const [batch, height, width, channels] = inputShape;
            const newHeight = height / blockSize;
            const newWidth = width / blockSize;
            const newChannels = channels * (blockSize * blockSize);
            const reshaped1 = x.reshape([batch, newHeight, blockSize, newWidth, blockSize, channels]);
            const transposed = reshaped1.transpose([0, 1, 3, 2, 4, 5]);
            return transposed.reshape([batch, newHeight, newWidth, newChannels]);
        } else {
            throw new Error("NCHW format not supported in this simplified debug script.");
        }
    });
}

/**
 * Registers a custom gradient for the DepthToSpace operation.
 */
function registerDepthToSpaceGradient() {
    tf.registerGradient({
        kernelName: 'DepthToSpace',
        gradFunc: (dy, saved, attrs) => ({
            x: () => spaceToDepthManual(dy, attrs.blockSize, attrs.dataFormat || 'NHWC')
        })
    });
}

/**
 * Custom TensorFlow.js layer for DepthToSpace (PixelShuffle).
 */
class DepthToSpaceLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.blockSize = config.blockSize;
        this.dataFormat = config.dataFormat || 'NHWC';
    }

    call(inputs) {
        return tf.tidy(() => tf.depthToSpace(Array.isArray(inputs) ? inputs[0] : inputs, this.blockSize, this.dataFormat));
    }

    computeOutputShape(inputShape) {
        if (Array.isArray(inputShape[0])) {
            inputShape = inputShape[0];
        }
        const [batch, height, width, channels] = inputShape;
        const newHeight = height ? height * this.blockSize : null;
        const newWidth = width ? width * this.blockSize : null;
        const newChannels = channels ? channels / (this.blockSize * this.blockSize) : null;
        if (channels !== null && newChannels !== null && !Number.isInteger(newChannels)) {
            throw new Error(`Input channels must be divisible by blockSize^2.`);
        }
        return [batch, newHeight, newWidth, newChannels];
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {
            blockSize: this.blockSize,
            dataFormat: this.dataFormat
        });
        return config;
    }

    static get className() {
        return 'DepthToSpaceLayer';
    }
}
tf.serialization.registerClass(DepthToSpaceLayer);


// ---------------------------------------------------------------------------
// SPAN MODEL CUSTOM LAYERS (FROM DEBUG.JS)
// ---------------------------------------------------------------------------

/**
 * Normalizes input by subtracting a mean and dividing by a range.
 */
class NormalizationLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.mean = config.mean || [0, 0, 0];
        this.range = config.range || 1.0;
    }
    
    call(inputs) {
        const inputTensor = Array.isArray(inputs) ? inputs[0] : inputs;
        return tf.tidy(() => {
            // NOTE: Assuming RGB mean is the same for all channels for simplicity here.
            // A more complex implementation would handle per-channel means.
            const meanTensor = tf.scalar(this.mean[0]);
            const rangeTensor = tf.scalar(this.range);
            return tf.div(tf.sub(inputTensor, meanTensor), rangeTensor);
        });
    }
    
    computeOutputShape(inputShape) {
        return Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
    }
    
    getConfig() {
        const config = super.getConfig();
        config.mean = this.mean;
        config.range = this.range;
        return config;
    }
    
    static get className() {
        return 'NormalizationLayer';
    }
}
tf.serialization.registerClass(NormalizationLayer);

/**
 * Subtracts 0.5 from the input tensor.
 */
class SubtractHalfLayer extends tf.layers.Layer {
    constructor(config) {
        super(config || {});
    }
    
    call(inputs) {
        const inputTensor = Array.isArray(inputs) ? inputs[0] : inputs;
        return tf.tidy(() => tf.sub(inputTensor, 0.5));
    }
    
    computeOutputShape(inputShape) {
        return Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
    }
    
    getConfig() {
        return super.getConfig();
    }
    
    static get className() {
        return 'SubtractHalfLayer';
    }
}
tf.serialization.registerClass(SubtractHalfLayer);

/**
 * Custom fused convolutional layer (Conv3XC).
 */
class Conv3XC extends tf.layers.Layer {
    constructor(config) {
        super(config);
        // Ensure we have proper defaults and handle multiple naming conventions
        this.c_in = config.c_in || config.cIn || config.inputChannels;
        this.c_out = config.c_out || config.cOut || config.outputChannels;
        this.gain = config.gain1 || config.gain || 1;
        this.s = config.s || config.stride || 1;
        this.has_relu = config.relu || false;
        this.use_bias = config.bias !== false;
        
        // Validate parameters
        if (!this.c_in || !this.c_out || isNaN(this.c_in) || isNaN(this.c_out)) {
            throw new Error(`Conv3XC: Invalid channel dimensions. c_in: ${this.c_in}, c_out: ${this.c_out}`);
        }
    }

    build(inputShape) {
        // Handle different input shape formats
        if (Array.isArray(inputShape[0])) {
            inputShape = inputShape[0];
        }
        
        // Validate input shape
        if (!inputShape || inputShape.length < 4) {
            // This can happen during model loading, but our weights depend on constructor args, so it's okay.
            console.warn(`Conv3XC: Received incomplete input shape during build: ${inputShape}. Proceeding with config values.`);
        }
        
        // Ensure all dimensions are valid numbers
        const c_in_val = parseInt(this.c_in);
        const c_out_val = parseInt(this.c_out);
        const gain_val = parseInt(this.gain);
        
        if (isNaN(c_in_val) || isNaN(c_out_val) || isNaN(gain_val)) {
            throw new Error(`Conv3XC: Invalid parameters - c_in: ${c_in_val}, c_out: ${c_out_val}, gain: ${gain_val}`);
        }
        
        // Create weights with validated dimensions
        this.conv1_kernel = this.addWeight('conv1_kernel', [1, 1, c_in_val, c_in_val * gain_val], 'float32', tf.initializers.glorotUniform());
        this.conv2_kernel = this.addWeight('conv2_kernel', [3, 3, c_in_val * gain_val, c_out_val * gain_val], 'float32', tf.initializers.glorotUniform());
        this.conv3_kernel = this.addWeight('conv3_kernel', [1, 1, c_out_val * gain_val, c_out_val], 'float32', tf.initializers.glorotUniform());
        this.sk_kernel = this.addWeight('sk_kernel', [1, 1, c_in_val, c_out_val], 'float32', tf.initializers.glorotUniform());
        
        if (this.use_bias) {
            this.conv1_bias = this.addWeight('conv1_bias', [c_in_val * gain_val], 'float32', tf.initializers.zeros());
            this.conv2_bias = this.addWeight('conv2_bias', [c_out_val * gain_val], 'float32', tf.initializers.zeros());
            this.conv3_bias = this.addWeight('conv3_bias', [c_out_val], 'float32', tf.initializers.zeros());
            this.sk_bias = this.addWeight('sk_bias', [c_out_val], 'float32', tf.initializers.zeros());
        }
        
        super.build(inputShape);
    }

    call(inputs) {
        return tf.tidy(() => {
            const x = Array.isArray(inputs) ? inputs[0] : inputs;
            
            // Main path
            let main = tf.conv2d(x, this.conv1_kernel.read(), 1, 'same');
            if (this.use_bias) {
                main = tf.add(main, this.conv1_bias.read());
            }
            
            main = tf.conv2d(main, this.conv2_kernel.read(), this.s, 'same');
            if (this.use_bias) {
                main = tf.add(main, this.conv2_bias.read());
            }
            
            main = tf.conv2d(main, this.conv3_kernel.read(), 1, 'same');
            if (this.use_bias) {
                main = tf.add(main, this.conv3_bias.read());
            }
            
            // Skip connection
            let skip = tf.conv2d(x, this.sk_kernel.read(), this.s, 'same');
            if (this.use_bias) {
                skip = tf.add(skip, this.sk_bias.read());
            }
            
            // Combine
            let out = tf.add(main, skip);
            if (this.has_relu) {
                out = tf.leakyRelu(out, 0.05);
            }
            return out;
        });
    }

    computeOutputShape(inputShape) {
        if (Array.isArray(inputShape[0])) {
            inputShape = inputShape[0];
        }
        const [batch, height, width] = inputShape;
        return [batch, height ? Math.floor(height / this.s) : null, width ? Math.floor(width / this.s) : null, this.c_out];
    }

    getConfig() {
        const config = super.getConfig();
        config.c_in = parseInt(this.c_in);
        config.c_out = parseInt(this.c_out);
        config.gain1 = parseInt(this.gain);
        config.s = parseInt(this.s);
        config.relu = this.has_relu;
        config.bias = this.use_bias;
        return config;
    }
    
    static get className() {
        return 'Conv3XC';
    }
}
tf.serialization.registerClass(Conv3XC);

/**
 * The Swift Parameter-free Attention Block (SPAB).
 */
class SPAB extends tf.layers.Layer {
    constructor(config) {
        super(config);
        // Handle multiple naming conventions and provide defaults
        this.in_channels = config.in_channels || config.inChannels || config.inputChannels;
        this.mid_channels = config.mid_channels || config.midChannels || this.in_channels;
        this.out_channels = config.out_channels || config.outChannels || config.outputChannels || this.in_channels;
        this.use_bias = config.bias !== false;
        
        // Validate parameters
        if (!this.in_channels || isNaN(this.in_channels)) {
            throw new Error(`SPAB: Invalid in_channels: ${this.in_channels}`);
        }
        
        // Ensure all values are integers
        this.in_channels = parseInt(this.in_channels);
        this.mid_channels = parseInt(this.mid_channels);
        this.out_channels = parseInt(this.out_channels);
        
        if (isNaN(this.in_channels) || isNaN(this.mid_channels) || isNaN(this.out_channels)) {
            throw new Error(`SPAB: Invalid channel dimensions after parsing - in: ${this.in_channels}, mid: ${this.mid_channels}, out: ${this.out_channels}`);
        }
    }

    build(inputShape) {
        // Handle different input shape formats
        if (Array.isArray(inputShape[0])) {
            inputShape = inputShape[0];
        }
        
        // IMPORTANT: Use the reliable channel counts from the constructor, NOT from inputShape,
        // as inputShape can be incomplete during model loading.
        this.c1_r = new Conv3XC({ 
            name: `${this.name}_c1`, 
            c_in: this.in_channels, 
            c_out: this.mid_channels, 
            gain1: 2, 
            bias: this.use_bias 
        });
        this.c2_r = new Conv3XC({ 
            name: `${this.name}_c2`, 
            c_in: this.mid_channels, 
            c_out: this.mid_channels, 
            gain1: 2, 
            bias: this.use_bias 
        });
        this.c3_r = new Conv3XC({ 
            name: `${this.name}_c3`, 
            c_in: this.mid_channels, 
            c_out: this.out_channels, 
            gain1: 2, 
            bias: this.use_bias 
        });
        this.subtractLayer = new SubtractHalfLayer();
        
        // Build the sublayers with proper shapes
        this.c1_r.build(inputShape);
        this.c2_r.build([null, null, null, this.mid_channels]);
        this.c3_r.build([null, null, null, this.mid_channels]);
        this.subtractLayer.build([null, null, null, this.out_channels]);
        
        super.build(inputShape);
    }

    call(inputs) {
        return tf.tidy(() => {
            const x = Array.isArray(inputs) ? inputs[0] : inputs;
            
            // Use the .call() method for direct execution, which is clearer than .apply() here.
            const out1 = this.c1_r.call(x);
            const out1_act = tf.layers.activation({ activation: 'swish' }).apply(out1);
            
            const out2 = this.c2_r.call(out1_act);
            const out2_act = tf.layers.activation({ activation: 'swish' }).apply(out2);
            
            const out3 = this.c3_r.call(out2_act);
            const sigmoid_out = tf.layers.activation({ activation: 'sigmoid' }).apply(out3);
            const sim_att = this.subtractLayer.call(sigmoid_out);
            
            const added_features = tf.add(out3, x);
            const out = tf.mul(added_features, sim_att);
            
            return [out, out1, sim_att];
        });
    }

    computeOutputShape(inputShape) {
        if (Array.isArray(inputShape[0])) {
            inputShape = inputShape[0];
        }
        const [batch, height, width] = inputShape;
        return [
            [batch, height, width, this.out_channels],
            [batch, height, width, this.mid_channels],
            [batch, height, width, this.out_channels]
        ];
    }

    getConfig() {
        const config = super.getConfig();
        config.in_channels = parseInt(this.in_channels);
        config.mid_channels = parseInt(this.mid_channels);
        config.out_channels = parseInt(this.out_channels);
        config.bias = this.use_bias;
        return config;
    }
    
    static get className() {
        return 'SPAB';
    }
}
tf.serialization.registerClass(SPAB);


// ---------------------------------------------------------------------------
// SPAN MODEL DEFINITION (FROM DEBUG.JS)
// ---------------------------------------------------------------------------

/**
 * Creates the main SPAN model architecture.
 * @param {object} config - Model configuration parameters.
 * @param {number} [height=INPUT_SIZE] - The input height for the model.
 * @param {number} [width=INPUT_SIZE] - The input width for the model.
 */
function createSPANModel(config, height = INPUT_SIZE, width = INPUT_SIZE) {
    const { 
        num_in_ch, 
        num_out_ch, 
        feature_channels = 48, 
        upscale = 4, 
        bias = true, 
        img_range = 1.0, 
        rgb_mean = [0, 0, 0] 
    } = config;
    
    // Validate all parameters
    if (!num_in_ch || !num_out_ch || !feature_channels || isNaN(num_in_ch) || isNaN(num_out_ch) || isNaN(feature_channels)) {
        throw new Error(`Invalid model parameters: num_in_ch=${num_in_ch}, num_out_ch=${num_out_ch}, feature_channels=${feature_channels}`);
    }
    
    console.log(`Creating SPAN model for input size ${height}x${width} with parameters:`, {
        num_in_ch, num_out_ch, feature_channels, upscale, bias
    });
    

    const inputs = tf.input({ shape: [height, width, num_in_ch], name: 'input1' });
    
    const norm = new NormalizationLayer({ 
        name: 'norm', 
        mean: rgb_mean, 
        range: img_range 
    }).apply(inputs);
    
    const conv1 = new Conv3XC({ 
        name: 'conv_1', 
        c_in: num_in_ch, 
        c_out: feature_channels, 
        gain1: 2, 
        bias: bias 
    }).apply(norm);
    
    const [b1_out, , ] = new SPAB({ 
        name: 'spab_1', 
        in_channels: feature_channels, 
        mid_channels: feature_channels,
        out_channels: feature_channels,
        bias: bias 
    }).apply(conv1);
    
    const [b2_out, , ] = new SPAB({ 
        name: 'spab_2', 
        in_channels: feature_channels, 
        mid_channels: feature_channels,
        out_channels: feature_channels,
        bias: bias 
    }).apply(b1_out);
    
    const [b3_out, , ] = new SPAB({ 
        name: 'spab_3', 
        in_channels: feature_channels, 
        mid_channels: feature_channels,
        out_channels: feature_channels,
        bias: bias 
    }).apply(b2_out);
    
    const [b4_out, , ] = new SPAB({ 
        name: 'spab_4', 
        in_channels: feature_channels, 
        mid_channels: feature_channels,
        out_channels: feature_channels,
        bias: bias 
    }).apply(b3_out);
    
    const [b5_out, , ] = new SPAB({ 
        name: 'spab_5', 
        in_channels: feature_channels, 
        mid_channels: feature_channels,
        out_channels: feature_channels,
        bias: bias 
    }).apply(b4_out);
    
    // IMPORTANT: The second output of the 6th block is needed for the concatenation layer.
    const [b6_out, b6_intermediate, ] = new SPAB({ 
        name: 'spab_6', 
        in_channels: feature_channels, 
        mid_channels: feature_channels,
        out_channels: feature_channels,
        bias: bias 
    }).apply(b5_out);
    
    const conv2 = new Conv3XC({ 
        name: 'conv_2', 
        c_in: feature_channels, 
        c_out: feature_channels, 
        gain1: 2, 
        bias: bias 
    }).apply(b6_out);
    
    // Correct concatenation based on the model's architecture
    const cat = tf.layers.concatenate({ name: 'cat' }).apply([norm, conv2, b1_out, b6_intermediate]);
    
    const conv_cat = tf.layers.conv2d({ 
        name: 'conv_cat', 
        filters: feature_channels, 
        kernelSize: 1, 
        padding: 'same', 
        useBias: bias 
    }).apply(cat);
    
    const upsampler = tf.layers.conv2d({ 
        name: 'upsampler', 
        filters: num_out_ch * (upscale ** 2), 
        kernelSize: 3, 
        padding: 'same' 
    }).apply(conv_cat);
    
    const pixel_shuffle = new DepthToSpaceLayer({ 
        name: 'pixel_shuffle', 
        blockSize: upscale 
    }).apply(upsampler);
    
    const output = tf.layers.activation({ 
        activation: 'sigmoid', 
        name: 'output' 
    }).apply(pixel_shuffle);
    
    return tf.model({ inputs: inputs, outputs: output, name: 'SPAN_MODEL' });
}


/**
 * Builds the SPAN Generator model.
 * @param {number} [feature_channels=48] - Number of filters in convolutional layers.
 * @param {number} [scale=UPSCALING_FACTOR] - Upscaling factor.
 * @param {number} [height=INPUT_SIZE] - The input height for the model.
 * @param {number} [width=INPUT_SIZE] - The input width for the model.
 */
function buildGenerator(feature_channels = 48, scale = UPSCALING_FACTOR, height = INPUT_SIZE, width = INPUT_SIZE) {
    const generator = createSPANModel({
        num_in_ch: CH_SIZE,
        num_out_ch: CH_SIZE,
        feature_channels: feature_channels,
        upscale: scale,
        bias: true,
        img_range: 1.0,
        rgb_mean: [0, 0, 0]
    }, height, width); // Pass dynamic size here
    
    console.log(`--- Generator Summary for ${height}x${width} input ---`);
    generator.summary();
    console.log(`------------------------------------`);
    
    return generator;
}

/**
 * Computes the pixel-wise L1 loss.
 */
function pixelLoss(hr_real, hr_fake) {
    return tf.losses.absoluteDifference(hr_real, hr_fake);
}

/**
 * Initializes the TensorFlow.js backend.
 */
async function initializeTfBackend() {
    // This needs to be called once before any model operations
    registerDepthToSpaceGradient();

    updateStatus(`Attempting to set backend to WEBGPU...`);
    try {
        await tf.setBackend('webgpu');
        currentBackend = 'webgpu';
        updateStatus(`Backend: WEBGPU.`);
        console.log(`Backend set to: ${tf.getBackend()}`);
        return;
    } catch (error) {
        console.warn(`WebGPU initialization failed. Falling back to WebGL.`);
    }

    updateStatus(`Attempting to set backend to WEBGL...`);
    try {
        await tf.setBackend('webgl');
        currentBackend = 'webgl';
        updateStatus(`Backend: WEBGL.`);
        console.log(`Backend set to: ${tf.getBackend()}`);
        return;
    } catch (error) {
        console.error(`WebGL initialization failed. No suitable GPU backend found.`);
        updateStatus(`Error: No GPU backend (WebGPU or WebGL) is available.`);
    }
}


/**
 * Initializes the Chart.js loss curve chart.
 */
function initializeLossChart() {
    if (lossChart) lossChart.destroy();
    const ctx = lossChartCanvas.getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochLabels,
            datasets: [
                { label: 'Generator Loss (L1)', data: generatorLossData, borderColor: 'rgb(75, 192, 192)', tension: 0.1, fill: false, yAxisID: 'y' },
                { label: 'Learning Rate', data: lrData, borderColor: 'rgb(54, 162, 235)', tension: 0.1, fill: false, yAxisID: 'y1' }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Iteration' } },
                y: { title: { display: true, text: 'Loss' }, beginAtZero: true, position: 'left' },
                y1: { type: 'linear', display: true, position: 'right', title: { display: true, text: 'Learning Rate' }, grid: { drawOnChartArea: false }, beginAtZero: true }
            }
        }
    });
}

/**
 * Updates the loss chart with new data.
 */
function updateLossChart(iteration, genLoss, lr) {
    epochLabels.push(`Iter ${iteration}`);
    generatorLossData.push(genLoss);
    lrData.push(lr);
    lossChart.update();
}

/**
 * Resets the loss chart data.
 */
function resetLossChart() {
    generatorLossData = [];
    epochLabels = [];
    lrData = [];
    if (lossChart) {
        lossChart.destroy();
        initializeLossChart();
    }
}

/**
 * Calculates the learning rate based on step decay.
 */
function calculateStepLR(currentIteration, initialLr, lrSteps, lrRate) {
    let lr = initialLr;
    for (let i = 0; i < lrSteps.length; i++) {
        if (currentIteration >= lrSteps[i]) {
            lr *= lrRate;
        }
    }
    return lr;
}

/**
 * Runs the training process.
 */
async function runTraining() {
    resetLossChart();
    stopTrainingFlag = false;
    pauseTrainingFlag = false;
    startTrainingBtn.disabled = true;
    pauseResumeTrainingBtn.style.display = 'inline-block';
    pauseResumeTrainingBtn.textContent = 'Pause Training';
    stopTrainingBtn.disabled = false;
    deleteModelBtn.disabled = true;
    saveModelBtn.disabled = true;
    loadModelInput.disabled = true;
    enhanceImageBtn.disabled = true;

    updateStatus('Creating Generator Model for Training...');
    // Dispose of old model if it exists
    if (generatorModel) {
        generatorModel.dispose();
    }
    // For training, we always use the fixed-size model.
    generatorModel = buildGenerator();

    updateStatus('Checking for existing Generator Model...');
    try {
        if ((await tf.io.listModels())[MODEL_SAVE_PATH_G]) {
            updateStatus('Loading existing generator weights...');
            const tempGenModel = await tf.loadLayersModel(MODEL_LOAD_PATH_G);
            generatorModel.setWeights(tempGenModel.getWeights());
            tempGenModel.dispose();
            updateStatus('Generator Model weights loaded.');
            console.log('Generator Model weights loaded.');
        } else {
            updateStatus('No existing Generator weights found. Starting fresh.');
            console.log('No existing Generator weights found. Starting fresh.');
        }
    } catch (e) {
        updateStatus(`Error loading weights: ${e.message}. Starting fresh.`);
        console.error(`Error loading models: ${e.message}`);
    }

    const generatorOptimizer = tf.train.adam(LR, ADAM_BETA1, ADAM_BETA2);

    updateStatus('Starting training...');
    const imageFiles = ['baboon.jpeg', 'barbara.jpeg', 'bridge.jpeg', 'coastguard.jpeg', 'comic.jpeg', 'face.jpeg', 'flowers.jpeg', 'foreman.jpeg', 'lenna.jpeg', 'man.jpeg', 'monarch.jpeg', 'pepper.jpeg', 'ppt3.jpeg', 'zebra.jpeg'];
    
    let currentIterationSampleHighResPatch = null;
    let currentIterationSampleLowResPatch = null;
    
    const trainingStartTime = performance.now();
    let totalIterationTime = 0; // To accumulate time for average calculation

    for (let iteration = 0; iteration < EPOCHS; iteration++) {
        if (stopTrainingFlag) {
            updateStatus(`Training stopped at iteration ${iteration + 1}.`);
            break;
        }
        
        while (pauseTrainingFlag) {
            updateStatus('Training Paused. Click Resume to continue.');
            stopTrainingBtn.disabled = true;
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        stopTrainingBtn.disabled = false;

        const currentLr = calculateStepLR(iteration, LR, LR_STEPS, LR_RATE);
        generatorOptimizer.learningRate = currentLr;

        const iterationStartTime = performance.now();
        let iterationGenLoss = 0;
        let iterationBatches = 0;

        imageFiles.sort(() => Math.random() - 0.5);

        if (currentIterationSampleHighResPatch) currentIterationSampleHighResPatch.dispose();
        if (currentIterationSampleLowResPatch) currentIterationSampleLowResPatch.dispose();
        currentIterationSampleHighResPatch = null;
        currentIterationSampleLowResPatch = null;

        for (const file of imageFiles) {
            if (stopTrainingFlag) break;
            while (pauseTrainingFlag) {
                await new Promise(resolve => setTimeout(resolve, 500));
            }

            updateStatus(`Iteration ${iteration + 1}/${EPOCHS} - Processing: ${file}`);
            tf.engine().startScope();
            try {
                // For training, we use fixed-size patches which aligns with our static model.
                const highResPatches = await loadPatchesFromSingleImage(tf, `${IMAGE_DATA_URL_PREFIX}${file}`, BATCH_SIZE, GT_SIZE, updateStatus, getTimestamp);
                
                if (highResPatches.shape[0] < BATCH_SIZE) {
                    highResPatches.dispose();
                    continue;
                }

                const hr_batch = highResPatches.slice([0, 0, 0, 0], [BATCH_SIZE, -1, -1, -1]);
                const lr_batch = downscaleImageTensor(tf, hr_batch, UPSCALING_FACTOR);

                if (!currentIterationSampleHighResPatch) {
                    currentIterationSampleHighResPatch = tf.keep(hr_batch.slice([0, 0, 0, 0], [1, -1, -1, -1]));
                    currentIterationSampleLowResPatch = tf.keep(lr_batch.slice([0, 0, 0, 0], [1, -1, -1, -1]));
                }

                // --- Generator Training (only L1 loss) ---
                const gLoss = generatorOptimizer.minimize(() => {
                    const generated_hr = generatorModel.apply(lr_batch, {training: true});
                    return pixelLoss(hr_batch, generated_hr);
                }, true);

                iterationGenLoss += (await gLoss.data())[0];
                iterationBatches++;

                tf.dispose([hr_batch, lr_batch, highResPatches, gLoss]);

            } catch (err) {
                console.error(`Error processing ${file}:`, err);
            } finally {
                tf.engine().endScope();
            }
        }
        const iterationDuration = (performance.now() - iterationStartTime) / 1000; // in seconds
        totalIterationTime += iterationDuration; // Accumulate time for average
        
        const avgGenLoss = iterationBatches > 0 ? iterationGenLoss / iterationBatches : 0;
        
        epochStatusElement.textContent = `Iteration: ${iteration + 1}/${EPOCHS}`;
        lossStatusElement.textContent = `G Loss: ${avgGenLoss.toFixed(6)}`;
        epochTimingElement.textContent = `${iterationDuration.toFixed(2)}s`;

        // Calculate ETA
        const remainingIterations = EPOCHS - (iteration + 1);
        const averageTimePerIteration = totalIterationTime / (iteration + 1);
        const estimatedTimeRemaining = remainingIterations * averageTimePerIteration;
        etaTimeElement.textContent = formatTime(estimatedTimeRemaining);
        
        updateLossChart(iteration + 1, avgGenLoss, currentLr);

        // Save the model to IndexedDB after each iteration.
        if (!stopTrainingFlag) {
            updateStatus(`Saving model to IndexedDB after iteration ${iteration + 1}...`);
            await generatorModel.save(MODEL_SAVE_PATH_G);
            updateStatus(`Model saved after iteration ${iteration + 1}.`);
        }

        updateStatus(`Visualizing sample for Iteration ${iteration + 1}...`);
        tf.engine().startScope();
        try {
            if (currentIterationSampleHighResPatch && currentIterationSampleLowResPatch) {
                const generatedSample = generatorModel.predict(currentIterationSampleLowResPatch);
                sampleContainer.innerHTML = '';
                await saveSideBySideImage(tf, currentIterationSampleHighResPatch, currentIterationSampleLowResPatch, generatedSample, `Iter ${iteration + 1}`, sampleContainer, getTimestamp);
                generatedSample.dispose();
            }
        } catch(e){
            console.error("Sample visualization error:", e);
        } finally {
            tf.engine().endScope();
        }
    }

    // After training completes or stops
    etaTimeElement.textContent = 'N/A'; // Clear ETA when training is done
    const totalTrainingDuration = ((performance.now() - trainingStartTime) / 1000).toFixed(2);
    
    updateStatus(`Training completed!`);

    startTrainingBtn.disabled = false;
    pauseResumeTrainingBtn.style.display = 'none';
    stopTrainingBtn.disabled = true;
    deleteModelBtn.disabled = false;
    saveModelBtn.disabled = false;
    loadModelInput.disabled = false;
    enhanceImageBtn.disabled = false;
    logMemoryUsage(tf);

    if (currentIterationSampleHighResPatch) currentIterationSampleHighResPatch.dispose();
    if (currentIterationSampleLowResPatch) currentIterationSampleLowResPatch.dispose();
}

async function saveModelToFile() {
    // We save the globally available training model. The enhancement model is temporary.
    const modelToSave = generatorModel || await tf.loadLayersModel(MODEL_LOAD_PATH_G);
    if (!modelToSave) {
        updateStatus('No model available to save. Please train or load one.');
        return;
    }
    
    updateStatus('Saving Generator Model to file...');
    await modelToSave.save('downloads://span-3xc-generator-model');
    updateStatus('Generator Model downloaded.');
    if (!generatorModel) modelToSave.dispose(); // Clean up if we loaded it just for this
}

async function loadModelFromFile(event) {
    if (event.target.files.length === 0) return;
    updateStatus('Loading Generator Model from files...');
    
    if (generatorModel) {
        generatorModel.dispose();
        generatorModel = null;
    }

    try {
        const files = Array.from(event.target.files);
        const jsonFile = files.find(f => f.name.endsWith('.json'));
        const weightFiles = files.filter(f => f.name.endsWith('.bin'));
        if (!jsonFile) {
            throw new Error("Could not find a .json file in the selection.");
        }

        // Load the model from the file. This will become the new base for training.
        generatorModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, ...weightFiles]));
        
        // Also save it to IndexedDB so it can be used for enhancement immediately.
        updateStatus('Saving loaded model to IndexedDB for future use...');
        await generatorModel.save(MODEL_SAVE_PATH_G);

        updateStatus('Generator Model loaded and saved. Ready for training or enhancement.');
        enhanceImageBtn.disabled = false;
        generatorModel.summary();
    } catch (error) {
        updateStatus(`Error loading generator model: ${error.message}`);
        console.error('Error loading generator model from files:', error);
        enhanceImageBtn.disabled = true;
    }
}

async function deleteModel() {
    updateStatus('Deleting model from IndexedDB...');
    try {
        await tf.io.removeModel(MODEL_SAVE_PATH_G);
        if (generatorModel) {
            generatorModel.dispose();
            generatorModel = null;
        }
        updateStatus('Model deleted. Ready for a new start.');
        epochStatusElement.textContent = 'Iteration: N/A';
        lossStatusElement.textContent = 'N/A';
        epochTimingElement.textContent = 'N/A'; // Clear iteration time
        etaTimeElement.textContent = 'N/A'; // Clear ETA
        enhanceImageBtn.disabled = true;
    } catch (error) {
        updateStatus(`Error deleting model: ${error.message}`);
        console.error('Error deleting model:', error);
    }
}

/**
 * Handles the image upload by resizing it to 64x64, running enhancement,
 * and displaying the result.
 */
async function enhanceImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    enhanceResultContainer.innerHTML = '';
    enhanceProcessingOverlay.style.display = 'flex';
    if (spinnerElement) spinnerElement.style.display = 'block';

    // Set backend
    try {
        await tf.setBackend('webgpu');
    } catch (e1) {
        console.warn("WebGPU not available, trying WebGL...");
        try {
            await tf.setBackend('webgl');
        } catch (e2) {
            console.warn("WebGL not available, using CPU.");
            await tf.setBackend('cpu');
        }
    }

    let originalModel;
    try {
        originalModel = await tf.loadLayersModel(MODEL_LOAD_PATH_G);
        updateStatus('Base model loaded.');
    } catch (e) {
        updateStatus('Could not load a pre-trained model from IndexedDB. Please train one first.');
        enhanceProcessingOverlay.style.display = 'none';
        return;
    }

    const startTime = performance.now();
    try {
        // Load the uploaded image into a tensor
        const uploadedImageTensor = await new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(tf.browser.fromPixels(img).div(255));
            img.onerror = reject;
            img.src = URL.createObjectURL(file);
        });
        
        updateStatus(`Resizing image to ${INPUT_SIZE}x${INPUT_SIZE}...`);

        // --- Main Simplification: Resize image and run prediction ---
        const enhancedTensor = tf.tidy(() => {
            // Resize the entire uploaded image to the model's required input size (64x64)
            const resizedInput = tf.image.resizeBilinear(uploadedImageTensor, [INPUT_SIZE, INPUT_SIZE]);

            // Add a batch dimension so the shape is [1, 64, 64, 3]
            const batchedInput = resizedInput.expandDims(0);

    originalModel.summary();
            // Run prediction
            //const prediction = originalModel.predict(resizedInput);
batchedInput.print()
const prediction = originalModel.predict(batchedInput);
prediction.print()
            // Remove the batch dimension from the output
            return prediction.squeeze();
        });

        // --- Display results ---
        enhanceResultContainer.innerHTML = '<h3>Enhanced Image:</h3>';
        // Show the original, un-resized image
        await displayTensorAsImage(tf, visualizeOriginal(tf, uploadedImageTensor), enhanceResultContainer, 'Your Original Input');
        // Show the upscaled output
        await displayTensorAsImage(tf, visualizeGenerated(tf, enhancedTensor), enhanceResultContainer, `Upscaled Output (from 64x64)`);

        enhanceEtaElement.textContent = `${((performance.now() - startTime) / 1000).toFixed(2)} seconds`;
        if (spinnerElement) spinnerElement.style.display = 'none';
        if (closeOverlayBtn) closeOverlayBtn.style.display = 'block';
        updateStatus('Image enhancement complete!');

        // --- Clean up all tensors ---
        uploadedImageTensor.dispose();
        enhancedTensor.dispose();

    } catch (error) {
        console.error('Error during enhancement:', error);
        updateStatus(`Error enhancing image: ${error.message}`);
        enhanceEtaElement.textContent = 'Error';
        if (spinnerElement) spinnerElement.style.display = 'none';
        if (closeOverlayBtn) closeOverlayBtn.style.display = 'block';
    } finally {
        if (originalModel) originalModel.dispose();
        logMemoryUsage(tf);
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    statusElement = document.getElementById('status');
    epochStatusElement = document.getElementById('epoch-status');
    lossStatusElement = document.getElementById('loss-status');
    sampleContainer = document.getElementById('sample-images').querySelector('.sample-grid');
    saveModelBtn = document.getElementById('save-model-btn');
    loadModelInput = document.getElementById('load-model-input');
    startTrainingBtn = document.getElementById('start-training-btn');
    pauseResumeTrainingBtn = document.getElementById('pause-resume-training-btn');
    deleteModelBtn = document.getElementById('delete-model-btn');
    stopTrainingBtn = document.getElementById('stop-training-btn');
    epochTimingElement = document.getElementById('epoch-time');
    etaTimeElement = document.getElementById('eta-time');
    lossChartCanvas = document.getElementById('lossChart');
    enhanceImageInput = document.getElementById('enhance-image-input');
    enhanceImageBtn = document.getElementById('enhance-image-btn');
    enhanceResultContainer = document.getElementById('enhance-results');
    enhanceProcessingOverlay = document.getElementById('enhance-processing-overlay');
    enhanceEtaElement = document.getElementById('enhance-eta');
    closeOverlayBtn = document.getElementById('close-overlay-btn');
    spinnerElement = enhanceProcessingOverlay.querySelector('.spinner');


    initializeLossChart();
    await initializeTfBackend();

    startTrainingBtn.addEventListener('click', runTraining);
    
    pauseResumeTrainingBtn.addEventListener('click', () => {
        pauseTrainingFlag = !pauseTrainingFlag;
        if (pauseTrainingFlag) {
            pauseResumeTrainingBtn.textContent = 'Resume Training';
            updateStatus('Training Paused.');
            stopTrainingBtn.disabled = true;
        } else {
            pauseResumeTrainingBtn.textContent = 'Pause Training';
            updateStatus('Resuming training...');
            stopTrainingBtn.disabled = false;
        }
    });

    saveModelBtn.addEventListener('click', saveModelToFile);
    loadModelInput.addEventListener('change', loadModelFromFile);
    deleteModelBtn.addEventListener('click', deleteModel);
    stopTrainingBtn.addEventListener('click', () => {
        stopTrainingFlag = true;
        updateStatus('Stopping training requested...');
        pauseResumeTrainingBtn.disabled = true;
    });
    
    stopTrainingBtn.disabled = true;
    pauseResumeTrainingBtn.style.display = 'none';
    
    try {
        const modelList = await tf.io.listModels();
        if (modelList[MODEL_SAVE_PATH_G]) {
            enhanceImageBtn.disabled = false;
            updateStatus('Ready. Found existing model in IndexedDB.');
        } else {
            enhanceImageBtn.disabled = true;
            updateStatus('Ready. No model in IndexedDB. Please train a model first.');
        }
    } catch (e) {
        console.warn("Could not check for existing model, keeping enhance button disabled:", e);
        enhanceImageBtn.disabled = true;
    }


    enhanceImageBtn.addEventListener('click', () => enhanceImageInput.click());
    enhanceImageInput.addEventListener('change', enhanceImage);
    
    if (enhanceProcessingOverlay) enhanceProcessingOverlay.style.display = 'none';
    if (spinnerElement) spinnerElement.style.display = 'none';
    
    if (closeOverlayBtn) {
        closeOverlayBtn.addEventListener('click', () => {
            enhanceProcessingOverlay.style.display = 'none';
            enhanceResultContainer.innerHTML = '';
            closeOverlayBtn.style.display = 'none';
        });
    }

    updateStatus('Ready to train or enhance.');
    logMemoryUsage(tf);
});
