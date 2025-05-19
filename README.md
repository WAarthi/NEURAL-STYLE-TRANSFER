# NEURAL-STYLE-TRANSFER
# Neural Style Transfer

This project implements neural style transfer using TensorFlow. It allows you to combine the content of one image with the style of another to create a new artistic image.

## Overview

The script `style_transfer.py` applies the style of a "style image" to the content of a "content image." It uses a pre-trained VGG19 network to extract style and content features and minimizes a loss function to generate the stylized image.

## Setup

1.  **Prerequisites:**
    * Python 3.6 or higher
    * TensorFlow 2.x
    * Matplotlib
    * PIL (Pillow)
    
2.  **Installation:**
    * Clone this repository (if applicable) or save the code as `style_transfer.py`.
    * Install the required Python packages.  It is highly recommended to use a virtual environment.

        ```bash
        pip install tensorflow matplotlib Pillow
        ```

## How to Use

1.  Save the Python code as `style_transfer.py`.
2.  Create an `images` directory in the same directory as the script.
3.  Place your content image (e.g., `building.jpg`) and style image (e.g., `drawing.jpg`) in the `images` directory.  You can modify the `content_path` and `style_path` variables in the script to point to your own images.
4.  Open your terminal or command prompt.
5.  Navigate to the directory where you saved the file.
6.  Run the script:

    ```bash
    python style_transfer.py
    ```

7.  The script will:
    * Load and preprocess the content and style images.
    * Display the original content and style images.
    * Perform the style transfer process.
    * Display the generated stylized image.

## Code Description

* `style_transfer.py`: Contains the Python script that performs the neural style transfer.  Here's a breakdown of the key functions:
    * `load_img(path_to_img)`: Loads an image from a given path, decodes it, and converts it to a TensorFlow float32 tensor.
    * `imshow(image, title=None)`: Displays a TensorFlow image using Matplotlib.
    * `preprocess_img(image, target_size)`: Resizes and expands the dimensions of an image to prepare it for the VGG19 model.
    * `deprocess_img(processed_image)`:  Converts the processed image tensor back into a format suitable for display (NumPy array, and scaling).
    * `content_loss(content, target)`: Calculates the mean squared error between the content features of the generated image and the target content image.
    * `gram_matrix(input_tensor)`: Computes the Gram matrix of a given input tensor, used to represent style.
    * `style_loss(style, gram_target)`: Calculates the mean squared error between the Gram matrices of the generated image's style features and the target style image's style features.
    * `total_variation_loss(inputs)`: Calculates the total variation loss of the generated image, which encourages smoothness.
    * `style_content_loss(outputs)`: Computes the total loss, which is a weighted sum of the style loss, content loss, and total variation loss.
    * `train_step(image)`: Performs one step of the optimization process using `tf.GradientTape()`. It calculates the gradients of the loss with respect to the generated image and updates the image using the Adam optimizer.

## Image Requirements

* The script assumes the images are located in an `images/` subfolder.
* The script works with JPEG images.  You may need to modify the `load_img` function if you use other image formats.
* The images are resized to 512x512. You can change this by modifying the `image_size` variable.

##  Customization

* **Content and Style Paths:** Change the `content_path` and `style_path` variables to use your own images.
* **Image Size:** Modify the `image_size` variable to change the size of the processed images.
* **Style and Content Layers:** The `style_layers` and `content_layers` lists define which layers of the VGG19 network are used for style and content extraction.  You can experiment with different layers.
* **Loss Weights:** The `style_weight`, `content_weight`, and `total_variation_weight` variables control the contribution of each loss component.  Adjust these values to change the style transfer result.
* **Optimizer:** The script uses the Adam optimizer. You can experiment with other optimizers available in `tf.keras.optimizers`.
* **Training Parameters:** The `epochs` and `steps_per_epoch` variables control the training process.  You can adjust these values to trade off between training time and result quality.

## Example Output

The script will display the content image, the style image, and the final generated image.  The generated image will have the content of the content image and the style of the style image.

## Troubleshooting

* **Resource Exhaustion:** Neural style transfer can be memory-intensive.  If you encounter "ResourceExhaustedError," try reducing the `image_size`.
* **No noticeable style transfer:** Try adjusting the `style_weight` and `content_weight`.  A higher `style_weight` will emphasize the style more.
* **Generated image is too noisy:** Increase the `total_variation_weight` to make the image smoother.

