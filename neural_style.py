import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

def preprocess_img(image, target_size):
    image = tf.image.resize(image, target_size)
    image = tf.expand_dims(image, axis=0)
    return image

def deprocess_img(processed_image):
    x = processed_image.numpy()  # Convert to NumPy array
    if len(x.shape) == 4:
        x = np.squeeze(x, axis=0)
    assert len(x.shape) == 3, ("Input should have rank 3, got {}".format(x.shape))
    if x.max() > 1.0 or x.min() < 0.0:
        x = (x * 255).astype(np.uint8)
    return x

def content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))

def gram_matrix(input_tensor):
    result = tf.linalg.matmul(input_tensor, tf.transpose(input_tensor, perm=[0, 2, 1]))
    return result

def style_loss(style, gram_target):
    gram_style = gram_matrix(style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def total_variation_loss(inputs):
    x_deltas = inputs[:, :, 1:, :] - inputs[:, :, :-1, :]
    y_deltas = inputs[:, 1:, :, :] - inputs[:, :-1, :, :]
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

def style_content_loss(outputs):
    style_outputs = outputs[:style_layers]
    content_outputs = outputs[style_layers:]

    style_loss_sum = 0
    for name, output in zip(style_layers, style_outputs):
        style_loss_sum += (style_weight/num_style_layers) * style_loss(output[0], style_targets[name])

    content_loss_sum = content_weight * content_loss(content_outputs[content_layer_name][0], content_targets[content_layer_name])
    tv_loss = total_variation_weight * total_variation_loss(generated_image)
    loss = style_loss_sum + content_loss_sum + tv_loss
    return loss

@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = vgg(image)
        loss = style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

# --- Main Execution ---
if __name__ == '__main__':
    # Define file paths
    content_path = 'images/building.jpg' # Replace
    style_path = 'images/drawing.jpg'     # Replace

    # Load and preprocess images
    image_size = 512
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    content_image = preprocess_img(content_image, (image_size, image_size))
    style_image = preprocess_img(style_image, (image_size, image_size))

    # Visualize loaded images
    imshow(deprocess_img(content_image), 'Content Image')
    imshow(deprocess_img(style_image), 'Style Image')

    # Load pre-trained VGG19 model
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # Define content and style layer names
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layers = ['block4_conv2']
    num_style_layers = len(style_layers)

    def vgg_layers(layer_names):
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    style_extractor = vgg_layers(style_layers)
    content_extractor = vgg_layers(content_layers)

    style_outputs = style_extractor(style_image)
    content_outputs = content_extractor(content_image)

    style_targets = {name: gram_matrix(output) for name, output in zip(style_layers, style_outputs)}
    content_targets = {name: output for name, output in zip(content_layers, content_outputs)}
    content_layer_name = content_layers[0]

    # Initialize the generated image with the content image
    generated_image = tf.Variable(content_image)

    # Define weights for loss components
    style_weight = 1e-2
    content_weight = 1e4
    total_variation_weight = 1e-6

    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.02)

    # Training loop
    epochs = 10
    steps_per_epoch = 100

    start_time = time.time()
    for n in range(epochs):
        start = time.time()
        for m in range(steps_per_epoch):
            train_step(generated_image)
            print(".", end='', flush=True)
        end = time.time()
        print(f"Train step: {n}, done in {end-start:.1f} seconds")

    end_time = time.time()
    print("Total training time: {:.2f} seconds".format(end_time - start_time))

    # Display the generated image
    final_image = deprocess_img(generated_image)
    imshow(final_image, 'Generated Image')

    # You can also save the image
    # final_image_pil = tf.keras.preprocessing.image.array_to_img(final_image)
    # final_image_pil.save('styled_image.png')