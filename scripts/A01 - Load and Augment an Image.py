# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
# from tensorflow.python.compiler.mlcompute import mlcompute
# %% [markdown]
# # Load and Augment an Image
# %% [markdown]
# **Expected input data.** Augmenting an image with `imgaug` takes only a few lines of code. But before doing that, we first have to load the image. `imgaug` expects images to be numpy arrays and works best with dtype `uint8`, i.e. when the array's values are in the range `0` to `255`. The channel-axis is always expected to be the last axis and may be skipped for grayscale images. For non-grayscale images, the expected input colorspace is RGB.
# 
# **Non-uint8 data.** If you work with other dtypes than `uint8`, such as `float32`, it is recommended to take a look at the [dtype documentation](https://imgaug.readthedocs.io/en/latest/source/dtype_support.html) for a rough overview of each augmenter's dtype support. The [API](https://imgaug.readthedocs.io/en/latest/source/api.html) contains further details. Keep in mind that `uint8` is always the most thoroughly tested dtype.
# 
# **Image loading function.** As `imgaug` only deals with augmentation and not image input/output, we will need another library to load our image. A common choice to do that in python is `imageio`, which we will use below. Another common choice is OpenCV via its function `cv2.imread()`. Note however that `cv2.imread()` returns images in BGR colorspace and not RGB, which means that you will have to re-order the channel axis, e.g. via `cv2.imread(path)[:, :, ::-1]`. You could alternatively also change every colorspace-dependent augmenter to BGR (e.g. `Grayscale` or any augmenter changing hue and/or saturation). See the [API](https://imgaug.readthedocs.io/en/latest/source/api.html) for details per augmenter. The disadvantage of the latter method is that all visualization functions (such as `imgaug.imshow()` below) are still going to expect RGB data and hence BGR images will look broken.
# %% [markdown]
# ## Load and Show an Image
# 
# Lets jump to our first example. We will use `imageio.imread()` to load an image and augment it. In the code block below, we call `imageio.imread(uri)` to load an image directly from wikipedia, but we could also load it from a filepath, e.g. via `imagio.imread("/path/to/the/file.jpg")` or for Windows `imagio.imread("C:\\path\to\the\file.jpg")`.
# `imageio.imread(uri)` returns a numpy array of dtype `uint8`, shape `(height, width, channels)` and RGB colorspace. That is exactly what we need. After loading the image, we use `imgaug.imshow(array)` to visualize the loaded image.

# %%
import imageio
import imgaug as ia
get_ipython().run_line_magic('matplotlib', 'inline')

image = imageio.imread("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png")

print("Original:")
ia.imshow(image)

# %% [markdown]
# ## Augment the Image
# 
# Now that we have loaded the image, let's augment it. `imgaug` contains many augmentation techniques in the form of classes deriving from the `Augmenter` parent class. To use one augmentation technique, we have to instantiate it with a set of hyperparameters and then later on apply it many times. Our first augmentation technique will be `Affine`, i.e. affine transformations. We keep it simple here and use that technique to rotate the image by a random value between -25° and +25°.

# %%
from imgaug import augmenters as iaa
ia.seed(4)

rotate = iaa.Affine(rotate=(-25, 25))
image_aug = rotate(image=image)

print("Augmented:")
ia.imshow(image_aug)

# %% [markdown]
# ## Augment a Batch of Images
# %% [markdown]
# Of course, in reality we rarely just want to augment a single image. We can achieve this using the same code as above, just changing the signular parameter `image` to `images`. It is often significantly faster to augment a batch of images than to augment each image individually.
# 
# For simplicity, we create a batch here by just copying the original image several times and then feeding it through our rotation augmenter. To visualize our results, we use numpy's `hstack()` function, which combines the images in our augmented batch to one large image by placing them horizontally next to each other.

# %%
import numpy as np

images = [image, image, image, image]
images_aug = rotate(images=images)

print("Augmented batch:")
ia.imshow(np.hstack(images_aug))

# %% [markdown]
# As you can see, all of the images in the batch were automatically rotated by different amounts. That's because when instantiating our affine transformation via `rotate = iaa.Affine(rotate=(-25, 25))`, we used an interval for the rotation, given as `(-25, 25)`, which denotes a uniform distribution `rotate ~ uniform(-25, 25)`. We could have also picked a constant value `rotate=-25` to always rotate by -25° or a list `rotate=[-25, -15, 0]` to rotate by -25° or by -15° or by 0°. We could have also pick many other probability distributions, such as gaussian or poisson distributions. Take a look at the other notebooks or at [the documentation](https://imgaug.readthedocs.io/en/latest/source/parameters.html) for details on how to do that.
# 
# **Lists of images or a single array.** Note that in the example above we used a *list* to combine our images to one batch. We could have also provided a *single numpy array* of shape `(N, H, W, [C])`, where `N` would have been the number of images, `H` their height, `W` their width and `C` (optionally) the channel-axis. Using numpy arrays is generally preferred, as they save memory and can be a little bit faster to augment. However, if your images have different heights, widths or numbers of channels they cannot be combined to a single array and hence a list must be used.
# %% [markdown]
# ## Use Many Augmentation Techniques Simultaneously
# %% [markdown]
# Performing only affine rotations is rather limiting. Therefore, in the next example we will combine several methods and apply them simultaneously to images. To do that, we could instantiate each technique on its own and apply them one after the other by calling `augmenter(images=...)` several times. Alternatively, we can use `Sequential` to combine the different augmenters into one pipeline and then apply them all in a single augmentation call. We will use `Sequential` below to apply affine rotations (`Affine`), add some gaussian noise (`AdditiveGaussianNoise`) and crop the images by removing 0% to 20% from each image side (`Crop`).

# %%
seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(10, 60)),
    iaa.Crop(percent=(0, 0.2))
])

images_aug = seq(images=images)

print("Augmented:")
ia.imshow(np.hstack(images_aug))

# %% [markdown]
# Note how some of the images are zoomed in. This comes from `Crop`. Note also how all of the images still have the same size. This is because `Crop` by default retains the input image size, i.e. after removing pixels it resizes the remaining image back to the input size. If you instead prefer to not resize back to the original image size, instantiate `Crop` as `Crop(..., keep_size=False)`, where `keep_size` denotes "keep the image sizes constant between input and output".
# %% [markdown]
# Above, we used `Sequential` to combine several augmentation techniques. In practice we could have also saved each technique in a list and iterated over the list manually to apply each technique on its own. So while `Sequential` simplified things, it didn't help that much. However, it does have one handy ability that we didn't use yet and that is random order augmenation. If activated, it applies augmentations in random order, greatly increasing the space of possible augmentations and saving us from having to implement that ourselves.
# %% [markdown]
# In the next example we will use that ability. It is as simple as just adding `random_order=True` to `Sequential`. To make things more visible, we increase the strength of the augmentations a bit and show more images. Note also how we augment the input image eight times via a loop instead of using a single call augmentation call for the whole batch. That is because the random order is sampled once *per batch* and not *per image* in the batch. To see many different orders here, we therefore augment several times.

# %%
seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(30, 90)),
    iaa.Crop(percent=(0, 0.4))
], random_order=True)

images_aug = [seq(image=image) for _ in range(8)]

print("Augmented:")
ia.imshow(ia.draw_grid(images_aug, cols=4, rows=2))

# %% [markdown]
# Take a close look at the images above. Some of them were cropped before rotating them and some were first rotated and then cropped. For some of them, the gaussian noise was added to the black pixels and for some not. These black pixels were added from the affine rotation to fill up newly created pixels. Images where the black pixels are not noisy are therefore augmentations where the gaussian noise was applied *before* the rotation.
# %% [markdown]
# ## Augmenting Images of Different Sizes
# %% [markdown]
# It was already mentioned above that `imgaug` supports batches containing images with different sizes, but so far we have always used the same image. The following example shows a case with different image sizes. We load three images from the wikipedia, augment them as a single batch and then show each image one by one (with the input and output shape). We also use some different augmentation techniques this time.

# %%
seq = iaa.Sequential([
    iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),  # crop and pad images
    iaa.AddToHueAndSaturation((-60, 60)),  # change their color
    iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
    iaa.Cutout()  # replace one squared area within the image by a constant intensity value
], random_order=True)

# load images with different sizes
images_different_sizes = [
    imageio.imread("https://upload.wikimedia.org/wikipedia/commons/e/ed/BRACHYLAGUS_IDAHOENSIS.jpg"),
    imageio.imread("https://upload.wikimedia.org/wikipedia/commons/c/c9/Southern_swamp_rabbit_baby.jpg"),
    imageio.imread("https://upload.wikimedia.org/wikipedia/commons/9/9f/Lower_Keys_marsh_rabbit.jpg")
]

# augment them as one batch
images_aug = seq(images=images_different_sizes)

# visualize the results
print("Image 0 (input shape: %s, output shape: %s)" % (images_different_sizes[0].shape, images_aug[0].shape))
ia.imshow(np.hstack([images_different_sizes[0], images_aug[0]]))

print("Image 1 (input shape: %s, output shape: %s)" % (images_different_sizes[1].shape, images_aug[1].shape))
ia.imshow(np.hstack([images_different_sizes[1], images_aug[1]]))

print("Image 2 (input shape: %s, output shape: %s)" % (images_different_sizes[2].shape, images_aug[2].shape))
ia.imshow(np.hstack([images_different_sizes[2], images_aug[2]]))


# %%



