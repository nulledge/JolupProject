import skimage
import skimage.io
from PIL import Image
image = skimage.img_as_float(skimage.io.imread('2.jpg'))
img = Image.fromarray(image)