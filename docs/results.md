---

---
# Results examples

### Grammar + Pix2Pix

Here are same examples of generated images using pipeline, in which
generative grammar generates a segmentation mask and Pix2Pix transaltes
it to a new image.

![grammar + pix2pix example 1](img/example-grammar-pix2pix-1.png)

![grammar + pix2pix example 2](img/example-grammar-pix2pix-2.png)

![grammar + pix2pix example 3](img/example-grammar-pix2pix-3.png)

### Pure grammar generation

We also tried to generate new images just with generative grammar.
We noticed that generated images in this way typically are very similar
to an image from the training data, but with some parts inserted from
other training images. An example is shown below

![pure grammar generation example](img/example-grammar-pure.png)
_On the left - a real photo from training data, on the right - an image
generated with grammars, with differences marked with arrows_
