---
title: "Using BeRT for inpainting squares Part 2"
date: "2022-12-16"
#author: "Me"



showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
#description: "Desc Text."
disableHLJS: true # to disable highlightjs
disableShare: true
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true

UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
---


## A brief update on previous results

Since the last report, the inpainting task from the last report ended up working pretty well! One large change made for stabilization is that each individual training image is shown to the model multiple times in a row (Presentations per Epoch) to stabilize training. This seemed to work, but may promote memorizing, i.e. worse generalization! Here are the parameters used in the successful trial:

{{<table “table table-striped table-bordered">}}
|———-|———-|
| Embedding Dimensions | $640$   |
| Transformer Latent Dim | $1600$  |
| Attention Heads | $8$ | 
| Transformer Blocks | $6$ |
| Num Epochs | $300$ | 
| Training Images | ~ $180$ |
| Presentations per Epoch| $40$ |
| Learning Rate | $10^{-5}$| 
{{</table>}}

To showcase the results, I use two similar problems. First is the problem of masking out a single pixel and seeing if the transformer can solve the task of correctly inpainting the results.

{{< figure src="images/SinglePixelSquareMaskingTrainingLoss.png" width="800" caption="Single pixel masking training loss essentially reached perfect accuracy" align="center">}}

A second slightly harder task involves keeping all else equal and increasing the number of masked pixels from a single pixel to 20 percent of the image. The transformer is able to succeed on this task as well. 

{{< figure src="images/20PercentSquareMasking.png" width="800" caption="Many pixel masking results" align="center">}}


## Moving from squares to rectangles

Okay, so this step is not as simple as the title may make it sound. The big difference in this work compared to the previous experiments is that we massively increase the resolution of the input image to $256 X 256$. Since the number of transformer parameters scales ~quadratically with the number of input pixels, we have to apply a smart "downsampling" (or tokenization) technique so that the number of tokens is within a computationally feasible range (less than 1000). For natural images, a common tokenization technique is to use the quantizer from a pretrained vector-quantized GAN network. 

### Quantization for rectangles

In our case, we do not need a quantization scheme as complex as a VQGAN, so we try the more interpretable and easier to implement method of k-means clustering. Each cluster center will correspond to a token, and any patch can be mapped to the closest centroid, and assigned its corresponding token. 

#### Specifics
To create this set of cluster centers, we generate 100 random rectangles of resolution $256 X 256$ and first downsample them to $16 X 16$. We use area downsampling to capture how many of the pixels were black or white.

{{< figure src="images/DownsampledRectangle.png" width="800" caption=Generating a rectangle, and then downsampling with a factor of 16" align="center">}}

We then take every overlapping $4X4$ patch of this $16X16$ and flatten each of these patches into a length $16$ vector. These vectors are then clustered into 20 clusters. The cluster centers are shown below.

{{< figure src="images/RectangleClusterCenters.png" width="800" caption=K-means clustering of patches of downsampled rectangles" align="center">}}


### First pass: Inpainting on a Grid

In order to ensure that the previously created model was robust enough to classify the increased number of tokens from the quantization scheme for the rectangular problem, we test the efficacy of that model on this new task. For each rectangle, the model is fed all overlapping patches (for example, in a $16x16$ image the model is fed $13X13$ tokens). Like in the previous task, 20 percent of these tokens are masked out and the model tries to guess what these tokens are. At inference time, these tokens are then converted to grayscale images by replacing them with their corresponding cluster centroid, and stitched together to create an output image. This works well! Results are shown below.

{{< figure src="images/RectangleGridInpainting1.png" width="800" align="center">}}
{{< figure src="images/RectangleGridInpainting2.png" width="800" caption="Some test results of grid based rectangle inpainting" align="center">}}

## Sparse Token Prediction (Working Title)

From this problem we take the jump to a much more difficult problem of sparse token prediction on the image. Where in the previous case we had a total of $169$ tokens each corresponding to the same $169$ locations across images, we now move to the smaller space of having a total of $24$ tokens in each image. In addition, each of these tokens can now correspond to any possible continuous coordinate in the image, and we can generate the correct corresponding tokens due to knowing the ground truth rectangle used to generate the image. 

The training scheme for this task is very naive.
1. Generate a set of "training rectangles" (100)
At training time:
2. Pick 24 random points within each image (image here used loosely because technically we treat each rectangle and its corresponding subspace in two dimensions as an infinite resolution image), and get their corresponding images and tokens. 
3. Mask out 20 percent of these points and try to estimate the true token of the masked tokens from their coordinates.

For the dataset of 100 images this model with minimal hyperparameter tuning is able to achieve decent training accuracy, as shown below:

{{< figure src="images/RectangleContinuousTrainingLoss.png" width="800" caption="Training loss for the training scheme listed above" align="center">}}

However for out of sample rectangles, this model does not seem to work well at all which implies that the model is memorizing the training rectangles. 

{{< figure src="images/RectangleTestResults.png" width="800" caption="Some low quality preliminary results. The ground truth rectangle is represented in red." align="center">}}

This could also mean that the model has the capacity to solve this problem, and training just needs to be engineered in such a way that this memorization does not happen as often. For example, using a loss different from cross entropy or trying a different scheme for sampling points or simply generating a new rectangle at every training step.


## Future Work

The next steps for this problem include trying some of the smarter methods mentioned in the previous section. In addition, there are also some other directions to take this work. For example, one way to make this problem easier would be to remove all patches that contain corners from the image. However, in a $16x16$ image, this would drop out a large number of patches (for $4x4$ patches, this would remove around $4x16 = 64$ patches out of a total $169$ patches) that otherwise may contain important information about the edges of the rectangle. 