---
title: "Using BeRT for inpainting squares"
date: "2022-10-20"
#author: "Me"



showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
#description: "Desc Text."
canonicalURL: "https://canonical.url/to/page"
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
## A motivating toy problem

Suppose I give you a prior in the form of a set of images of black and white squares on an axis-aligned grid. (sample image here)

Next, I give you the following images, and ask you to replace the gray ("masked") squares with either black or white squares in order to fit your prior. 

Easy, right? Humans are able to leverage "long-range" (across image) information in order to determine that each image contains a single square, and the masked squares should be replaced with the correct colors that will make this square.

### So why do we care about this simple example?
Popular deep learning models in computer vision are built off of the back of Convolutional neural networks (CNNs), where neurons in the network each have a limited receptive field size. If the size of the receptive field is too small, the network will not have the ability to incorporate long-range information into it's inference. 

So instead of a CNN, we can instead use a bidirectional transformer (BERT) to solve our toy problem. BERT explicitly computes these long range interactions between squares in the image in the form of attention. 

# How to insert pictures?

![Image](/posts/Firstpost/test_graph.png)
<img src="/posts/Firstpost/test_graph.png" alt="drawing" width="200"/>

# How to insert videos?

{{< video src = "/posts/Firstpost/ArchBoundaries_R21.mp4">}}

## Examples


Inline math: $\(\varphi = \dfrac{1+\sqrt5}{2}= 1.6180339887…\)$


Block math:
$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } } 
$$
