---
title: "Using BeRT for inpainting squares Part 3: Masked Rectangles (minor update)"
date: "2022-1-10"
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

Since the last report, some minor changes to the way that the data for the squares is generated and fed into the model allowed for more general success of the model. 
- The shapes of the rectangles were normalized so that their aspect ratios ranged from 1:1 to 1:2
- The rectangles were all "approximately centered" such that the center of the image contains a point within the rectangle. This point along with the previous were meant to simulate centering and scaling the rectangles. This should make the positional encoding easier.

{{< figure src="images/CenteredRectangle.png" width="200" caption="" align="center">}}

- Selected patches that correspond to unmasked regions are now centered along points on the edges of the rectangle. This more closely corresponds with the expected application of this model: *images that are represented as a sparse set of junctions*. 

With these small changes, the model performs much better. The accuracy of this model is around ~85%, which is pretty good for a 20 class classification task where many of the classes are very similar to each other.

{{< figure src="images/EdgeBasedInputs.png" width="800" caption="Resulting input and outputs for the updated model" align="center">}}

## A example practical application

In order to test this model on a semi-practical set of toy problems, we extend the previous results to a set of occluded rectangles.
{{< figure src="images/MaskedRectangle.png" width="200" caption="Same rectangle as shown above, but the area in gray is not sampled" align="center">}}

In this case all input locations randomly lie along the rectangle edge, but exclude points that are within the masked region. The model is then queried with points inside the masked region, resulting in inpainting using a sparse set of points in the image.

{{< figure src="images/MaskedResults.png" width="800" caption="Resulting input and outputs for the toy problem" align="center">}}
