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

Suppose I give you a prior in the form of a set of images of black and white squares on an axis-aligned grid.


{{< figure src="images/PriorSample.png" width="400" caption="Our prior: squares on a grid" align="center">}}


Next, I give you the following images, and ask you to replace the gray ("masked") squares with either black or white squares in order to fit your prior. 

{{< figure src="./images/MaskedSample2.png" width="400" caption="Our inputs: masked squares" align="center">}}

Easy, right? Humans are able to leverage "long-range" (across image) information in order to determine that each image contains a single square, and the masked squares should be replaced with the correct colors that will make this square.

{{< figure src="../images/MasktoPrior.png" width="800" caption="Our goal: Train a network to recover the squares" align="center">}}

### So why do we care about this simple example?
Popular deep learning models in computer vision are built off of the back of Convolutional neural networks (CNNs), where neurons in the network each have a limited receptive field size. If the size of the receptive field is too small, the network will not have the ability to incorporate long-range information into it's inference. 

So instead of a CNN, we can instead use a bidirectional transformer (BERT) to solve our toy problem. BERT explicitly computes these long range interactions between squares in the image in the form of attention. 

# A note on quantization

BERT was initially developed as a natural language processing (NLP) model, so the authours had to find a way to convert the input words into numbers that can be interpreted by a neural network. BERT uses a common NLP approach, mapping each word to a a scalar, e.g. it's index in a dictionary. Drawing analog to a black and white image, it makes sense to have a mapping dictionary consisting of two entries: ```black -> 0, white -> 1```. For our example, we can represent the masked patches as a third dictionary entry ```[mask] -> 2```.

This idea of mapping elements to scalars can be extended to more complex images, and even to patches of images (check out vector quantized gans)! In particular, representing an image as quantized patch can be helpful in reducing the number of transformer parameters by reducing the number of attention weights calculated.   


# The key parts of the BERT model

As commonly implemented, BERT is very simple (or at least more simple than the name would imply). BERT takes in quantized inputs and learns a sentence completion task by randomly masking out words in a sentence and learning to put the correct word back in place of the mask. We can draw analog to this task by training a network to inpaint images with arbitary masked patches.

For an image with some masked patches, BERT predicts the true value of the masked patch through a couple of key steps.

1. **Embed** the information of the patch (quantized color and positional information). 
2. **Update** the patch embedding with the *attention-based weightings* with all other patches.
3. **Predict** the true value of the patch, usually through the use of a simple multi-layer perceptron (MLP) neural network.

# Model specifics

## Input data

As shown in the motivating examples, each image in the training set is a 8x8 black and white image.  Each image contains a square with a **side length ranging from 2 to 6 units**, located in a random location in the image. All possible permutations of this input results in **126 input images**. Each image is then flattened into a 64 length vector, with each element chosen using the binary quantization scheme. 

## Masking

Before talking about embedding it is worth remembering that rather than two embeddings corresponding to black and white, we will need one more corresponding to a mask element for our image inpainting task. The masks are computed using the MaskGIT masking scheme, where a random fraction $\gamma(r)$ of patches are masked out. Specifically:
$$
\gamma(r) = \cos(\frac{\pi}{2} r)
$$

During training $r$ is drawn like $r \sim Unif(0,1)$, while the assumption is that at inference time you will input a masked image into the model, so you will not provide $r$. 


## Embedding

As a recap, each input to the model is a length 64 vector with each element being a key from the dictionary ```{0:black, 1:white, 2:[mask]}```. Now we must embed this vector in an arbitary higher dimensional latent space that will correspond to the learned codebook. In my case I use a **codebook dimension of $d_{model} = 768$** as used in MaskGIT. 

Along side the codebook encoding, we also need to include a positional encoding to each encoded key. Treating the input image as a cartesian x-y grid ranging from (0,0) to (7,7), the sinsoidal positional encoding for each index is computed as:

$$
pe(x,y,\delta) = \begin{cases} 
pe(x,\delta) & \text{if $\delta < d_{model}/2$} \\\\
pe(y,\delta) & \text{if $\delta \geq d_{model}/2$}
\end{cases}
$$
where $\delta$ is the dimension in the latent space, and $pe(x,\delta)$ is given by the following.
$$
pe(x,\delta) = \begin{cases} 
\sin\frac{x}{10000^{\delta/d_{model}}} & \text{if $\delta$ is even} \\\\
\cos\frac{x}{10000^{\delta/d_{model}}} & \text{if $\delta$ is odd}
\end{cases}
$$
However this is not the only way implement positional encodings, so do not be surprised if a different method works better!

## Transformer architecture

### Encoder block
At a high level, the easiest way to explain the encoder architecture is to see the code

```python {linenos=true}
def forward(self, x):
    #Use built-in multihead attention, where x is the key, query and value
    attn, _ = self.MultiHeadAttention(x, x, x, need_weights=False)
    attn = self.dropout(attn)
    x = x.add(attn)
    x = self.LayerNorm1(x)
    mlp = self.MLP(x)
    x = x.add(mlp)
    x = self.LayerNorm2(x)
    return x
```
where the MLP is simply:
```python {linenos=true}
self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        ])
```
Other specifics of the model are taken from MaskGIT, including ```hidden_dim``` = 3072, ```dropout``` = 10% ```num_attention_heads``` = 8.

### Prediction block

Token prediction in MaskGIT is done by using a two-layer MLP, that takes in the $64 X d_{model}$ dimensional inputs and returns outputs of the same dimension. These inputs are then mapped to a coresponding codebook key through cosine similarity with the learned codebook vectors.  An alternative to this approach is to just have an MLP that predicts one hot encodings of each code, but this technique is not used in BERT.

### The full transformer

The full transformer simply consists of **6 encoder blocks**, followed by a token predicton of the encoded inputs.

## Model training

At every epoch each image is randomly masked following the masking scheme presented before. The images as well as their masked versions are passed into the model, with the goal of predicting the original image from the masked one.

However, it does not seem to work! This is what the training loss looks like

{{< figure src="/posts/Firstpost/TrainingLoss.png" width="400" caption="Training loss fails to converge" align="center">}}
