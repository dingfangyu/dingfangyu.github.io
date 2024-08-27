---
title: 'Transformer Gradients'
tags:
  - LLM
  - training
  - fine-tuning
  - linear algebra
---


This post summarizes the matrix gradients in the training / LoRA Fine-Tuning of Transformer

* TOC 
{:toc}

<!-- git add . && git commit -m 'grad' && git push  -->

## Preliminary

### trace

basic properties:


$$
tr(A+B) = tr(A) + tr(B),\quad tr(AB) = \sum_{i,j} A_{ij}B_{ij}
$$


and trace is a symmetric operation:

$$
tr(AB) = tr(BA)
$$


### matrix gradients 

trace for matrix gradients:


$$
d\mathcal{L} = tr(\frac{\partial \mathcal{L}}{\partial X}^\top dX) = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial X_{ij}}X_{ij}
$$




where 
$$
\mathcal{L}
$$
is a scalar loss, and 
$$
\frac{\partial \mathcal{L}}{\partial X}
$$
is the gradient of matrix X (a.k.a. X.grad), which has a same data shape as X



## FFN

the basic components of FFN computation: 1) linear transformation, 2) elementwise function (e.g. ReLU, 
$$
\odot
$$
)

### linear

given a linear transformation


$$
Y = XW
$$


where 
$$
X, Y
$$
are input data and output data, and
$$
W
$$
is the weight matrix of a linear module



given the backpropagated gradient of Y (i.e. Y.grad or 
$$
\frac{\partial \mathcal{L}}{\partial Y}
$$
), we can derive the gradients of X and W using the trace calculations:


$$
\begin{align}
d\mathcal{L} &= tr(\frac{\partial \mathcal{L}}{\partial Y}^\top dY) \\
&= tr(\frac{\partial \mathcal{L}}{\partial Y}^\top (XdW + (dX) W)) \\
&= tr(\frac{\partial \mathcal{L}}{\partial Y}^\top XdW ) + tr(W\frac{\partial \mathcal{L}}{\partial Y}^\top dX)  \\
\end{align}
$$


therefore,


$$
\frac{\partial \mathcal{L}}{\partial W}^\top = \frac{\partial \mathcal{L}}{\partial Y}^\top X,\quad \frac{\partial \mathcal{L}}{\partial X}^\top = W\frac{\partial \mathcal{L}}{\partial Y}^\top
$$


### more than linear: LoRA 

LoRA is a prevalent method for parameter-efficient fine-tuning (PEFT), whose forward calculation is:


$$
Y = X (W +AB)
$$


where W is the base model weights, and A, B are low rank adapters



the gradients can be obtained in a similar manner:


$$
\begin{align}
d\mathcal{L} &= tr(\frac{\partial \mathcal{L}}{\partial Y}^\top dY) \\
&= tr(\frac{\partial \mathcal{L}}{\partial Y}^\top d(X(W+AB))) \\
&= tr(\frac{\partial \mathcal{L}}{\partial Y}^\top (X(dW + AdB + (dA)B) + (dX) (W+AB))) \\
&= tr(\frac{\partial \mathcal{L}}{\partial Y}^\top XdW ) + tr(\frac{\partial \mathcal{L}}{\partial Y}^\top XAdB ) +tr(B\frac{\partial \mathcal{L}}{\partial Y}^\top XdA ) + tr((W + AB)\frac{\partial \mathcal{L}}{\partial Y}^\top dX)  \\
\end{align}
$$


therefore,


$$
\frac{\partial \mathcal{L}}{\partial W}^\top = \frac{\partial \mathcal{L}}{\partial Y}^\top X,\quad 
\frac{\partial \mathcal{L}}{\partial B}^\top = \frac{\partial \mathcal{L}}{\partial Y}^\top XA,\quad 
\frac{\partial \mathcal{L}}{\partial A}^\top = B\frac{\partial \mathcal{L}}{\partial Y}^\top X,\quad 
\frac{\partial \mathcal{L}}{\partial X}^\top = (W + AB)\frac{\partial \mathcal{L}}{\partial Y}^\top
$$


where the first item, the transpose of the gradient of the base weights 
$$
\frac{\partial \mathcal{L}}{\partial W}^\top
$$
should be 0 if we use LoRA 



### elementwise function

$$
Y = \sigma(X)
$$

then


$$
d\mathcal{L} = tr(\frac{\partial \mathcal{L}}{\partial Y}^\top dY) = tr(\frac{\partial \mathcal{L}}{\partial Y}^\top [\sigma^{'}(X) \odot dX]) = tr([\frac{\partial \mathcal{L}}{\partial Y}^\top \odot \sigma^{'}(X)]  dX)
$$


## SA

### softmax



except for the linear and elementwise transformations, what special in self-attention (SA) block is the softmax operation in attention, consider one attention head

$$
O = softmax(QK^\top)V
$$


the core computation (i.e. other than linear transformations) is a function of a vector s:


$$
p = softmax(s) = exp(s) \oslash [exp(s) 11^T]
$$




we can adopt the former rules, and get:


$$
ds = (diag(p) - pp^\top) dp
$$
