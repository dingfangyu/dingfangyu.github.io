---
title: "VAE Framework and Diffusion Model"
tags:
  - VAE
  - ELBO
  - probability
  - Diffusion
---

The mathematic definition of the framework of Variational Autoencoder, and an instance of Diffusion model (only the content of VAE)

* TOC 
{:toc}

<!-- git add . && git commit -m 'vae' && git push  -->

## VAE 

### decoder

A VAE models the joint distribution of data 
$$
x
$$
 and latent 
$$
z
$$
, by introducing a prior distribution 
$$
p(z)
$$
 (e.g. Standard Gaussian, in vanilla VAE), and a generative model 
$$
p_\theta(x | z) 
$$
(a.k.a. likelihood / decoder) to reconstruct data 
$$
x
$$
 from a latent variable sample 
$$
z
$$
:


$$
p_\theta(x, z) = p_\theta(x | z) p(z)
$$

### encoder 

to enable the training of VAE, we need an encoder (a.k.a. posterior) to encode data
$$
x
$$
into 
$$
z
$$
in latent space:


$$
q_\phi(z|x)
$$


### ELBO

the learning objective of VAE is the evidence lower bound (ELBO):


$$
\begin{align}
\log p_\theta(x) &= \log \int p_\theta(x, z) dz \\
&= \log \int p_\theta(x|z)p(z) dz \\
&= \log \int \frac{q_\phi(z|x)p_\theta(x|z)p(z)}{q_\phi(z|x)} dz \\
&\overset{(i)}{\ge} \int q_\phi(z|x) \log \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)} dz \\
&= \int q_\phi(z|x) \log p_\theta(x|z) dz + \int q_\phi(z|x) \log \frac{p(z)}{q_\phi(z|x)} dz  \\
&=\mathbb{E}_{z \sim q_\phi(z|x)} \log p_\theta(x|z) - \text{KL}(q_\phi(z|x) \Vert p(z))\\
&=\text{ELBO}
\end{align}
$$


where (i) is Jensen's inequality, by maximizing the ELBO, the likelihood of 
$$
p_\theta (x)
$$
can also be optimized





## Diffusion

### decoder

the decoder (a.k.a. generative submodel) of Diffusion is modeled as:


$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)
$$


where 
$$
x_T
$$
is noise, and 
$$
x_0
$$
is reconstructed data



### encoder

the encoder submodel of Diffusion is a fixed (w\o learnable parameters) diffusion process:


$$
q(x_{0:T}) = q(x_0) \prod_{t=1}^T q(x_t | x_{t-1})
$$


### ELBO


$$
\begin{align}
\log p_\theta(x_0) &= \log \int p_\theta(x_{0:T}) dx_{1:T}  \\
&= \log \int \frac{q(x_{1:T}|x_0) p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} dx_{1:T}  \\
&\ge \int q(x_{1:T}|x_0) \log  \frac{ p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} dx_{1:T} \\
&= \int q(x_{1:T}|x_0) \log  \frac{p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)}{\prod_{t=1}^T q(x_t | x_{t-1})} dx_{1:T} \\
&= \int q(x_{1:T}|x_0) [\log p(x_T) + \sum_{t=1}^T \log \frac{  p_\theta(x_{t-1} | x_t)}{q(x_t | x_{t-1})}] dx_{1:T} \\
&=\text{ELBO}
\end{align}
$$


due to the Markov property
$$
q(x_t | x_{t-1}) = q(x_t | x_{t-1}, x_0)
$$
we have:


$$
\begin{align}
\text{ELBO} &= \int q(x_{1:T}|x_0) [\log p(x_T) + \sum_{t=1}^T \log \frac{  p_\theta(x_{t-1} | x_t)}{q(x_t | x_{t-1}, x_0)}] dx_{1:T} \\
\end{align}
$$


instead of 
$$
q(x_t | x_{t-1}, x_0)
$$
, we want some distribution of 
$$
q(x_{t-1} | x_t, x_0)
$$
to align with the role of 
$$
p_\theta(x_{t-1}|x_t)
$$




by several derivations of Bayes' Theorem, we have:


$$
\begin{align}
\text{ELBO}&= \int q(x_{1:T}|x_0) [\log p(x_T) + \sum_{t=1}^T \log \frac{  p_\theta(x_{t-1} | x_t)  q(x_{t-1}, x_0)}{q(x_t , x_{t-1}, x_0) }] dx_{1:T} \\
&= \int q(x_{1:T}|x_0) [\log p(x_T) + \sum_{t=1}^T \log \frac{  p_\theta(x_{t-1} | x_t)  q(x_{t-1}, x_0) q(x_{t}, x_0)}{q(x_t , x_{t-1}, x_0)q(x_{t}, x_0) }] dx_{1:T} \\
&= \int q(x_{1:T}|x_0) [\log p(x_T) + \sum_{t=1}^T \log \frac{  p_\theta(x_{t-1} | x_t)  q(x_{t-1}| x_0)q(x_0) }{q( x_{t-1} | x_t , x_0)q(x_{t}| x_0) q(x_0) }] dx_{1:T} \\
&= \int q(x_{1:T}|x_0) [\log p(x_T) + \sum_{t=1}^T \log \frac{  p_\theta(x_{t-1} | x_t)  q(x_{t-1}| x_0) }{q( x_{t-1} | x_t , x_0)q(x_{t}| x_0) }] dx_{1:T} \\
\end{align}
$$


then 


$$
\begin{align}
\text{ELBO}
&= \int q(x_{1:T}|x_0) [\log p(x_T) + \sum_{t=1}^T \log \frac{  p_\theta(x_{t-1} | x_t)   }{q( x_{t-1} | x_t , x_0) } + \sum_{t=1}^T \log \frac{   q(x_{t-1}| x_0) }{q(x_{t}| x_0) }] dx_{1:T} \\
&= \int q(x_{1:T}|x_0) [
\log p(x_T) + 
\sum_{t=1}^T \log \frac{  p_\theta(x_{t-1} | x_t)   }{q( x_{t-1} | x_t , x_0) } + 
\log \frac{   q(x_{0}| x_0) }{q(x_{T}| x_0) }] dx_{1:T} \\
&= \int q(x_{1:T}|x_0) [
\log p(x_T) 

+ 
\sum_{t=2}^T \log \frac{  p_\theta(x_{t-1} | x_t)   }{q( x_{t-1} | x_t , x_0) } + 
 \log \frac{  p_\theta(x_{0} | x_1)   }{q( x_{0} | x_1 , x_0) } + 
\log \frac{   q(x_{0}| x_0) }{q(x_{T}| x_0) }] dx_{1:T} \\

&= \int q(x_{1:T}|x_0) [
\log p(x_T) + 
\sum_{t=2}^T \log \frac{  p_\theta(x_{t-1} | x_t)   }{q( x_{t-1} | x_t , x_0) } + 
 \log \frac{  p_\theta(x_{0} | x_1)   }{1} + 
\log \frac{   1}{q(x_{T}| x_0) }] dx_{1:T} \\

&= \int q(x_{1:T}|x_0) [
\log \frac{   p(x_T)}{q(x_{T}| x_0) } + 
\sum_{t=2}^T \log \frac{  p_\theta(x_{t-1} | x_t)   }{q( x_{t-1} | x_t , x_0) } + 
 \log  p_\theta(x_{0} | x_1)
] dx_{1:T} \\

\end{align}
$$


then 


$$
\begin{align}
\text{ELBO}
&= 
\mathbb{E}_{q(x_{1:T}|x_0)} [ -\text{KL}( q(x_{T}| x_0)  \Vert p(x_T) ) - \sum_{t=2}^T \text{KL}( q( x_{t-1} | x_t , x_0) \Vert p_\theta(x_{t-1} | x_t)  ) + \log  p_\theta(x_{0} | x_1)]

\end{align}
$$
