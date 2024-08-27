---
tags:
  - VAE
  - probability
---
<!-- git add . && git commit -m 'vae' && git push  -->

* TOC 
{:toc}

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
&=\mathbb{E}_{z \sim q_\phi(z|x)} \log p_\theta(x|z) - \text{KL}(q_\phi(z|x) \Vert p(z))
\end{align}
$$


where (i) is Jensen's inequality, by maximizing the ELBO, the likelihood of 
$$
p_\theta (x)
$$
can also be improved
