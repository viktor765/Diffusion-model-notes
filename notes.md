# Diffusion model notes

The idea of this document is to note down interesting articles/posts/videos, own insights, references, ideas, etc. around diffusion models.

This document will function as the main document for the notes. To begin with everything will land here, possibly branching of to multiple documents later.

# Mathematical background

## Literature

### Reference books

**SDE**
- Djehiche
- Oksendal
- Le Gall
- Revuz & Yor
- Karatzas & Shreve

**Optimal transport**
- Villani
- Santambrogio

**Statistics**
- Schervish

### Blog posts etc.

- Thiéry, [Reverse diffusions, Score & Tweedie](https://alexxthiery.github.io/posts/reverse_and_tweedie/reverse_and_tweedie.html)

# Methods

## Stochastic differential equations / score-based methods

Let $X_t$ be a diffusion process, i.e. a solution to the SDE

$$
\tag{SDE}
dX_t = f(t, X_t) dt + g(t) dW_t.
$$

Then its density $p_t$ follows the Fokker-Planck equation:

$$
\tag{FP}
\frac{\partial}{\partial t} p + \nabla \cdot (p f) = \frac{g^2}{2} \Delta p.
$$

Note that for $g = 0$, this reduces to the continuity equation $\dot p + \nabla \cdot (p f) = 0$. Noting that $\nabla p = p \nabla \log p$, and $g$ being space independent, we get:

$$
\tag{ODE}
\frac{\partial}{\partial t} p + \nabla \cdot (p (f - \frac{g^2}{2} \nabla \log p)) = 0,
$$

i.e. the continuity equation of the ODE 

$$\dot X_t = f - \frac{g^2}{2} \nabla \log p.$$

So the SDE and ODE above have the same density evolution $p(t,x)$.
Thus if one learns the (Stein) \textbf{score} $\nabla \log p : [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$, for example as a "score network" $s_\theta$, and one is able to generate samples from $p_1$, then one can generate samples from $p_0$ by running the ODE backward.

Another way to generate samples by going "backward" from $p_1$ is to use the \textbf{reverse SDE}, wrt (SDE). This result can be found in [BDO Anderson '82](https://www.sciencedirect.com/science/article/pii/0304414982900515). The reverse SDE is given by

$$
\tag{R-SDE}
dX_t = (f - g^2 \nabla \log p) dt + g d\overline{W}_t,
$$

where $\overline{W}_\cdot$ is a Brownian motion running backwards in time. This result is elucidated by considering Tweedie's formula, found in [Efron '11](https://efron.ckirby.su.domains/papers/2011TweediesFormula.pdf), which states that if $Y = X + Z$, where $Z \sim \mathcal{N}(0, \sigma^2)$, then

$$
\mathbb{E}[X | Y] = Y + \sigma^2 \nabla \log p_Y(Y).
$$

### Learning the score

TODO: Expand on this.

## KL-based

E.g. Ho et al. 2020, Sohl-Dickstein 2015. Will probably not expand on this.

## Flow matching

An alternative that entirely avoids SDE/KL formalism and directly learns a vector field (VF) $v(t, x)$ that flows a probability distribution to another. This, like (ODE) in the score-matching formalism, is a deterministic mapping and thus a type of normalizing flow, namely a \textbf{continuous normalizing flow} (CNF).

# General fluff

Every small but seemingly important thing that enters into diffusion models.

### Fourier features [(main paper)](https://arxiv.org/abs/2006.10739)

These have something to do with, but are not exactly [random Fourier features](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html). These are useful for positional encoding in images and help overcome spectral bias inherent in MLP.
Instead of feeding in $v = (x, y)$, one feeds in the "Fourier" features

$\gamma(v) = [a_1 \cos(2\pi b_1^T v), a_1 \sin(2\pi b_1^T v), \dots, a_m \cos(2\pi b_m^T v), a_m \sin(2\pi b_m^T v)]$.

Typically helps learn high frequency components.

### Self-attention

Often used at low-resolution parts of the score U-net. Helps different parts of the images see each other, to produce consistent output over distance.

### U-nets

Popular architecture for predicting functions on image data that have a similar (same resolution) domain and codomain, i.e. functions $\mathbb{R}^{C_{in} \times h \times w} \to \mathbb{R}^{C_{out} \times h \times w}$. Popular for image segmentation and medical applications. Also is appropriate for use as score network $s_\theta \approx \nabla_x \log p_t$.

### Time-embeddings

Used to effectively insert the time into the U-net. Alternatively use random Fourier features. 