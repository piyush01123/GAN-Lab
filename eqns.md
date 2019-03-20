
<!--
python3 -m readme2tex  --output out.md eqns.md
-->

$$
log[P_\theta(x|z)] = E_z[log(P_\theta(x|z))] - D_{KL}[q_\phi(z|x) || P_\theta(z)]
$$

$$
\min_G \max_D V(D, G) = E_{x \sim X}log(D(x)) + E_{z \sim Z}log(1-D(G(z)))
$$

EQ1
$$
\max_D V(D, G) = E_{x \sim X}log(D(x)) + E_{z \sim Z}log(1-D(G(z)))
$$
EQ2
$$
\max_G V(D, G) =  E_{z \sim Z}log(D(G(z)))
$$

$$
\min_G \max_D V(D, G) = E_{x \sim X}log(D(x|y)) + E_{z \sim Z}log(1-D(G(z|y)))
$$



EQ1
$$
\min_D V(D, G) = \frac12E_{x \sim X}(D(x)-1)^2 + \frac12E_{z \sim Z}(D(G(z)))^2
$$
EQ2
$$
\min_G V(D, G) =  \frac12E_{z \sim Z}(D(G(z))-1)^2
$$


$$
\mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{GAN}(G, D_Y, X, Y) + \mathcal{L}_{GAN}(F, D_X, Y, X) + \lambda\mathcal{L}_{cyc}(G, F)
$$

$$
\mathcal{L}_{cyc}(G, F) = E_{x \sim X}||F(G(x))-x||_1 +  E_{y \sim Y}||G(F(y))-y||_1 
$$
