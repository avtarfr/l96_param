# Proposal

Here we shall use the Lorenz-96 toy model to try and see how close we can come to real life observations.

## Setup (1 Dimensional Ring)

We begin by setting up a latitude ring at the equator with K grid points. The Lorenz-96 model is defined by the following set of ordinary differential equations:

$$
\frac{dX_k}{dt} = -X_{k-1} (X_{k-2} - X_{k+1}) - X_k + F - \frac{hc}{b}\sum_{j=1}^{J}Y_{j, k}
$$

$$
\frac{dY_{j,k}}{dt} = -cbY_{j+1, k}(Y_{j+2,k}-Y_{j-1,k})-cY_{j,k}+\frac{hc}{b}X_k
$$

The second equation doesn't really matter for us that much other than to run the model.

The first thing we begin modelling is the "wind" vector field over the latitude ring. Defining the ACW direction as $(+k)$ and CW as $(-k)$, we say that:

* $X_k > 0 \implies (+k) \text{ wind direction }$
* $X_k < 0 \implies (-k) \text { wind direction }$
* $|X_k|$ = Magnitude of Wind Speed

This wind field is forced by some F. The wind may contain moisture and heat and other constituents (carriers of heat in general). 

Next, we define our temperature tendency as:

$$
C\frac{dT_k}{dt} = R_k - D_k \qquad R_k = ISR-OLR \qquad D_k = \Phi_{k + \frac{1}{2}} - \Phi_{k - \frac{1}{2}} \qquad \Phi_{k+ \frac{1}{2}} = \Phi(x_k + \frac{\Delta x}{2}) = \rho H c_p X_k T_k = QX_kT_{up} \qquad T_{up} = \begin{cases} \begin{align*} &T_k \quad & X_{k+\frac{1}{2}} > 0 \\
  &T_{k+1} \quad & X_{k+\frac{1}{2}} < 0 \end{align*}\end{cases}
$$

Applying radiative laws to this system (1D Atmosphere):

$$
ISR = S_0(1-\alpha)\max(0, \cos(\theta_k, \omega t)), \quad OLR = \varepsilon \sigma T_k^4
$$

$$
T_E = \left(\frac{1}{K}\sum_{k=1}^{K}T_k^4\right)^\frac{1}{4}
$$

**Assumption 1:** $\varepsilon = 1$ for surface, i.e. our latitude ring is a perfect blackbody.

$$
C\frac{dT_k}{dt} = S_0(1-\alpha)\max(0, \cos(\theta_k, \omega t)) - \sigma T_k^4 - \frac{\partial \phi}{\partial x}
$$

$$
X_{k + \frac{1}{2}} = \frac{1}{2}(X_k + X_{k+1}), \, X_{k-\frac{1}{2}} = \frac{1}{2}(X_k + X_{k-1})
$$

$$
X_{k+\frac{1}{2}}-X_{k-\frac{1}{2}} = \frac{1}{2}(X_{k+1} - X_{k-1})
$$

$$
\frac{\partial\Phi}{\partial x} = \frac{\partial{QXT}}{\partial x} = Q(X\frac{\partial T}{\partial x} + T \frac{\partial X}{\partial x}) = Q\left(X_k\frac{T_{k+1}-T{k-1}}{2\Delta x} + T_k\frac{X_{k+1} - X_{k-1}}{2\Delta x}\right)
$$


$$
C\frac{dT_k}{dt} = S_0(1-\alpha)\max(0, \cos(\theta_k, \omega t)) - \sigma T_k^4 - Q\left(X_k\frac{T_{k+1}-T_{k-1}}{2\Delta x} + T_k\frac{X_{k+1} - X_{k-1}}{2\Delta x}\right)
$$

This gives us ordered pairs of $(X_i, T_i) \to (\dot{X_i}, \dot{T_i})$ and the collection of $\textbf{T} = (T_1, \dots, T_K)$ gives the temperature at each $k$ point.

Now, suppose we start with initial values of a set of values for $\textbf{X} = (X_1, \dots, X_K)$ and $\textbf{T}$.  We run the tendencies for a while, discarding the initial values and using the final returning value as the updated value. How does our simulation look like? Let's see!
