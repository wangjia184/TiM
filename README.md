

## Transition Model

TiM introduces a unified framework that generalizes diffusion and flow-matching models through time-dependent interpolation.

### Forward Interpolant

The noisy state $x_t$ at time $t$ is constructed as a linear combination of clean data $x$ and noise $\epsilon \sim \mathcal{N}(0, I)$:
 
```math
\begin{align*}
x_t &= \alpha_t x + \sigma_t \epsilon \\
&\quad \Downarrow \\
\alpha_t x &= x_t - \sigma_t \epsilon &\text{(1)}
\end{align*}
```

This allows solving for the noise component $\epsilon$, which is used in various training objectives.

### Transition Model Extension

TiM extends this by learning $F_\theta(x_t, t, r)$ for **arbitrary-step transitions** from time $t$ to $r$.

where $\alpha_t$ and $\sigma_t$ are **time-dependent coefficients** (not constants) that define the interpolation path.

### Model Prediction

The neural network output $f_\theta(x_t, t)$ can be parameterized to predict a combination of data and noise:

```math
\begin{align*}
f_{\theta}(x_t, t) &= \hat{\alpha}_t x + \hat{\sigma}_t \epsilon \\
&\quad \Downarrow \\
\epsilon &= \frac{ f_{\theta}(x_t, t) - \hat{\alpha}_t x_t}{\hat{\sigma}_t} &\text{(2)}
\end{align*}
```


First we derive $\hat{x}$, the predicted clean data:

```math
\begin{align*}
\alpha_t x &= x_t - \sigma_t \epsilon &\text{Substitute Eq.(2) into Eq.(1)} \\

\alpha_t x &= x_t - \sigma_t \frac{ f_{\theta}(x_t, t) - \hat{\alpha}_t x_t}{\hat{\sigma}_t} \\

\alpha_t x &= x_t - \frac{\sigma_t}{\hat{\sigma}_t}f_{\theta}(x_t, t) + \frac{\sigma_t}{\hat{\sigma}_t} \hat{\alpha}_t x \\

\left( \alpha_t - \frac{\sigma_t}{\hat{\sigma}_t} \hat{\alpha}_t \right) x &= x_t - \frac{\sigma_t}{\hat{\sigma}_t}f_{\theta}(x_t, t) \\

\left( \frac{\hat{\sigma_t}}{\hat{\sigma}_t} \alpha_t - \frac{\sigma_t}{\hat{\sigma}_t} \hat{\alpha}_t \right) x &= x_t - \frac{\sigma_t}{\hat{\sigma}_t}f_{\theta}(x_t, t) \\

 \frac{\hat{\sigma_t} \alpha_t - \sigma_t \hat{\alpha_t}}{\hat{\sigma}_t}   x &= x_t - \frac{\sigma_t}{\hat{\sigma}_t}f_{\theta}(x_t, t) \\

\hat{x} &= \frac{\hat{\sigma}_t}{\hat{\sigma_t} \alpha_t - \sigma_t \hat{\alpha_t}} \left(x_t - \frac{\sigma_t}{\hat{\sigma}_t}f_{\theta}(x_t, t) \right)   \\

\hat{x} &= \frac{\hat{\sigma}_t x_t - \sigma_t  f_{\theta}(x_t, t)}{\hat{\sigma_t} \alpha_t - \sigma_t \hat{\alpha_t}}    \\
\end{align*}

```

Next, we deduct $\hat{\epsilon}$, the predicted noise:

```math
\begin{align*}
\epsilon &= \frac{ f_{\theta}(x_t, t) - \hat{\alpha}_t x}{\hat{\sigma}_t} &\text{Substitute Eq.(1) into Eq.(2)} \\
\epsilon &= \frac{ f_{\theta}(x_t, t) - \hat{\alpha}_t \frac{x_t - \sigma_t \epsilon}{\alpha_t} }{\hat{\sigma}_t} \\
\epsilon &= \frac{ f_{\theta}(x_t, t) }{\hat{\sigma}_t} - \frac{\hat{\alpha}_t}{\hat{\sigma}_t \alpha_t} x_t + \frac{\hat{\alpha}_t \sigma_t}{\hat{\sigma}_t \alpha_t} \epsilon \\
\left( 1 - \frac{\hat{\alpha}_t \sigma_t}{\hat{\sigma}_t \alpha_t} \right) \epsilon &= \frac{ f_{\theta}(x_t, t) }{\hat{\sigma}_t} - \frac{\hat{\alpha}_t}{\hat{\sigma}_t \alpha_t} x_t \\
\left( \hat{\sigma}_t \alpha_t - \hat{\alpha}_t \sigma_t \right) \epsilon &= \alpha_t f_{\theta}(x_t, t) - \hat{\alpha}_t x_t \\
\hat{\epsilon} &= \frac{ \alpha_t f_{\theta}(x_t, t) - \hat{\alpha}_t x_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t }
\end{align*}
```

We now introduce the target timestep `r` and express $x_r$ via the interpolant using the predicted clean data  $\hat{x}$ and predicted noise $\hat{ϵ}$ : 

```math
\begin{align*}
x_r &= \alpha_r \hat{x} + \sigma_r \hat{\epsilon} \\

x_r &= \alpha_r \left( \frac{\hat{\sigma}_t x_t - \sigma_t  f_{\theta}(x_t, t)}{\hat{\sigma_t} \alpha_t - \sigma_t \hat{\alpha_t}}  \right) + \sigma_r \left( \frac{ \alpha_t f_{\theta}(x_t, t) - \hat{\alpha}_t x_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) \\

x_r &= \frac{\hat{\sigma}_t \alpha_r x_t - \sigma_t \alpha_r f_{\theta}(x_t, t)}{\hat{\sigma_t} \alpha_t - \sigma_t \hat{\alpha}_t} + \frac{\sigma_r \alpha_t f_{\theta}(x_t, t) - \sigma_r \hat{\alpha}_t x_t}{\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t} \\

x_r &= \frac{ \left( \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t \right) x_t + \left( \sigma_r \alpha_t - \sigma_t \alpha_r \right) f_{\theta}(x_t, t) }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \\


x_r &= \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } x_t + \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } f_{\theta}(x_t, t) \\


\end{align*}
```

More generally, to learn the `t → r` transition, we condition the predictor on `r` and obtain:


```math
\begin{align*}
x_r &= \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } x_t + \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } f_{\theta}(x_t,t,r)   &\text{(3)}
\end{align*}
```

For simplicity, we define the coefficients:

```math
\begin{align*}

A_{t,r} &:= \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \\

B_{t,r} &:= \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \\

f_{\theta,t,r} &:= f_{\theta}(x_t,t,r)

\end{align*}
```

Then the transition can be written as:

```math
\begin{align*}
x_r &= A_{t,r} x_t + B_{t,r} f_{\theta,t,r} &\text{(4)}
\end{align*}
```

We now differentiate both sides of Eq.(4) with respect to t . Since $x_r$ is independent of `t` , its derivative is zero. Applying the product rule and chain rule yields:


```math
\begin{align*}
\frac{d A_{t,r}}{dt} &= 
\frac{\partial A_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt} \\
&&\text{(5)} \\ 
\frac{d B_{t,r}}{dt} &= 
\frac{\partial B_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
\frac{\partial B_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
\frac{\partial B_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
\frac{\partial B_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}

\end{align*}
```

Define $C_{t,r}:=\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t$. We compute each partial derivative in Eq. (5) explicitly:

```math
\begin{aligned}
&\begin{cases}
\displaystyle
\frac{\partial A_{t,r}}{\partial \alpha_t} 
&= \frac{\partial}{\partial \alpha_t} \left( \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ 0 \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) \cdot \hat{\sigma}_t }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= -\frac{ \hat{\sigma}_t (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= - \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial A_{t,r}}{\partial \sigma_t} 
&= \frac{\partial}{\partial \sigma_t} \left( \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ 0 \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) \cdot (-\hat{\alpha}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \hat{\alpha}_t (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} 
&= \frac{\partial}{\partial \hat{\alpha}_t} \left( \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ (-\sigma_r) \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) \cdot (-\sigma_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \hat{\sigma}_t (\sigma_t \alpha_r - \sigma_r \alpha_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= - \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} 
&= \frac{\partial}{\partial \hat{\sigma}_t} \left( \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ \alpha_r \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) \cdot \alpha_t }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \hat{\alpha}_t (\sigma_r \alpha_t - \sigma_t \alpha_r) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} }
\end{cases} 
\\
\\
&\begin{cases}
\displaystyle
\frac{\partial B_{t,r}}{\partial \alpha_t} 
&= \frac{\partial}{\partial \alpha_t} \left( \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ \sigma_r \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\sigma_r \alpha_t - \sigma_t \alpha_r) \cdot \hat{\sigma}_t }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \sigma_t (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \sigma_t \frac{ A_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial B_{t,r}}{\partial \sigma_t} 
&= \frac{\partial}{\partial \sigma_t} \left( \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ (-\alpha_r) \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\sigma_r \alpha_t - \sigma_t \alpha_r) \cdot (-\hat{\alpha}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \alpha_t (\sigma_r \hat{\alpha}_t - \alpha_r \hat{\sigma}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= - \alpha_t \frac{ A_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial B_{t,r}}{\partial \hat{\alpha}_t} 
&= \frac{\partial}{\partial \hat{\alpha}_t} \left( \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ 0 \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\sigma_r \alpha_t - \sigma_t \alpha_r) \cdot (-\sigma_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \sigma_t (\sigma_r \alpha_t - \sigma_t \alpha_r) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \sigma_t \frac{ B_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial B_{t,r}}{\partial \hat{\sigma}_t} 
&= \frac{\partial}{\partial \hat{\sigma}_t} \left( \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ 0 \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\sigma_r \alpha_t - \sigma_t \alpha_r) \cdot \alpha_t }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= -\frac{ \alpha_t (\sigma_r \alpha_t - \sigma_t \alpha_r) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= - \alpha_t \frac{ B_{t,r} }{ C_{t,r} }
\end{cases} 
\end{aligned}
```

Substituting Eq.(6) into Eq.(5) gives the total time derivatives:

```math
\begin{equation}
\begin{aligned}
\frac{d A_{t,r}}{dt} &= 
-\hat{\sigma}_t \frac{A_{t,r}}{C_{t,r}} \cdot \frac{d\alpha_t}{dt} &+ 
\hat{\alpha}_t \frac{A_{t,r}}{C_{t,r}} \cdot \frac{d\sigma_t}{dt} &- 
\hat{\sigma}_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\alpha}_t}{dt} &+ 
\hat{\alpha}_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\sigma}_t}{dt} \\[0.6em]
\frac{d B_{t,r}}{dt} &= 
\sigma_t \frac{A_{t,r}}{C_{t,r}} \cdot \frac{d\alpha_t}{dt} &- 
\alpha_t \frac{A_{t,r}}{C_{t,r}} \cdot \frac{d\sigma_t}{dt} &+ 
\sigma_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\alpha}_t}{dt} &- 
\alpha_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\sigma}_t}{dt}
\end{aligned}
\tag{7}
\end{equation}
```

From the interpolant equation $x_t=\alpha_t x + \sigma_t \epsilon$, we compute its time derivative:

```math
\begin{align*}
x_t &= \alpha_t x + \sigma_t \epsilon \\
&\quad \Downarrow \\
\frac{dx_t}{dt} &= \frac{d \alpha_t}{dt} x +  \frac{d \sigma_t}{dt} \epsilon \tag{8}
\end{align*}
```

We now differentiate Eq.(4) with respect to `t` :

```math
\begin{align*}
A_{t,r} x_t + B_{t,r} f_{\theta,t,r} &= x_r \\

\frac{d}{dt} \left( A_{t,r} x_t + B_{t,r} f_{\theta,t,r} \right) &= \frac{d x_r}{dt} \\

\frac{d}{dt} \left( A_{t,r} x_t \right) + \frac{d}{dt} \left( B_{t,r} f_{\theta,t,r} \right) &= 0  \\
 
\left[ \frac{d A_{t,r}}{dt} x_t + \frac{d x_t}{dt} A_{t,r} \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 \\

\left[ \frac{d A_{t,r}}{dt} \left(\alpha_t x + \sigma_t \epsilon \right) + \frac{d x_t}{dt} A_{t,r} \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right] 
&= 0 &\text{\because x_t = \alpha_t x + \sigma_t \epsilon} \\

\left[ \frac{d A_{t,r}}{dt} \left(\alpha_t x + \sigma_t \epsilon \right) + \left( \frac{d \alpha_t}{dt} x +  \frac{d \sigma_t}{dt} \epsilon \right) A_{t,r} \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right] 
&= 0 &\text{\because Eq.(8)} \\

\left[ \frac{d A_{t,r}}{dt} \alpha_t x + \frac{d A_{t,r}}{dt} \sigma_t \epsilon  +A_{t,r} \frac{d \alpha_t}{dt} x +  A_{t,r} \frac{d \sigma_t}{dt} \epsilon   \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\

\left[ \left( A_{t,r} \frac{d \alpha_t}{dt} + \alpha_t \frac{d A_{t,r}}{dt}    \right) x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\


\left[ \left( A_{t,r} \frac{d \alpha_t}{dt} + \alpha_t 
\left(
\frac{\partial A_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}
\right)    \right) x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  &\text{\because Eq.(5)} \\


\left[ \left( 
  \left( A_{t,r}  + \alpha_t \frac{\partial A_{t,r}}{\partial \alpha_t}  \right) \frac{d \alpha_t}{dt} +
 \alpha_t \frac{\partial A_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
 \alpha_t \frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \alpha_t \frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\


\left[ \left( 
  \left( A_{t,r}  - \alpha_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} }  \right) \frac{d \alpha_t}{dt} +
 \alpha_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} -
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \alpha_t  \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  &\text{\because Eq.(6)} \\

\left[ \left( 
  \left( A_{t,r} \frac{\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t}{C_{t,r}} - \alpha_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} }  \right) \frac{d \alpha_t}{dt} +
 \alpha_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} -
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \alpha_t  \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  &\text{\because C_{t,r}:=\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t} \\

\left[ \left( 
   - \sigma_t  \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} }  \cdot \frac{d \alpha_t}{dt} +
 \alpha_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} -
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \alpha_t  \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\


\left[ \left( 
   - \sigma_t  \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} }  \cdot \frac{d \alpha_t}{dt} +
 \alpha_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} -
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \alpha_t  \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt} +
 \hat{\alpha}_t \sigma_t \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}-
 \hat{\alpha}_t \sigma_t \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\

\left[ \left( 
  - \hat{\alpha}_t
  \left(
    \sigma_t   \frac{ A_{t,r} }{ C_{t,r} }  \cdot \frac{d \alpha_t}{dt} 
    - \alpha_t  \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} 
    + \sigma_t \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}
    - \alpha_t  \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
  \right)
 -
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} +
 \hat{\alpha}_t \sigma_t \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
-
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} +
 \hat{\alpha}_t \sigma_t \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{\because Eq.(7)}  \\

\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - \left(
    \alpha_t  \hat{\sigma}_t - \hat{\alpha}_t \sigma_t
  \right)
   \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{\because C_{t,r}:=\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t} \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t 
\left(
\frac{\partial A_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}
\right)     \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{\because Eq.(5)} \\



\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} +
 \sigma_t \frac{\partial A_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
 \sigma_t \frac{\partial A_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
 \sigma_t \frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \sigma_t \frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
 \sigma_t \frac{\partial A_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
\left(  A_{t,r} + \sigma_t \frac{\partial A_{t,r}}{\partial \sigma_t} \right) \cdot \frac{d \sigma_t}{dt} + 
 \sigma_t \frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \sigma_t \frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\
 
\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \sigma_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \alpha_t}{dt} + 
\left(  A_{t,r} + \sigma_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \right) \cdot \frac{d \sigma_t}{dt} -
 \sigma_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \sigma_t \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{\because Eq.(6)} \\
 
\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \sigma_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \alpha_t}{dt} + 
\left(  A_{t,r} \frac{\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t}{C_{t,r}} + \sigma_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \right) \cdot \frac{d \sigma_t}{dt} -
 \sigma_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \sigma_t \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{\because C_{t,r}:=\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t} \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \sigma_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \alpha_t}{dt} + 
 \hat{\sigma}_t\alpha_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} -
 \sigma_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \sigma_t \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
- \left( 
 \sigma_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \alpha_t}{dt} - 
 \hat{\sigma}_t\alpha_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} +
 \sigma_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} - 
 \sigma_t \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
 - \hat{\sigma}_t \alpha_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\sigma}_t}{dt}
+ \hat{\sigma}_t \alpha_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\
 
\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \hat{\sigma}_t \left( 
 \sigma_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \alpha_t}{dt} - 
 \alpha_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} +
 \sigma_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} -
 \alpha_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\sigma}_t}{dt}
 \right)
 -
 \left(  
  \left( \sigma_t \hat{\alpha}_t - \hat{\sigma}_t \alpha_t \right)
  \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}

      \right) \right) \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\



\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \hat{\sigma}_t \frac{d B_{t,r}}{dt}
 -
 \left(  
  \left( \sigma_t \hat{\alpha}_t - \hat{\sigma}_t \alpha_t \right)
  \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}

      \right) \right) \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{\because Eq.(7)}\\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \hat{\sigma}_t \frac{d B_{t,r}}{dt}
 -
    B_{t,r}  \cdot \frac{d \hat{\sigma}_t}{dt}
 \right) \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{\because C_{t,r}:=\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t} \\

 
  - \hat{\alpha}_t  x 
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r}  x  \cdot \frac{ d \hat{\alpha}_t} {dt}
- \hat{\sigma}_t \epsilon \frac{d B_{t,r}}{dt}
 -
    B_{t,r} \epsilon \cdot \frac{d \hat{\sigma}_t}{dt}
   + 
\frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} 
&= 0 \\

\hat{\alpha}_t  x 
  \frac{d B_{t,r}}{dt}
+ B_{t,r}  x  \cdot \frac{ d \hat{\alpha}_t} {dt}
+ \hat{\sigma}_t \epsilon \frac{d B_{t,r}}{dt}
+  B_{t,r} \epsilon \cdot \frac{d \hat{\sigma}_t}{dt}
-\frac{d B_{t,r}}{dt}  f_{\theta,t,r} 
- \frac{d f_{\theta,t,r}}{dt} B_{t,r} 
&= 0 \\

\left(\hat{\alpha}_t  x + \hat{\sigma}_t \epsilon -  f_{\theta,t,r}  \right)
 \frac{d B_{t,r}}{dt} 
+ B_{t,r}
\left(
x  \cdot \frac{ d \hat{\alpha}_t} {dt}
+   \epsilon \cdot \frac{d \hat{\sigma}_t}{dt}
- \frac{d f_{\theta,t,r}}{dt}  
\right)
&= 0 \\


\left(\hat{\alpha}_t  x + \hat{\sigma}_t \epsilon -  f_{\theta,t,r}  \right)
 \frac{d B_{t,r}}{dt} 
+ B_{t,r}
\left(
 \frac{ d \hat{\alpha}_t x } {dt}
+    \frac{d \hat{\sigma}_t  \epsilon}{dt}
- \frac{d f_{\theta,t,r}}{dt}  
\right)
&= 0 \\

\left(\hat{\alpha}_t  x + \hat{\sigma}_t \epsilon -  f_{\theta,t,r}  \right)
 \frac{d B_{t,r}}{dt} 
+ B_{t,r}
 \frac{ d\left(\hat{\alpha}_t  x + \hat{\sigma}_t \epsilon -  f_{\theta,t,r}  \right) } {dt}
&= 0 &\text{(9)} \\

\frac{d}{dt}
\left[
  B_{t,r} \cdot \left(\hat{\alpha}_t  x + \hat{\sigma}_t \epsilon -  f_{\theta,t,r}  \right)
\right]
&= 0 &\text{(10)} \\

\end{align*}
```

 
