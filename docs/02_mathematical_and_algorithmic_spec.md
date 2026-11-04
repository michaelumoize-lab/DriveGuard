# Deliverable 2: Mathematical & Algorithmic Specification (PDBP & Risk Engine)

**Project:** DriveGuard  
**Component:** Personalised Driver Behaviour Profile (PDBP), Feature Extraction & Adaptive Risk Engine  

---

## 1. Coordinate & Feature Extraction Formulation

### 1.1 3D-Aware Eye Aspect Ratio ($\text{EAR}$)
Using MediaPipe Face Landmarker (478 3D landmarks), for each eye $e \in \{L, R\}$ with 6 key contour points $\{p_1, p_2, p_3, p_4, p_5, p_6\}$:
- **Left Eye Indices:** `[33, 160, 158, 133, 153, 144]`
- **Right Eye Indices:** `[362, 385, 387, 263, 373, 380]`

$$
\text{EAR}_e = \frac{\|p_2 - p_6\|_2 + \|p_3 - p_5\|_2}{2 \|p_1 - p_4\|_2 + \epsilon}
$$

To handle partial face rotation (yaw $\theta_{\text{yaw}}$), combine left and right eyes using **yaw-weighted reliability weights** $w_L, w_R$:
- If $\theta_{\text{yaw}} > +15^\circ$ (turned right, left eye more visible to camera): $w_L = 0.75, w_R = 0.25$
- If $\theta_{\text{yaw}} < -15^\circ$ (turned left, right eye more visible): $w_L = 0.25, w_R = 0.75$
- Otherwise: $w_L = 0.50, w_R = 0.50$

$$
\text{EAR}_{\text{raw}}(t) = w_L \cdot \text{EAR}_L(t) + w_R \cdot \text{EAR}_R(t)
$$

### 1.2 Pose-Conditioned Normalization
Using MediaPipe's $4 \times 4$ facial transformation matrix, extract Euler angles $(\theta_{\text{pitch}}, \theta_{\text{yaw}}, \theta_{\text{roll}})$.  
During guided calibration, fit a second-order polynomial correction factor $g(\Delta \theta_{\text{pitch}}, \Delta \theta_{\text{yaw}})$ where $\Delta \theta = \theta - \theta_{\text{neutral}}$:

$$
\text{EAR}_{\text{norm}}(t) = \frac{\text{EAR}_{\text{raw}}(t)}{1 + \beta_1 \Delta \theta_{\text{pitch}} + \beta_2 |\Delta \theta_{\text{yaw}}| + \beta_3 \Delta \theta_{\text{pitch}}^2}
$$

This prevents downward dashboard glances or side-mirror checks from falsely depressing EAR.

---

## 2. Sub-Frame Blink Kinematics & Waveform Analysis

Given timestamped observations $(t_i, \text{EAR}_{\text{norm}}(t_i))$, smooth jitter using an adaptive Exponential Moving Average ($\alpha = 0.6$):

$$
s_i = \alpha \cdot \text{EAR}_{\text{norm}}(t_i) + (1 - \alpha) \cdot s_{i-1}
$$

For each detected blink event $k$ (where $s_i$ drops below the driver's dynamic closure threshold $\tau_{\text{close}} = \text{median}(B_{\text{EAR}}) - 2.5 \cdot \text{MAD}_{\text{scaled}}(B_{\text{EAR}})$):

1. **Interpolated Blink Duration ($D_k$):**
   Use linear/quadratic interpolation between frames bracketing $\tau_{\text{close}}$ to find exact entry time $t_{\text{enter}}$ and exit time $t_{\text{exit}}$:
   $$
   D_k = t_{\text{exit}} - t_{\text{enter}} \quad (\text{in ms})
   $$
2. **Closing Velocity ($v_{\text{close}, k}$) & Reopening Velocity ($v_{\text{open}, k}$):**
   Let $t_{\text{min}}$ be the timestamp of minimum eye openness $s_{\min}$ during blink $k$:
   $$
   v_{\text{close}, k} = \frac{s(t_{\text{enter}}) - s_{\min}}{t_{\text{min}} - t_{\text{enter}}}, \qquad v_{\text{open}, k} = \frac{s(t_{\text{exit}}) - s_{\min}}{t_{\text{exit}} - t_{\text{min}}}
   $$
3. **Reopening-to-Closing Velocity Ratio ($\text{RCR}_k$):**
   $$
   \text{RCR}_k = \frac{v_{\text{open}, k}}{v_{\text{close}, k} + \epsilon}
   $$
   *Physiological rationale:* Alert blinks have a crisp, fast reopening phase. Drowsy eyelids reopen sluggishly ($\text{RCR}_k$ drops significantly below personal baseline).

---

## 3. Confidence-Gated Fusion with `MobileNetV3-Small`

When the cascade gate activates on frame $t$, crop both eye regions ($64 \times 64$ RGB normalized), run `MobileNetV3-Small`, and obtain $P_{\text{closed}}(t) \in [0, 1]$.

Define the **Fused Eye Closure Signal** $C(t) \in [0, 1]$:

$$
C_{\text{geom}}(t) = \text{clip}\left( \frac{\text{median}(B_{\text{EAR}}) - s(t)}{\text{median}(B_{\text{EAR}}) - \tau_{\text{closed\_full}}}, 0, 1 \right)
$$

$$
C(t) = \begin{cases} 
\gamma \cdot P_{\text{closed}}(t) + (1 - \gamma) \cdot C_{\text{geom}}(t) & \text{if CNN gate triggered } (\gamma = 0.65) \\
0.7 \cdot C_{\text{geom}}(t) + 0.3 \cdot \text{Blendshape}_{\text{blink}}(t) & \text{otherwise}
\end{cases}
$$

### Rolling Time-Weighted PERCLOS ($W = 60\text{ s}$)
Instead of simple frame counting (which fails under variable FPS), compute **time-integrated PERCLOS** over the rolling window $[t - W, t]$:

$$
\text{PERCLOS}(t) = \frac{\sum_{i: t_i \in [t-W, t]} \mathbb{I}\big(C(t_i) \ge 0.80\big) \cdot \Delta t_i}{\sum_{i: t_i \in [t-W, t]} \Delta t_i} \times 100\%
$$

---

## 4. Personalised Driver Behaviour Profile (PDBP) & Robust Deviation Scores

### 4.1 Baseline Vector ($B_0$)
Collected during the 45-second guided calibration (requiring $\ge 85\%$ valid face tracking frames and $\ge 6$ natural blinks):

| Parameter | Baseline Center $\mu_m = \text{median}(B_m)$ | Robust Spread $\sigma_m = 1.4826 \cdot \text{MAD}(B_m) + \epsilon$ | Floor $\sigma_{\min}$ |
| :--- | :--- | :--- | :--- |
| **Open-Eye $\text{EAR}_{\text{norm}}$** | $\mu_{\text{EAR}}$ | $\sigma_{\text{EAR}}$ | $0.015$ |
| **Blink Duration ($D$)** | $\mu_{D}$ (ms) | $\sigma_{D}$ (ms) | $25\text{ ms}$ |
| **Reopening Ratio ($\text{RCR}$)** | $\mu_{\text{RCR}}$ | $\sigma_{\text{RCR}}$ | $0.08$ |
| **Blink Rate ($f_{\text{blink}}$)** | $\mu_{\text{freq}}$ (blinks/min) | $\sigma_{\text{freq}}$ | $3.0\text{ /min}$ |
| **Neutral Head Pitch ($\theta_{\text{pitch}}$)** | $\mu_{\text{pitch}}$ (deg) | $\sigma_{\text{pitch}}$ (deg) | $2.0^\circ$ |

*Note:* Enforcing a minimum scale floor $\sigma_{\min}$ is critical so that an unusually uniform 45-second calibration does not produce an artificially tiny denominator and cause hyper-sensitive false alarms.

### 4.2 Directional Robust $z$-Scores
Since drowsiness corresponds to **one-sided** physiological changes (lower EAR, longer blink duration, slower reopening velocity), define directional positive-fatigue scores $z_m^+ \ge 0$:

$$
z_{\text{EAR}}^+(t) = \max\left(0, \frac{\mu_{\text{EAR}} - \overline{s}_{5\text{s}}(t)}{\max(\sigma_{\text{EAR}}, \sigma_{\min})}\right)
$$

$$
z_{D}^+(t) = \max\left(0, \frac{\overline{D}_{\text{last 3}}(t) - \mu_D}{\max(\sigma_D, \sigma_{\min})}\right)
$$

$$
z_{\text{RCR}}^+(t) = \max\left(0, \frac{\mu_{\text{RCR}} - \overline{\text{RCR}}_{\text{last 3}}(t)}{\max(\sigma_{\text{RCR}}, \sigma_{\min})}\right)
$$

$$
z_{\text{PERCLOS}}^+(t) = \max\left(0, \frac{\text{PERCLOS}(t) - \text{PERCLOS}_{\text{base}}}{4.0\%}\right)
$$

### 4.3 Composite Fatigue Risk Score $R_t \in [0, 1]$
Combine the indicators via a bounded logistic risk function (with supporting yawn/nod boosts):

$$
u(t) = w_1 z_{\text{PERCLOS}}^+(t) + w_2 z_{D}^+(t) + w_3 z_{\text{RCR}}^+(t) + w_4 z_{\text{EAR}}^+(t) + w_5 I_{\text{nod}}(t) + w_6 I_{\text{yawn}}(t)
$$

$$
R_t = \frac{1}{1 + \exp\big(-k (u(t) - u_0)\big)}
$$

**Initial Default Weights (to be validated/learned via Logistic Regression / XGBoost in Experiment 3):**
- $w_1 (\text{PERCLOS}) = 0.30$
- $w_2 (\text{Blink Duration}) = 0.25$
- $w_3 (\text{Blink Reopening RCR}) = 0.20$
- $w_4 (\text{Sustained EAR drop}) = 0.15$
- $w_5 (\text{Head Nodding}) = 0.06$
- $w_6 (\text{Yawning Frequency}) = 0.04$

**Immediate Microsleep Override Rule (Safety Critical Bypass):**
Regardless of rolling window scores, if $C(t) \ge 0.85$ continuously for $\Delta t_{\text{closed}} \ge \max(1.2\text{ s}, 3.5 \times \mu_D)$ while head yaw is within road-facing bounds ($|\theta_{\text{yaw}}| < 25^\circ$), immediately set $R_t = 0.98$ (`CRITICAL` with reason `PROLONGED_EYE_CLOSURE_MICROSLEEP`).

---

## 5. Hysteresis State Machine & Recovery Rules

| Current State | Transition To | Condition Required |
| :--- | :--- | :--- |
| **ANY** | `UNAVAILABLE` | Face missing or confidence $< 0.50$ for $\ge 1.5\text{ s}$, OR WebSocket frame gap $> 2.0\text{ s}$. |
| `NORMAL` | `WARNING` | $R_t \ge 0.55$ sustained for $\ge 2.5\text{ s}$ (Persistence Gate). |
| `WARNING` | `CRITICAL` | $R_t \ge 0.80$ sustained for $\ge 1.5\text{ s}$, OR Immediate Microsleep Override, OR `WARNING` unacknowledged for $> 8.0\text{ s}$. |
| `CRITICAL` | `WARNING` | $R_t < 0.65$ sustained for $\ge 5.0\text{ s}$ (Hysteresis gap prevents oscillation). |
| `WARNING` | `NORMAL` | $R_t < 0.35$ sustained for $\ge 8.0\text{ s}$ (Cooldown/Recovery window). |
