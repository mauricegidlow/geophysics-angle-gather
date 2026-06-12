import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

st.set_page_config(
    page_title="Synthetic Angle Gather",
    page_icon="🦏",
    layout="wide"
)

st.markdown("""
<style>

/* Space above logo and title */
div.block-container {
    padding-top: 5rem;
    padding-bottom: 1rem;
}

/* Compact layout */
[data-testid="stVerticalBlock"] {
    gap: 0rem;
}

div[data-testid="stNumberInput"] {
    margin-bottom: -12px;
}

div[data-testid="stSelectbox"] {
    margin-bottom: -12px;
}

h1, h2, h3 {
    margin-top: 0.2rem;
    margin-bottom: 0.2rem;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# Header with logo
# -----------------------------

col_logo, col_title = st.columns([1, 6])

with col_logo:
    if os.path.exists("logo.jpg"):
        logo = Image.open("logo.jpg")
        st.image(logo, width=130)
    elif os.path.exists("logo.png"):
        logo = Image.open("logo.png")
        st.image(logo, width=130)

with col_title:
    st.title("Synthetic Angle Gather")
    st.caption("AVO Modelling and Synthetic Angle Gather Generation")


# -----------------------------
# Functions
# -----------------------------

def ricker_wavelet(freq, dt, length=0.2):
    t = np.arange(-length / 2, length / 2, dt)
    w = (1 - 2 * np.pi**2 * freq**2 * t**2) * np.exp(
        -np.pi**2 * freq**2 * t**2
    )
    return t, w


def rotate_phase(wavelet, phase_deg):
    phase = np.deg2rad(phase_deg)
    spectrum = np.fft.fft(wavelet)
    freqs = np.fft.fftfreq(len(wavelet))
    phase_shift = np.exp(1j * phase * np.sign(freqs))
    return np.real(np.fft.ifft(spectrum * phase_shift))


def acoustic_rc(vp1, rho1, vp2, rho2):
    ai1 = vp1 * rho1
    ai2 = vp2 * rho2
    return (ai2 - ai1) / (ai2 + ai1)


def aki_richards(vp1, vs1, rho1, vp2, vs2, rho2, theta):
    vp_avg = (vp1 + vp2) / 2
    vs_avg = (vs1 + vs2) / 2
    rho_avg = (rho1 + rho2) / 2

    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1

    term1 = 0.5 * (dvp / vp_avg + drho / rho_avg)

    term2 = (
        0.5 * dvp / vp_avg
        - 4 * (vs_avg**2 / vp_avg**2)
        * (drho / rho_avg + 2 * dvs / vs_avg)
    ) * np.sin(theta) ** 2

    term3 = 0.5 * (dvp / vp_avg) * (
        np.tan(theta) ** 2 - np.sin(theta) ** 2
    )

    return term1 + term2 + term3


def depth_to_twt(depth, thicknesses, velocities):
    twt = 0.0
    remaining = depth

    for h, v in zip(thicknesses, velocities):
        if remaining <= 0:
            break

        dz = min(remaining, h)
        twt += 2 * dz / v
        remaining -= dz

    return twt


def build_twt_log(thicknesses, velocities, values):
    depth_total = np.sum(thicknesses)
    depth_axis = np.linspace(0, depth_total, 2000)

    twt_axis = np.array([
        depth_to_twt(d, thicknesses, velocities)
        for d in depth_axis
    ])

    value_log = np.zeros_like(depth_axis)
    top = 0.0

    for h, val in zip(thicknesses, values):
        base = top + h
        mask = (depth_axis >= top) & (depth_axis < base)
        value_log[mask] = val
        top = base

    value_log[depth_axis >= top] = values[-1]

    return twt_axis, depth_axis, value_log


def plot_wiggles(ax, gather, time, angles, angle_inc, scale):
    for j, angle in enumerate(angles):
        trace = gather[:, j]
        max_amp = np.max(np.abs(trace))

        if max_amp > 0:
            trace = trace / max_amp

        x = angle + trace * angle_inc * 0.5 * scale

        ax.plot(x, time, color="black", linewidth=0.8)

        ax.fill_betweenx(
            time,
            angle,
            x,
            where=(x > angle),
            color="black",
            alpha=1.0,
        )

    ax.invert_yaxis()
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("TWT (s)")
    ax.grid(True)


# -----------------------------
# Sidebar options
# -----------------------------

with st.sidebar:
    st.header("Options")

    avo_method = st.selectbox("AVO", ["Aki-Richards", "Acoustic"])
    display_mode = st.selectbox("Display", ["Wiggle", "Colour", "Both"])

    dt = st.number_input("dt (s)", value=0.001, format="%.4f")
    freq = st.number_input("Freq (Hz)", value=25.0)
    phase_deg = st.selectbox("Phase", [0, 45, 90, -45, -90, 180])

    tmax = st.number_input("Length (s)", value=1.0)

    min_angle = st.number_input("Min angle", value=0.0)
    max_angle = st.number_input("Max angle", value=60.0)
    angle_inc = st.number_input("Angle inc", value=5.0)

    wiggle_scale = st.slider("Wiggle scale", 0.1, 5.0, 1.0)


# -----------------------------
# Lithology properties
# -----------------------------

st.subheader("Lithologies")

lith_names = ["Sand", "Shale", "Limestone", "X", "Y"]

default_props = {
    "Sand": [2800.0, 1400.0, 2.20],
    "Shale": [3200.0, 1600.0, 2.40],
    "Limestone": [5000.0, 2800.0, 2.60],
    "X": [3500.0, 1800.0, 2.30],
    "Y": [4200.0, 2300.0, 2.50],
}

c0, c1, c2, c3 = st.columns([1.0, 0.8, 0.8, 0.8])
c0.markdown("**Lith**")
c1.markdown("**Vp**")
c2.markdown("**Vs**")
c3.markdown("**Rho**")

lithology = {}

for lith in lith_names:
    c0, c1, c2, c3 = st.columns([1.0, 0.8, 0.8, 0.8])

    c0.write(lith)

    vp = c1.number_input(
        "",
        value=default_props[lith][0],
        step=50.0,
        key=f"{lith}_vp"
    )

    vs = c2.number_input(
        "",
        value=default_props[lith][1],
        step=50.0,
        key=f"{lith}_vs"
    )

    rho = c3.number_input(
        "",
        value=default_props[lith][2],
        step=0.05,
        key=f"{lith}_rho"
    )

    lithology[lith] = {
        "Vp": vp,
        "Vs": vs,
        "Density": rho,
    }


# -----------------------------
# Layer model
# -----------------------------

st.subheader("Layer Model")

n_layers = st.number_input(
    "Number of Layers",
    min_value=2,
    max_value=40,
    value=3,
    step=1
)

c0, c1, c2 = st.columns([0.4, 0.8, 1.2])
c0.markdown("**#**")
c1.markdown("**Thick**")
c2.markdown("**Lithology**")

layers = []

for i in range(n_layers):
    c0, c1, c2 = st.columns([0.4, 0.8, 1.2])

    c0.write(f"{i + 1}")

    thickness = c1.number_input(
        "",
        value=100.0,
        step=5.0,
        key=f"thick_{i}"
    )

    lith = c2.selectbox(
        "",
        lith_names,
        index=min(i, len(lith_names) - 1),
        key=f"lith_{i}"
    )

    vp = lithology[lith]["Vp"]
    vs = lithology[lith]["Vs"]
    rho = lithology[lith]["Density"]

    layers.append([thickness, lith, vp, vs, rho])

df = pd.DataFrame(
    layers,
    columns=["Thickness", "Lithology", "Vp", "Vs", "Density"]
)

df["AI"] = df["Vp"] * df["Density"]


# -----------------------------
# Interface table
# -----------------------------

interface_rows = []
interface_twt = []
interface_depth = []

for i in range(n_layers - 1):
    depth = df["Thickness"].iloc[: i + 1].sum()

    twt = 0.0
    for k in range(i + 1):
        twt += 2 * df["Thickness"].iloc[k] / df["Vp"].iloc[k]

    r0 = acoustic_rc(
        df["Vp"].iloc[i],
        df["Density"].iloc[i],
        df["Vp"].iloc[i + 1],
        df["Density"].iloc[i + 1],
    )

    interface_depth.append(depth)
    interface_twt.append(twt)

    interface_rows.append(
        {
            "Interface": f"{i + 1}-{i + 2}",
            "Lithologies": f"{df['Lithology'].iloc[i]}-{df['Lithology'].iloc[i + 1]}",
            "Depth (m)": depth,
            "TWT (s)": twt,
            "Acoustic RC": r0,
        }
    )

interface_df = pd.DataFrame(interface_rows)

with st.expander("Show expanded model tables"):
    st.subheader("Expanded Layer Model")
    st.dataframe(df, use_container_width=True)

    st.subheader("Interfaces")
    st.dataframe(interface_df, use_container_width=True)


# -----------------------------
# Time, angles, wavelet
# -----------------------------

time = np.arange(0, tmax, dt)
angles = np.arange(min_angle, max_angle + angle_inc, angle_inc)

_, wavelet = ricker_wavelet(freq, dt)
wavelet = rotate_phase(wavelet, phase_deg)


# -----------------------------
# Flat angle gather
# -----------------------------

gather = np.zeros((len(time), len(angles)))

for j, angle_deg in enumerate(angles):
    trace_reflectivity = np.zeros_like(time)

    for i in range(n_layers - 1):
        t0 = interface_twt[i]
        sample = int(t0 / dt)

        if sample < len(time):
            theta = np.deg2rad(angle_deg)

            if avo_method == "Acoustic":
                amp = interface_df["Acoustic RC"].iloc[i]
            else:
                amp = aki_richards(
                    df["Vp"].iloc[i],
                    df["Vs"].iloc[i],
                    df["Density"].iloc[i],
                    df["Vp"].iloc[i + 1],
                    df["Vs"].iloc[i + 1],
                    df["Density"].iloc[i + 1],
                    theta,
                )

            trace_reflectivity[sample] = amp

    gather[:, j] = np.convolve(trace_reflectivity, wavelet, mode="same")


# -----------------------------
# Logs in TWT
# -----------------------------

thicknesses = df["Thickness"].values
velocities = df["Vp"].values

vp_twt, depth_log, vp_log = build_twt_log(
    thicknesses,
    velocities,
    df["Vp"].values
)

vs_twt, _, vs_log = build_twt_log(
    thicknesses,
    velocities,
    df["Vs"].values
)

rho_twt, _, rho_log = build_twt_log(
    thicknesses,
    velocities,
    df["Density"].values
)

ai_twt, _, ai_log = build_twt_log(
    thicknesses,
    velocities,
    df["AI"].values
)


# -----------------------------
# Plot logs and gather
# -----------------------------

st.subheader("Logs and Flat Angle Gather")

fig, axes = plt.subplots(
    1,
    5,
    figsize=(16, 8),
    sharey=True,
    gridspec_kw={"width_ratios": [1, 1, 1, 1, 3]}
)

axes[0].plot(vp_log, vp_twt, color="black")
axes[0].invert_yaxis()
axes[0].set_ylim(tmax, 0)
axes[0].set_title("Vp")
axes[0].set_xlabel("m/s")
axes[0].set_ylabel("TWT (s)")
axes[0].grid(True)

axes[1].plot(vs_log, vs_twt, color="black")
axes[1].set_title("Vs")
axes[1].set_xlabel("m/s")
axes[1].grid(True)

axes[2].plot(rho_log, rho_twt, color="black")
axes[2].set_title("Density")
axes[2].set_xlabel("g/cc")
axes[2].grid(True)

axes[3].plot(ai_log, ai_twt, color="black")
axes[3].set_title("AI")
axes[3].set_xlabel("AI")
axes[3].grid(True)

plot_wiggles(
    axes[4],
    gather,
    time,
    angles,
    angle_inc,
    wiggle_scale
)

axes[4].set_ylim(tmax, 0)
axes[4].set_title(f"{avo_method} Flat Angle Gather")

for ax in axes:
    for d, t in zip(interface_depth, interface_twt):
        ax.axhline(t, color="gray", linestyle="--", linewidth=0.6)

for d, t in zip(interface_depth, interface_twt):
    axes[0].text(
        axes[0].get_xlim()[0],
        t,
        f"{d:.0f} m",
        va="bottom",
        ha="left",
        fontsize=8,
        color="black"
    )

plt.tight_layout()
st.pyplot(fig)


# -----------------------------
# Colour gather
# -----------------------------

if display_mode in ["Colour", "Both"]:
    st.subheader("Colour Flat Angle Gather")

    fig2, ax2 = plt.subplots(figsize=(10, 7))

    vmax = np.max(np.abs(gather))
    if vmax == 0:
        vmax = 1

    ax2.imshow(
        gather,
        aspect="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        extent=[angles[0], angles[-1], time[-1], time[0]],
    )

    ax2.set_xlabel("Angle (degrees)")
    ax2.set_ylabel("TWT (s)")
    ax2.set_ylim(tmax, 0)
    ax2.set_title(f"{avo_method} Flat Angle Gather")

    for d, t in zip(interface_depth, interface_twt):
        ax2.axhline(t, color="black", linestyle="--", linewidth=0.5)
        ax2.text(angles[0], t, f"{d:.0f} m", va="bottom", fontsize=8)

    st.pyplot(fig2)