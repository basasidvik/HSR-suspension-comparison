"""Compare passive, semi-active SH--ADD, and active suspension.

This is a reproducible numerical companion to Model.slx.  The slx file is
used as the parameter reference; MATLAB/Simulink is not available in this
workspace, so the exposed quarter-car equations are integrated directly.
The same seeded rail realization is applied to all three cases.
"""
from pathlib import Path
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import welch
import matplotlib.pyplot as plt

OUT = Path("results")
OUT.mkdir(exist_ok=True)

# Model.slx values and the LaTeX assumptions.
MS, MU = 875.0, 875.0
KS, KT = 2_000.0, 8_000.0
CTIRE = 35.0
CPASSIVE = 0.0  # requested passive baseline: damper inactive
K0_MR, X0_MR = 0.92, 0.59
G = 9.8
DT, T_END = 0.002, 60.0
SEED = 20261101


def make_rail(t):
    """Track elevation: deterministic sleepers/roughness plus model-like noise."""
    rng = np.random.default_rng(SEED)
    # The two model random blocks are represented by common, interpolated
    # disturbances (0.1 s ballast variation and 0.001 s rail variation).
    tb = np.arange(0, t[-1] + 0.1, 0.1)
    tr = np.arange(0, t[-1] + 0.001, 0.001)
    ballast = np.interp(t, tb, rng.uniform(-1e-3, 1e-3, tb.size))
    rail = np.interp(t, tr, rng.uniform(-1e-3, 1e-3, tr.size))
    sleeper = 2.0e-4 * np.sin(2 * np.pi * 1.45 * t)
    long_wave = 3.0e-4 * np.sin(2 * np.pi * 0.35 * t)
    zr = sleeper + long_wave + 0.35 * ballast + 0.20 * rail
    zr_dot = np.gradient(zr, t)
    return zr, zr_dot


def operating_scenario_outputs(t):
    """Generate speed, acceleration, lateral, and four-corner load values.

    This follows the four-corner equations in HSR_Suspension_Assumptions.tex.
    The signs identify front/rear and left/right load shifts; the curve cases
    use the two stated radius limits so the geometric sensitivity is visible.
    """
    m, L, W, H = 3500.0, 20.5, 1.435, 5.45
    vmax = 360.0 / 3.6
    a_long = vmax / t[-1]
    speed_up = a_long * t
    speed_down = vmax - a_long * t

    cant_deg = 6.0
    cant_accel = G * np.tan(np.deg2rad(cant_deg))
    # Keep a 20% rollover/load margin.  This is the maximum unbalanced
    # lateral acceleration allowed by the supplied H and W values.
    lateral_limit = 0.80 * G * W / (2 * H)

    def corner_masses(speed, longitudinal_accel, radius=None, lateral_accel=None):
        ft = m * longitudinal_accel
        longitudinal = ft * H / (2 * G * L)
        if lateral_accel is None:
            lateral = 0.0 if radius is None else m * speed**2 * H / (2 * G * W * radius)
        else:
            lateral = m * lateral_accel * H / (2 * G * W)
        # [front-left, front-right, rear-left, rear-right]
        return np.column_stack((
            np.full(t.size, m / 4) + longitudinal + lateral,
            np.full(t.size, m / 4) + longitudinal - lateral,
            np.full(t.size, m / 4) - longitudinal + lateral,
            np.full(t.size, m / 4) - longitudinal - lateral,
        ))

    scenarios = {
        "acceleration": (speed_up, np.full(t.size, a_long),
                         np.zeros(t.size), corner_masses(speed_up, a_long)),
        "deceleration": (speed_down, np.full(t.size, -a_long),
                         np.zeros(t.size), corner_masses(speed_down, -a_long)),
    }
    for radius in (2500.0, 4000.0):
        # v^2/R - g*tan(cant) is the lateral acceleration remaining after
        # cant compensation.  Limit speed so the corner normal loads stay
        # positive with the chosen 20% safety margin.
        vmax_curve = np.sqrt(radius * (lateral_limit + cant_accel))
        speed_curve = vmax_curve * t / t[-1]
        unbalanced_lateral = speed_curve**2 / radius - cant_accel
        scenarios[f"curve_R{int(radius)}m"] = (
            speed_curve, np.zeros(t.size), unbalanced_lateral,
            corner_masses(speed_curve, 0.0, radius, unbalanced_lateral))
    return scenarios


def save_operating_scenarios(t):
    scenarios = operating_scenario_outputs(t)
    for name, (speed, along, lateral, masses) in scenarios.items():
        with (OUT / f"{name}_values.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "speed_kmh", "longitudinal_accel_mps2",
                        "lateral_accel_mps2", "front_left_kg", "front_right_kg",
                        "rear_left_kg", "rear_right_kg", "mass_sum_kg"])
            for i in range(t.size):
                w.writerow([t[i], speed[i] * 3.6, along[i], lateral[i],
                            *masses[i], np.sum(masses[i])])

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    speed, along, lateral, _ = scenarios["acceleration"]
    axes[0].plot(t, speed * 3.6, color="#333333")
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].set_title("Operating scenarios: speed and acceleration")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(t, along, label="Acceleration", color="#D55E00")
    axes[1].plot(t, scenarios["deceleration"][1], label="Deceleration", color="#0072B2")
    axes[1].set_ylabel("Longitudinal acceleration (m/s²)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(t, scenarios["curve_R2500m"][2], label="Curve R = 2500 m")
    axes[2].plot(t, scenarios["curve_R4000m"][2], label="Curve R = 4000 m")
    axes[2].set_ylabel("Lateral acceleration (m/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "operating_scenarios.png", dpi=180)
    plt.close(fig)


def effective_corner_masses(raw):
    """Keep the raw LaTeX result, but make masses usable by a dynamic solver."""
    # Limit each corner to 10--190% of its nominal quarter mass.  This is a
    # numerical/physical admissibility bound; the unbounded raw result is
    # retained in the CSV files for validation.
    clipped = np.clip(raw, 0.10 * MS, 1.90 * MS)
    return clipped * (3500.0 / clipped.sum(axis=1, keepdims=True))


def simulate_full_operating_case(case, scenario, t, track_x, track_z):
    """Four-corner vertical model driven by a common spatial rail profile."""
    speed, along, lateral, raw_mass = scenario
    # The raw load-transfer values are reported separately.  The supplied
    # curve equation can produce negative masses at 360 km/h, so the bounded
    # effective value is used only as a quasi-static vertical load shift;
    # inertia remains the validated Model.slx quarter-car mass.
    mass = effective_corner_masses(raw_mass)
    xpos = np.cumsum(speed) * DT
    offsets = np.array([0.0, 1.435, 20.5, 21.935])
    states = np.zeros((4, 5))
    body_acc = np.zeros(t.size)
    corner_acc = np.zeros((t.size, 4))
    currents = np.zeros((t.size, 4))
    for k in range(t.size):
        accs = []
        for j in range(4):
            zs, vs, zu, vu, z = states[j]
            ms = MS
            road = np.interp(xpos[k] + offsets[j], track_x, track_z)
            road2 = np.interp(min(xpos[k] + offsets[j] + 0.01, track_x[-1]), track_x, track_z)
            road_dot = (road2 - road) / 0.01
            rel, relv = zs - zu, vs - vu
            tire = KT * (zu - road) + CTIRE * (vu - road_dot)
            load_shift = (mass[k, j] - MS) * G
            if case == "passive":
                force, current, zdot = KS * rel, 0.0, 0.0
            elif case == "semi-active":
                # Bounded SH--ADD-like damper command.
                body_est = -KS * rel / ms
                sigma = body_est**2 - (0.5 * vs)**2
                cdes = 1.0e5 if ((vs * relv > 0) if sigma <= 0 else (body_est * relv > 0)) else 100.0
                desired = cdes * relv
                current = np.clip((abs(desired) - 5.62 * abs(relv)) / (31.58 * (abs(z) + 1e-6)), 0, 2.5)
                alpha, c0, gamma, beta, bwA, n = mr_params(current)
                zabs = max(abs(z), 1e-12)
                zdot = -gamma * abs(relv) * z * zabs**(n - 1) - beta * relv * zabs**n + bwA * relv
                force = KS * rel + c0 * relv + K0_MR * (rel - X0_MR) + alpha * z
            else:
                u = -float(K_LQR @ np.array([zs, vs, zu, vu]))
                force, current, zdot = KS * rel - np.clip(u, -2.0e4, 2.0e4), 0.0, 0.0
            sprung_acc = (-force + load_shift) / ms
            unsprung_acc = (force - tire - load_shift) / MU
            accs.append(sprung_acc)
            corner_acc[k, j] = sprung_acc
            currents[k, j] = current
            if k + 1 < t.size:
                states[j, 4] += DT * zdot
                states[j, 1] += DT * sprung_acc
                states[j, 3] += DT * unsprung_acc
                states[j, 0] += DT * states[j, 1]
                states[j, 2] += DT * states[j, 3]
        body_acc[k] = np.mean(accs)
    return body_acc, corner_acc, currents, raw_mass, mass


def save_full_model_outputs(t):
    scenarios = operating_scenario_outputs(t)
    max_distance = 3500.0
    track_x = np.linspace(0.0, max_distance, int(max_distance / 0.02) + 1)
    rng = np.random.default_rng(SEED)
    spatial_noise = np.interp(track_x, np.arange(0, max_distance + 0.5, 0.5),
                              rng.uniform(-1e-3, 1e-3, int(max_distance / 0.5) + 1))
    track_z = (2e-4 * np.sin(2 * np.pi * track_x / 0.75) +
               3e-4 * np.sin(2 * np.pi * track_x / 7.0) + 0.25 * spatial_noise)
    window = max(1, int(round(1.0 / DT)))
    kernel = np.ones(window) / window
    colors = {"passive": "#555555", "semi-active": "#0072B2", "active": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, (scenario_name, scenario) in zip(axes.flat, scenarios.items()):
        for case, color in colors.items():
            body_acc, corner_acc, currents, raw, effective = simulate_full_operating_case(
                case, scenario, t, track_x, track_z)
            amplitude = np.sqrt(np.convolve(body_acc**2, kernel, mode="same"))
            ax.plot(t, amplitude, label=case.replace('-', ' ').title(), color=color, lw=1.2)
            if case == "passive":
                speed, along, lateral, _ = scenario
                with (OUT / f"{scenario_name}_full_values.csv").open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["time_s", "speed_kmh", "longitudinal_accel_mps2", "lateral_accel_mps2",
                                "raw_FL_kg", "raw_FR_kg", "raw_RL_kg", "raw_RR_kg",
                                "effective_FL_kg", "effective_FR_kg", "effective_RL_kg", "effective_RR_kg",
                                "passive_body_accel_amplitude_mps2"])
                    for i in range(t.size):
                        w.writerow([t[i], speed[i] * 3.6, along[i], lateral[i], *raw[i], *effective[i], amplitude[i]])
            with (OUT / f"{scenario_name}_{case}_body_accel.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time_s", "body_acceleration_mps2", "body_acceleration_amplitude_rms_mps2"])
                for i in range(t.size):
                    w.writerow([t[i], body_acc[i], amplitude[i]])
        ax.set_title(scenario_name.replace('_', ' ').replace('R', 'R = '))
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Vertical acceleration RMS (m/s²)")
    axes[0, 0].legend()
    fig.suptitle("Full operating-case comparison: carriage vertical amplitude")
    fig.tight_layout()
    fig.savefig(OUT / "full_model_amplitude_by_operating_case.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, color in [("acceleration", "#D55E00"), ("deceleration", "#0072B2"),
                        ("curve_R2500m", "#009E73"), ("curve_R4000m", "#CC79A7")]:
        masses = scenarios[name][3]
        for j, label in enumerate(("FL", "FR", "RL", "RR")):
            axes[0 if "curve" not in name else 1].plot(
                t, masses[:, j], color=color, alpha=0.7,
                label=f"{name}, {label}" if j == 0 else None)
    axes[0].set_title("Corner supported-mass values: acceleration/deceleration")
    axes[1].set_title("Corner supported-mass values: curved travel")
    axes[0].set_ylabel("Corner mass (kg)")
    axes[1].set_ylabel("Corner mass (kg)")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend()
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "corner_mass_shifting.png", dpi=180)
    plt.close(fig)


def mr_params(current):
    i = np.clip(current, 0.0, 2.5)
    alpha = 31.58 * i + 2.21e3
    c0 = 7.21 * i + 5.62
    gamma = 1.14 * np.exp(0.026 * i) + 6.90 * np.exp(2.13e-5 * i)
    beta = 2.01e-3
    A = 1.08
    n = 0.34 * np.exp(0.95 * i) + 0.42 * np.exp(0.58 * i)
    return alpha, c0, gamma, beta, A, n


def lqr_gain():
    # x = [zs, vs, zu, vu], linearized around the requested passive
    # spring-only baseline (CPASSIVE = 0 N s/m).
    c = CPASSIVE
    A = np.array([[0, 1, 0, 0], [-KS/MS, -c/MS, KS/MS, c/MS],
                  [0, 0, 0, 1], [KS/MU, c/MU, -(KS+KT)/MU, -c/MU]])
    B = np.array([[0], [1/MS], [0], [-1/MU]])
    Q = np.diag([2e5, 2e3, 2e3, 20.0])
    R = np.array([[1e-3]])
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P).ravel()


K_LQR = lqr_gain()


def simulate(case, t, zr, zr_dot):
    x = np.zeros(5)  # zs, vs, zu, vu, Bouc-Wen z
    out = np.zeros((t.size, 5))  # body acceleration, travel, displacement, force, current
    for k in range(t.size):
        ti = t[k]
        road, road_dot = zr[k], zr_dot[k]
        zs, vs, zu, vu, z = x
        rel = zs - zu
        relv = vs - vu
        tire = KT * (zu - road) + CTIRE * (vu - road_dot)
        if case == "passive":
            force, current, zdot = KS * rel, 0.0, 0.0
        elif case == "semi-active":
            # SH--ADD switching metric from the LaTeX specification.
            body_acc_est = (-KS * rel) / MS
            sigma = body_acc_est**2 - (0.5 * vs)**2
            sky = 1.0e5 * vs if vs * relv > 0 else 100.0 * relv
            add = 1.0e5 * relv if relv * body_acc_est > 0 else 100.0 * relv
            desired = sky if sigma <= 0 else add
            # Approximate inverse current command, then use the mandatory
            # current-dependent Bouc--Wen force model below.
            current = np.clip((abs(desired) - 2.21e3 * abs(z)) / 31.58, 0, 2.5)
            alpha, c0, gamma, beta, bwA, n = mr_params(current)
            zabs = max(abs(z), 1e-12)
            zdot = -gamma * abs(relv) * z * zabs**(n - 1) - beta * relv * zabs**n + bwA * relv
            force = KS * rel + c0 * relv + K0_MR * (rel - X0_MR) + alpha * z
        else:  # active LQG request: full-state LQG design, state feedback path
            # Actuator force is parallel to the spring. Limit is only a
            # numerical protection against an unrealistic transient.
            u = -float(K_LQR @ np.array([zs, vs, zu, vu]))
            # The state-space input u acts upward on the sprung mass, while
            # ``force`` is defined here as the restoring force on that mass.
            force, current, zdot = KS * rel - np.clip(u, -2.0e4, 2.0e4), 0.0, 0.0
        acc = (-force) / MS
        unsprung_acc = (force - tire) / MU
        out[k] = [acc, rel, zs, force, current]
        if k + 1 == t.size:
            break
        # Semi-implicit Euler is robust to the switching damper while keeping
        # exactly the same time base for every case.
        x[4] += DT * zdot
        x[1] += DT * acc
        x[3] += DT * unsprung_acc
        x[0] += DT * x[1]
        x[2] += DT * x[3]
    return out


def main():
    t = np.arange(0, T_END + DT / 2, DT)
    zr, zr_dot = make_rail(t)
    cases = {name: simulate(name, t, zr, zr_dot)
             for name in ("passive", "semi-active", "active")}
    fs, psds = None, {}
    for name, y in cases.items():
        fs, psds[name] = welch(y[:, 0], fs=1 / DT, window="hann",
                               nperseg=8192, detrend="constant")
    plt.figure(figsize=(10, 5.8))
    for name, color in [("passive", "#555555"), ("semi-active", "#0072B2"),
                        ("active", "#D55E00")]:
        plt.semilogy(fs, np.sqrt(psds[name]), label=name.replace('-', ' ').title(),
                     lw=1.8, color=color)
    plt.xlim(0, 10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Carriage vertical acceleration ASD (m/s²/√Hz)")
    plt.title("Frequency transferred to the main carriage")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "carriage_frequency_comparison.png", dpi=180)
    plt.close()

    # Simple time-domain comparison requested by the user: moving RMS
    # amplitude of the same carriage acceleration signal.
    window = max(1, int(round(1.0 / DT)))
    kernel = np.ones(window) / window
    plt.figure(figsize=(10, 5.8))
    for name, color in [("passive", "#555555"), ("semi-active", "#0072B2"),
                        ("active", "#D55E00")]:
        amplitude = np.sqrt(np.convolve(cases[name][:, 0] ** 2, kernel, mode="same"))
        plt.plot(t, amplitude, label=name.replace('-', ' ').title(),
                 lw=1.4, color=color)
    plt.xlabel("Time (s)")
    plt.ylabel("Carriage vertical acceleration amplitude (m/s² RMS)")
    plt.title("Carriage acceleration amplitude versus time")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "carriage_acceleration_amplitude_vs_time.png", dpi=180)
    plt.close()

    with (OUT / "carriage_frequency_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frequency_Hz"] + list(cases))
        for i, freq in enumerate(fs):
            w.writerow([freq] + [np.sqrt(psds[n][i]) for n in cases])
    summary = {"seed": SEED, "dt_s": DT, "duration_s": T_END,
               "model_basis": "Model.slx exposed quarter-car parameters",
               "frequency_measure": "Welch acceleration ASD of sprung mass",
               "peak_asd": {n: float(np.max(np.sqrt(psds[n]))) for n in cases},
               "rms_acceleration": {n: float(np.sqrt(np.mean(cases[n][:, 0]**2))) for n in cases}}
    (OUT / "simulation_summary.json").write_text(json.dumps(summary, indent=2))
    save_operating_scenarios(t)
    save_full_model_outputs(t)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
