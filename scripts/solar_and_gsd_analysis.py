"""
Exact Solar Geometry (NOAA Algorithm) & GSD Mathematical Verification
for Borobudur Temple (7.6079 S, 110.2038 E) on 2025-08-27 (UTC+7 / WIB)
"""
import math
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = r"D:\~Ideas n Innovation\25-8-26 --Borobudur\Revision\experiments"
os.makedirs(OUT_DIR, exist_ok=True)

LAT = -7.6079
LON = 110.2038
TZ = 7.0
# Day of year for 2025-08-27: Jan(31)+Feb(28)+Mar(31)+Apr(30)+May(31)+Jun(30)+Jul(31)+27 = 239
DOY = 239

def solar_position(hour_wib):
    # NOAA fractional year gamma (radians)
    gamma = (2.0 * math.pi / 365.0) * (DOY - 1 + (hour_wib - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.001480 * math.sin(3 * gamma)
    )
    time_offset = eqtime + 4.0 * LON - 60.0 * TZ
    tst = hour_wib * 60.0 + time_offset
    ha = math.radians((tst / 4.0) - 180.0)
    lat_r = math.radians(LAT)
    cos_zenith = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    altitude = 90.0 - math.degrees(zenith)
    # Azimuth clockwise from North
    sin_zen = math.sin(zenith)
    if sin_zen < 1e-9:
        az = 0.0
    else:
        cos_az = (math.sin(lat_r) * math.cos(zenith) - math.sin(decl)) / (math.cos(lat_r) * sin_zen)
        cos_az = max(-1.0, min(1.0, cos_az))
        az = math.degrees(math.acos(cos_az))
        if ha > 0:
            az = (az + 180.0) % 360.0
        else:
            az = (180.0 - az) % 360.0
    return altitude, az

hours = np.linspace(6.0, 12.0, 145)
alts, azs = [], []
with open(os.path.join(OUT_DIR, "solar_geometry_borobudur_20250827.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Time_WIB", "Solar_Altitude_deg", "Solar_Azimuth_deg"])
    for h in np.arange(6.0, 12.25, 0.25):
        alt, az = solar_position(float(h))
        hh = int(h)
        mm = int((h - hh) * 60)
        w.writerow([f"{hh:02d}:{mm:02d}", f"{alt:.2f}", f"{az:.2f}"])

for h in hours:
    alt, az = solar_position(float(h))
    alts.append(alt)
    azs.append(az)

for t_str, h_val in [("07:00", 7.0), ("07:44", 7 + 44/60), ("08:00", 8.0), ("08:30", 8.5), ("09:00", 9.0), ("09:04", 9 + 4/60), ("10:00", 10.0), ("12:00", 12.0)]:
    alt, az = solar_position(h_val)
    print(f"Time {t_str} WIB -> Altitude: {alt:.2f} deg, Azimuth: {az:.2f} deg")

# Render 300 dpi Solar Path Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
plt.rcParams.update({"font.size": 9, "font.family": "serif"})

ax1.plot(hours, alts, color="#1f4e79", lw=2.2, label="Solar Altitude")
ax1.axvspan(8.0, 9.0, color="#f4b183", alpha=0.40, label="Nominal Window (08:00–09:00 WIB)")
ax1.axvspan(7 + 44/60, 9 + 4/60, color="#ffd966", alpha=0.22, linestyle="--", label="Actual Field Capture (07:44–09:04)")
ax1.set_xlabel("Local Time (WIB, UTC+7)", fontsize=9)
ax1.set_ylabel("Solar Altitude (degrees)", fontsize=9)
ax1.set_title("(a) Solar Elevation Profile (27 Aug 2025)", fontsize=10, fontweight="bold")
ax1.set_xlim(6.0, 12.0)
ax1.set_ylim(0, 85)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(fontsize=8, loc="upper left")

ax2.plot(hours, azs, color="#833c0c", lw=2.2, label="Solar Azimuth (from North)")
ax2.axvspan(8.0, 9.0, color="#f4b183", alpha=0.40, label="Nominal Window (08:00–09:00 WIB)")
ax2.axvspan(7 + 44/60, 9 + 4/60, color="#ffd966", alpha=0.22, label="Actual Field Capture (07:44–09:04)")
ax2.set_xlabel("Local Time (WIB, UTC+7)", fontsize=9)
ax2.set_ylabel("Solar Azimuth (degrees)", fontsize=9)
ax2.set_title("(b) Solar Azimuth Trajectory (27 Aug 2025)", fontsize=10, fontweight="bold")
ax2.set_xlim(6.0, 12.0)
ax2.set_ylim(0, 95)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(fontsize=8, loc="lower left")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Fig_Solar_Geometry_300dpi.png"), dpi=300)
plt.close(fig)

# GSD theoretical verification
# Sony a7 III (ILCE-7M3) & Sony a7C (ILCE-7C): 35.6 x 23.8 mm full-frame CMOS, 6024 x 4024 px (RAW) / 6000 x 4000 px (JPEG)
sensor_w_mm = 35.6
img_w_px = 6024
pitch_mm = sensor_w_mm / img_w_px  # ~0.005910 mm/px
print(f"\nSensor pixel pitch: {pitch_mm*1000:.3f} um/px ({sensor_w_mm} mm / {img_w_px} px)")
for d_cm in [230.0, 153.33, 73.66]:
    for f_mm in [24.0, 28.0, 50.0]:
        gsd = (d_cm * 10.0 * pitch_mm) / f_mm
        fov_w_cm = (d_cm * sensor_w_mm) / f_mm
        print(f"  d={d_cm:6.2f} cm, f={f_mm:4.1f} mm -> GSD={gsd:.3f} mm/px, FOV_width={fov_w_cm:.1f} cm")
