import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------------------------
# LOAD DATA SAFELY
# -------------------------------------------------------
file_name = "Acc_Mag.csv"
df = pd.read_csv(file_name, skipinitialspace=True)

# Force numeric conversion
mag_df = df[["MagX", "MagY", "MagZ"]].apply(pd.to_numeric, errors='coerce')

# Remove rows containing invalid values
valid_rows = ~mag_df.isna().any(axis=1)
df = df[valid_rows].reset_index(drop=True)
mag = mag_df[valid_rows].values

print(f"Loaded {len(mag)} valid magnetometer samples")

# -------------------------------------------------------
# HARD IRON CALIBRATION
# -------------------------------------------------------
mag_min = np.min(mag, axis=0)
mag_max = np.max(mag, axis=0)
bias = (mag_max + mag_min) / 2

print("\n=== CALIBRATION PARAMETERS ===")
print(f"Bias (Hard Iron Offset): [{bias[0]:.2f}, {bias[1]:.2f}, {bias[2]:.2f}]")

mag_corrected = mag - bias

# -------------------------------------------------------
# SOFT IRON CALIBRATION
# -------------------------------------------------------
mag_radius = (mag_max - mag_min) / 2
avg_radius = np.mean(mag_radius)
scale = avg_radius / mag_radius

print(f"Scale Factors: [{scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f}]")

mag_calibrated = mag_corrected * scale

# -------------------------------------------------------
# QUALITY METRICS
# -------------------------------------------------------
# Check sphericity of calibrated data
cal_radius = np.linalg.norm(mag_calibrated, axis=1)
radius_std = np.std(cal_radius)
radius_mean = np.mean(cal_radius)
sphericity = (1 - radius_std / radius_mean) * 100

print(f"\nCalibrated Data Sphericity: {sphericity:.2f}%")
print(f"Mean Radius: {radius_mean:.2f}")
print(f"Radius Std Dev: {radius_std:.2f}")

# -------------------------------------------------------
# SAVE OUTPUT
# -------------------------------------------------------
df["MagX_cal"] = mag_calibrated[:, 0]
df["MagY_cal"] = mag_calibrated[:, 1]
df["MagZ_cal"] = mag_calibrated[:, 2]
df.to_csv("calibrated_mag_output.csv", index=False)
print("\n✅ Calibration complete. Output saved to 'calibrated_mag_output.csv'")

# -------------------------------------------------------
# ENHANCED VISUALIZATION
# -------------------------------------------------------
fig = plt.figure(figsize=(16, 7))

# --- RAW DATA PLOT ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(mag[:, 0], mag[:, 1], mag[:, 2], 
            c=np.linalg.norm(mag, axis=1), 
            cmap='viridis', 
            s=10, 
            alpha=0.6,
            edgecolors='none')
ax1.set_xlabel('MagX (μT)', fontsize=10)
ax1.set_ylabel('MagY (μT)', fontsize=10)
ax1.set_zlabel('MagZ (μT)', fontsize=10)
ax1.set_title("Raw Magnetometer Data\n(Ellipsoidal - Uncalibrated)", fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Set equal aspect ratio for raw data
max_range_raw = np.max(mag_radius)
mid_raw = bias
ax1.set_xlim(mid_raw[0] - max_range_raw*1.2, mid_raw[0] + max_range_raw*1.2)
ax1.set_ylim(mid_raw[1] - max_range_raw*1.2, mid_raw[1] + max_range_raw*1.2)
ax1.set_zlim(mid_raw[2] - max_range_raw*1.2, mid_raw[2] + max_range_raw*1.2)

# --- CALIBRATED DATA PLOT ---
ax2 = fig.add_subplot(122, projection='3d')
scatter = ax2.scatter(mag_calibrated[:, 0], 
                      mag_calibrated[:, 1], 
                      mag_calibrated[:, 2],
                      c=cal_radius,
                      cmap='plasma',
                      s=10,
                      alpha=0.6,
                      edgecolors='none')
ax2.set_xlabel('MagX (μT)', fontsize=10)
ax2.set_ylabel('MagY (μT)', fontsize=10)
ax2.set_zlabel('MagZ (μT)', fontsize=10)
ax2.set_title(f"Calibrated Magnetometer Data\n(Spherical - Sphericity: {sphericity:.1f}%)", 
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Set equal aspect ratio for calibrated data (critical for sphere visualization)
ax2.set_box_aspect([1, 1, 1])
max_range_cal = np.max(np.abs(mag_calibrated))
ax2.set_xlim(-max_range_cal*1.1, max_range_cal*1.1)
ax2.set_ylim(-max_range_cal*1.1, max_range_cal*1.1)
ax2.set_zlim(-max_range_cal*1.1, max_range_cal*1.1)

# Add colorbar for calibrated data
cbar = plt.colorbar(scatter, ax=ax2, pad=0.1, shrink=0.8)
cbar.set_label('Radius (μT)', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('magnetometer_calibration.png', dpi=150, bbox_inches='tight')
print("📊 Visualization saved as 'magnetometer_calibration.png'")
plt.show()

# -------------------------------------------------------
# OPTIONAL: 2D PROJECTIONS FOR DETAILED ANALYSIS
# -------------------------------------------------------
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))

projections = [('MagX', 'MagY', 0, 1), ('MagX', 'MagZ', 0, 2), ('MagY', 'MagZ', 1, 2)]

for idx, (xlabel, ylabel, ix, iy) in enumerate(projections):
    # Raw data
    axes[0, idx].scatter(mag[:, ix], mag[:, iy], s=5, alpha=0.5, c='red')
    axes[0, idx].set_xlabel(xlabel)
    axes[0, idx].set_ylabel(ylabel)
    axes[0, idx].set_title(f"Raw: {xlabel} vs {ylabel}")
    axes[0, idx].grid(True, alpha=0.3)
    axes[0, idx].set_aspect('equal')
    
    # Calibrated data
    axes[1, idx].scatter(mag_calibrated[:, ix], mag_calibrated[:, iy], s=5, alpha=0.5, c='blue')
    axes[1, idx].set_xlabel(xlabel)
    axes[1, idx].set_ylabel(ylabel)
    axes[1, idx].set_title(f"Calibrated: {xlabel} vs {ylabel}")
    axes[1, idx].grid(True, alpha=0.3)
    axes[1, idx].set_aspect('equal')
    
    # Add circles to calibrated plots for reference
    circle = plt.Circle((0, 0), radius_mean, fill=False, color='green', 
                        linestyle='--', linewidth=2, alpha=0.5, label='Expected')
    axes[1, idx].add_patch(circle)
    axes[1, idx].legend()

plt.tight_layout()
plt.savefig('magnetometer_2d_projections.png', dpi=150, bbox_inches='tight')
print("📊 2D projections saved as 'magnetometer_2d_projections.png'")
plt.show()
