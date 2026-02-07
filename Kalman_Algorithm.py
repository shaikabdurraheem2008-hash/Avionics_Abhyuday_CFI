import numpy as np
import pandas as pd
from numpy.linalg import inv, eig

# =====================================================
# Load Dataset
# =====================================================
df = pd.read_csv("raw_data_fusion.csv", encoding="utf-8-sig")

# =====================================================
# SAFE NORMALIZATION
# =====================================================
def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n

# =====================================================
# Quaternion Helpers
# =====================================================
def quat_multiply(q, r):
    w0,x0,y0,z0 = q
    w1,x1,y1,z1 = r

    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_to_euler(q):
    w,x,y,z = q

    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    return np.degrees([roll, pitch, yaw])

# =====================================================
# ⭐ Gyroscope Bias (first 1000 samples assumed stationary)
# =====================================================
gyro_bias = df.iloc[:1000][["gyro_x","gyro_y","gyro_z"]].mean().values
print("\nGyro Bias:", gyro_bias)

# =====================================================
# ⭐ Robust Ellipsoid Calibration
# =====================================================
def ellipsoid_calibration(data):

    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]

    D = np.column_stack([
        X*X, Y*Y, Z*Z,
        2*Y*Z, 2*X*Z, 2*X*Y,
        2*X, 2*Y, 2*Z,
        np.ones(len(X))
    ])

    _, _, V = np.linalg.svd(D)
    params = V[-1]

    A = np.array([
        [params[0], params[5], params[4], params[6]],
        [params[5], params[1], params[3], params[7]],
        [params[4], params[3], params[2], params[8]],
        [params[6], params[7], params[8], params[9]]
    ])

    A3 = A[0:3,0:3]
    B = params[6:9]

    bias = -inv(A3) @ B

    # Translation
    T = np.eye(4)
    T[3,0:3] = bias

    R = T @ A @ T.T
    M = R[0:3,0:3] / -R[3,3]

    eigvals, eigvecs = eig(M)

    eigvals = np.abs(eigvals)  # Prevent negative sqrt
    scale = eigvecs @ np.diag(np.sqrt(1/eigvals)) @ eigvecs.T

    return bias.real, scale.real

# =====================================================
# Accelerometer Calibration
# =====================================================
acc_data = df[["acc_x","acc_y","acc_z"]].values.astype(float)
acc_bias, acc_scale = ellipsoid_calibration(acc_data)

print("\nAccelerometer Bias:\n", acc_bias)
print("\nAccelerometer Scale:\n", acc_scale)

# =====================================================
# Magnetometer Calibration
# =====================================================
mag_data = df[["mag_x","mag_y","mag_z"]].values.astype(float)
mag_bias, mag_scale = ellipsoid_calibration(mag_data)

print("\nMagnetometer Bias:\n", mag_bias)
print("\nMagnetometer Scale:\n", mag_scale)

# =====================================================
# ⭐ EKF Initialization
# =====================================================
q = np.array([1.0,0.0,0.0,0.0])
P = np.eye(4)*0.01

Q = np.eye(4)*0.0005
R = np.eye(3)*0.05

dt = 0.01

quat_log = []
euler_log = []

# =====================================================
# ⭐ EKF Fusion Loop
# =====================================================
for i in range(len(df)):

    gyro = df.loc[i,["gyro_x","gyro_y","gyro_z"]].values.astype(float)
    acc  = df.loc[i,["acc_x","acc_y","acc_z"]].values.astype(float)
    mag  = df.loc[i,["mag_x","mag_y","mag_z"]].values.astype(float)

    # ----- Apply Calibration -----
    gyro = gyro - gyro_bias
    acc  = acc_scale @ (acc - acc_bias)
    mag  = mag_scale @ (mag - mag_bias)

    acc = normalize(acc)

    # ----- Prediction -----
    omega = np.array([0, *gyro])
    dq = 0.5 * quat_multiply(q, omega) * dt
    q_pred = normalize(q + dq)

    P = P + Q

    # ----- Measurement Update -----
    qw,qx,qy,qz = q_pred

    g_est = np.array([
        2*(qx*qz - qw*qy),
        2*(qw*qx + qy*qz),
        qw*qw - qx*qx - qy*qy + qz*qz
    ])

    y = acc - g_est

    H = np.array([
        [-2*qy,  2*qz, -2*qw, 2*qx],
        [ 2*qx,  2*qw,  2*qz, 2*qy],
        [ 2*qw, -2*qx, -2*qy, 2*qz]
    ])

    S = H @ P @ H.T + R
    K = P @ H.T @ inv(S)

    q_temp = q_pred + K @ y
    q = normalize(q_temp)

    P = (np.eye(4) - K @ H) @ P

    # Safety check
    if np.isnan(q).any():
        print("NaN detected at index:", i)
        break

    quat_log.append(q)
    euler_log.append(quat_to_euler(q))

# =====================================================
# Save Output
# =====================================================
result = pd.DataFrame(quat_log, columns=["qw","qx","qy","qz"])
angles = pd.DataFrame(euler_log, columns=["Roll","Pitch","Yaw"])

final = pd.concat([result, angles], axis=1)
final.to_csv("fusion_output.csv", index=False)

print("\n✅ Fusion Completed Successfully")

