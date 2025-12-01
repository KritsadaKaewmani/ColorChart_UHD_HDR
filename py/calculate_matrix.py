import numpy as np

def xy_to_XYZ(x, y):
    return np.array([x/y, 1.0, (1.0-x-y)/y])

def get_rgb_to_xyz_matrix(primaries, white_point):
    # primaries: [[rx, ry], [gx, gy], [bx, by]]
    # white_point: [wx, wy]
    
    r_xy, g_xy, b_xy = primaries
    w_xy = white_point
    
    Xr, Yr, Zr = xy_to_XYZ(*r_xy)
    Xg, Yg, Zg = xy_to_XYZ(*g_xy)
    Xb, Yb, Zb = xy_to_XYZ(*b_xy)
    Xw, Yw, Zw = xy_to_XYZ(*w_xy)
    
    M_p = np.array([
        [Xr, Xg, Xb],
        [Yr, Yg, Yb],
        [Zr, Zg, Zb]
    ])
    
    S = np.linalg.inv(M_p).dot(np.array([Xw, Yw, Zw]))
    
    M = M_p * S
    return M

def get_bradford_matrix(source_wp, dest_wp):
    # source_wp, dest_wp: XYZ
    
    M_A = np.array([
        [0.8951000, 0.2664000, -0.1614000],
        [-0.7502000, 1.7135000, 0.0367000],
        [0.0389000, -0.0685000, 1.0296000]
    ])
    
    M_A_inv = np.linalg.inv(M_A)
    
    src_cone = M_A.dot(source_wp)
    dst_cone = M_A.dot(dest_wp)
    
    M_cone = np.array([
        [dst_cone[0]/src_cone[0], 0, 0],
        [0, dst_cone[1]/src_cone[1], 0],
        [0, 0, dst_cone[2]/src_cone[2]]
    ])
    
    M_cat = M_A_inv.dot(M_cone).dot(M_A)
    return M_cat

# ACES AP0 Primaries and White Point
ap0_primaries = [[0.7347, 0.2653], [0.0000, 1.0000], [0.0001, -0.0770]]
ap0_white = [0.32168, 0.33767]

# Rec.2020 Primaries and White Point (D65)
rec2020_primaries = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]]
rec2020_white = [0.3127, 0.3290]

# Calculate Matrices
M_ap0_to_xyz = get_rgb_to_xyz_matrix(ap0_primaries, ap0_white)
M_rec2020_to_xyz = get_rgb_to_xyz_matrix(rec2020_primaries, rec2020_white)

# Calculate CAT (Bradford)
XYZ_ap0_w = xy_to_XYZ(*ap0_white)
XYZ_rec2020_w = xy_to_XYZ(*rec2020_white)
M_cat = get_bradford_matrix(XYZ_ap0_w, XYZ_rec2020_w)

# Total Matrix: AP0 -> XYZ -> CAT -> XYZ -> Rec2020
M_total = np.linalg.inv(M_rec2020_to_xyz).dot(M_cat).dot(M_ap0_to_xyz)

print("ACES AP0 to Rec.2020 Matrix (Linear, Bradford Adapted):")
print(M_total)

print("\nFlattened for copy-paste:")
print(list(M_total.flatten()))
