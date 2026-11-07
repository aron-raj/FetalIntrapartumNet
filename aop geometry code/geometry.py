import cv2
import math
import numpy as np
def get_largest_contour_ellipse(mask):
    if mask.sum() == 0:
        return None
    mask_uint = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None
    return cv2.fitEllipse(largest)
def compute_aop_angle(fh_ellipse, ps_ellipse):
    element = (fh_ellipse[0], (fh_ellipse[1][1], fh_ellipse[1][0]), fh_ellipse[2] - 90)
    element1 = (ps_ellipse[0], (ps_ellipse[1][1], ps_ellipse[1][0]), ps_ellipse[2] - 90)
    e1_2 = float(element1[2])
    d11 = float(element1[0][0]) - float(element1[1][0]) / 2 * math.cos(e1_2 * 0.01745)
    d12 = float(element1[0][1]) - float(element1[1][0]) / 2 * math.sin(e1_2 * 0.01745)
    d21 = float(element1[0][0]) + float(element1[1][0]) / 2 * math.cos(e1_2 * 0.01745)
    d22 = float(element1[0][1]) + float(element1[1][0]) / 2 * math.sin(e1_2 * 0.01745)
    a = float(element[1][0]) / 2
    b = float(element[1][1]) / 2
    angle = 2 * math.pi * float(element[2]) / 360
    dp21 = d21 - float(element[0][0])
    dp22 = d22 - float(element[0][1])
    dp2 = np.array([[dp21], [dp22]])
    Transmat1 = np.array([
        [math.cos(-angle), -math.sin(-angle)],
        [math.sin(-angle),  math.cos(-angle)]
    ])
    Transmat2 = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)]
    ])
    dpz2 = Transmat1 @ dp2
    dpz21 = dpz2[0][0]
    dpz22 = dpz2[1][0]
    if dpz21 ** 2 - a ** 2 == 0:
        dpz21 += 1
    val = (b**2 * dpz21**2 + a**2 * dpz22**2 - a**2 * b**2)
    if val >= 0:
        slope = (dpz21 * dpz22 - math.sqrt(val)) / (dpz21**2 - a**2)
    else:
        slope = 0
    bias = dpz22 - slope * dpz21
    qiepz1 = (-2 * slope * bias / b**2) / (2 * (1/a**2 + slope**2 / b**2))
    qiepz2 = qiepz1 * slope + bias
    qiepz = np.array([[qiepz1], [qiepz2]])
    qiep = Transmat2 @ qiepz
    qie1 = qiep[0][0] + element[0][0]
    qie2 = qiep[1][0] + element[0][1]
    ld1d3 = math.sqrt((d11 - d21)**2 + (d12 - d22)**2)
    ld3x4 = math.sqrt((d21 - qie1)**2 + (d22 - qie2)**2)
    ld1x4 = math.sqrt((d11 - qie1)**2 + (d12 - qie2)**2)
    if 2 * ld1d3 * ld3x4 == 0:
        return 0.0
    acos_val = (ld1d3**2 + ld3x4**2 - ld1x4**2) / (2 * ld1d3 * ld3x4)
    acos_val = max(-1.0, min(1.0, acos_val)) 
    aod = math.acos(acos_val) / math.pi * 180
    return aod, (d11, d12), (d21, d22), (qie1, qie2)
def draw_geometry_overlay(image_bgr, fh_mask, ps_mask):
    fh_ellipse = get_largest_contour_ellipse(fh_mask)
    ps_ellipse = get_largest_contour_ellipse(ps_mask)
    overlay = image_bgr.copy()
    overlay[fh_mask > 0] = overlay[fh_mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4
    overlay[ps_mask > 0] = overlay[ps_mask > 0] * 0.6 + np.array([0, 0, 255]) * 0.4
    if fh_ellipse is None or ps_ellipse is None:
        return overlay, None

    try:
        aop, p_start, p_end, intersection = compute_aop_angle(fh_ellipse, ps_ellipse)
        cv2.ellipse(overlay, fh_ellipse, (0, 255, 0), 2)
        cv2.ellipse(overlay, ps_ellipse, (0, 0, 255), 2)
        pt1 = (int(p_start[0]), int(p_start[1]))
        pt2 = (int(p_end[0]), int(p_end[1]))
        cv2.line(overlay, pt1, pt2, (255, 255, 0), 2)
        pt3 = (int(intersection[0]), int(intersection[1]))
        cv2.line(overlay, pt2, pt3, (0, 255, 255), 2)
        cv2.putText(overlay, f"AoP: {aop:.1f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        return overlay, aop
    except Exception as e:
        print(f"Geometry computation failed: {e}")
        return overlay, None