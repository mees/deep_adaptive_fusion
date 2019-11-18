from math import sqrt, isnan


def intersection_dist(px, py, qx, qy, rx, ry, dx, dy):
    l = (dy * (rx - qx) + dx * (qy - ry)) / (dy * (px - qx) + dx * (qy - py))
    m = (py * (rx - qx) + px * (qy - ry) + qx * ry - qy * rx) / (dy * (px - qx) + dx * (qy - py))
    if isnan(l) or isnan(m):
        return 0.0
    elif 0 <= l <= 1 and m > 0:
        return m / sqrt(dx * dx + dy * dy) * 2
    else:
        return 0.0
