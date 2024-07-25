import math
import numpy as np
from Utils import Sphere, Capsule, Cylinder, AABB


def saturate(t):
    return min(max(t, 0), 1)


# 点P在线段上最近的点
def ClosestPointOnLineSegment(a, b, point):
    vec = b - a
    t = np.dot(point - a, vec) / np.dot(vec, vec)
    return a + saturate(t) * vec


def SphereLineSegmentCollision(a, b, s):
    cp = ClosestPointOnLineSegment(a, b, s.center)
    dis = np.linalg.norm(s.center - cp)
    if dis <= s.radius:
        return dis, True
    else:
        return dis, False


def SphereSphereCollision(s_a, s_b):
    dis = np.linalg.norm(s_a.center - s_b.center)
    if dis <= s_a.radius + s_b.radius:
        return dis, True
    else:
        return dis, False


def CapsuleCylinderCollision(c_a: Capsule, c_b: Cylinder):
    line_end_offset_a = c_a.up * c_a.radius
    a_A = c_a.base + line_end_offset_a
    a_B = c_a.tip - line_end_offset_a

    b_A = c_b.base
    b_B = c_b.tip

    v0 = b_A - a_A  # a1 - b1
    v1 = b_B - a_A
    v2 = b_A - a_B
    v3 = b_B - a_B

    d0 = np.dot(v0, v0)
    d1 = np.dot(v1, v1)
    d2 = np.dot(v2, v2)
    d3 = np.dot(v3, v3)

    if d2 < d0 or d2 < d1 or d3 < d0 or d3 < d1:
        best_A = a_B
    else:
        best_A = a_A

    best_B = ClosestPointOnLineSegment(b_A, b_B, best_A)
    best_A = ClosestPointOnLineSegment(a_A, a_B, best_B)

    dis = np.linalg.norm(best_A - best_B)
    if (best_B == b_A).all() or (best_B == b_B).all():
        vec_cb = best_B - c_b.center
        vec_ba = best_A - best_B
        theta = np.arccos(np.dot(vec_cb, vec_ba) / (np.linalg.norm(vec_cb) * np.linalg.norm(vec_ba)))
        max_dis = c_b.radius * np.sin(theta) + c_a.radius
    else:
        max_dis = c_b.radius + c_a.radius

    if dis <= max_dis:
        return dis, dis - max_dis, True
    else:
        return dis, dis - max_dis, False


