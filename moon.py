import ephem
import cv2
import numpy as np
import random
import math

def rotate(mat, angle):
    # rotate image from https://stackoverflow.com/a/33564950

    if angle == 0:
        return mat

    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)

    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=(1,1,1))
    return rotated_mat


#####################

h = 20
w = 20

border_h = 8
border_w = 8

gatech = ephem.Observer()
gatech.lon, gatech.lat = '51.509865', '-0.118092' # london
######################

d = ephem.Date('2018/01/01 12:00')

moon = ephem.Moon()

# get min max distances to sun and earth
max_dist = 0
min_dist = 9999

max_dist_sun = 0
min_dist_sun = 9999

for i in xrange(365):
    gatech.date = ephem.Date(d + i)
    moon.compute(gatech)
    
    dist_earth = moon.earth_distance
    dist_sun = moon.sun_distance

    if dist_earth < min_dist:
        min_dist = dist_earth
    if dist_earth > max_dist:
        max_dist = dist_earth

    if dist_sun < min_dist_sun:
        min_dist_sun = dist_sun
    if dist_sun > max_dist_sun:
        max_dist_sun = dist_sun


max_h = max_w = int(np.sqrt(h ** 2 + w ** 2) + 0.5)
cal = np.ones((13 * border_h + 12 * max_h, 32 * border_w + 31 * max_w, 3)).astype('float32')

prev_month = 1

for i in xrange(365):
    gatech.date = ephem.Date(d + i)
    moon.compute(gatech)

    year, month, day = gatech.date.triple()

    if month != prev_month:
        prev_month = month

    day = int(day - 0.5)

    m = np.ones((h, w, 3)).astype('float32') * moon.phase / 100.

    m[:, :, 1] = (moon.earth_distance - min_dist) / (max_dist - min_dist)
    m[:, :, 0] = 1 - (moon.sun_distance - min_dist_sun) / (max_dist_sun - min_dist_sun)

    m[0:2, :, :] = 0
    m[-2:, :, :] = 0
    m[:, 0:2, :] = 0
    m[:, -2:, :] = 0

    off_h = random.randint(0, border_h // 2)
    off_w = random.randint(0, border_w // 2)

    deg = random.randint(-15, 15)
    rot_m = rotate(m, deg)

    start_r = (month - 1) * max_h + month * border_h + off_h
    end_r = month * max_h + month * border_h + off_h

    start_c = (day - 1) * max_w + day * border_w + off_w
    end_c = day * max_w + day * border_w + off_w

    d_r = rot_m.shape[0] - (end_r - start_r)
    d_c = rot_m.shape[1] - (end_c - start_c)

    cal[start_r : end_r + d_r, start_c : end_c + d_c] = rot_m

rotated = rotate(cal, 3)

rotated = cv2.copyMakeBorder(rotated, top=150, bottom=200, left=50, right=50, borderType=cv2.BORDER_CONSTANT, value=[1, 1, 1])

cv2.imshow('cal', rotated)

cv2.waitKey(0)
cv2.imwrite('cal.png', rotated * 255)


