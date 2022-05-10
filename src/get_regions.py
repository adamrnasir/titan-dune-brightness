import cv2
import numpy as np
from scipy.spatial import KDTree
import time

N_NEIGHBORS = 5

# base = cv2.imread('in/base_clean.png', cv2.IMREAD_GRAYSCALE)
# inv = cv2.bitwise_not(base)


def get_theta(line):
    return np.arctan2(
        abs((line[0][3] - line[0][1])), abs((line[0][2] - line[0][0])))


def threshold(img):
    # Normalize image values from 0 to 255
    normalized = cv2.normalize(
        img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Gausian blur
    # blurred = cv2.GaussianBlur(normalized, (49, 49), 0)

    # Normalize image values from 0 to 255
    # normalized = cv2.normalize(
    #     blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Threshold the image using Otsu's method
    thresh = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 255, 0)

    return thresh


def get_light_lines(thresh):

    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 15,
                            minLineLength=100, maxLineGap=15)
    line_centers = list(map(
        lambda line: [(line[0][0] + line[0][2]) / 2, (line[0][1] + line[0][3]) / 2], lines))
    tree = KDTree(line_centers)
    d, j = tree.query(line_centers, k=N_NEIGHBORS)

    removal_indices = set()

    avg_theta = 0
    for line in lines:
        avg_theta += get_theta(line)
    avg_theta /= len(lines)

    for i, line in enumerate(lines):
        k_theta = 0
        for k in j[i]:
            k_theta += get_theta(lines[k])
        k_theta /= N_NEIGHBORS

        theta = get_theta(line)

        if abs(theta - k_theta) < 0.1 or abs(theta - avg_theta) < 0.1:
            removal_indices.add(i)

    ret = []
    for i, line in enumerate(lines):
        if i not in removal_indices:
            ret.append(line)

    return ret


def draw_lines(lines, img):
    bg = np.zeros(img.shape, np.uint8)
    for line in lines:
        cv2.line(bg, (line[0][0], line[0][1]),
                 (line[0][2], line[0][3]), (255, 255, 255), 2)
    # Save the image
    return bg


def main():
    timestr = time.strftime("%m%d%Y_%H%M%S")
    base = cv2.imread('in/base_clean.png', cv2.IMREAD_GRAYSCALE)

    thresh = threshold(base)
    lines = get_light_lines(thresh)
    lights = draw_lines(lines, base)

    cv2.imwrite('out/thresh_{}.png'.format(timestr), thresh)

    thresh = cv2.bitwise_not(thresh)
    lines = get_light_lines(thresh)
    darks = draw_lines(lines, base)

    cv2.imwrite('regions_out/lights_lines_{}.png'.format(timestr),
                lights - darks)
    cv2.imwrite('regions_out/darks_lines_{}.png'.format(timestr),
                cv2.bitwise_not(darks - lights))


if __name__ == '__main__':
    main()
