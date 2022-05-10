import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import get_regions as gr

NBINS = 256


def get_obstacle_coords(obstacles):
    obstacles_indices = np.where(obstacles == 255)
    obstacles_coords = list(zip(obstacles_indices[0], obstacles_indices[1]))


def get_obstacle_centers(obstacles):
    # Get geometric center of white points in obstacles image
    obstacles_indices = np.where(obstacles == 255)
    obstacles_coords = list(zip(obstacles_indices[0], obstacles_indices[1]))
    obstacles_center = np.mean(obstacles_coords, axis=0)
    return obstacles_center


def get_light_regions(base, comp_thresh, obstacles, obstacles_center):

    # Get furthest white pixel from obstacles center in lights image
    # lights_indices = np.where(lights == 255)
    lights_indices = np.where(comp_thresh - obstacles == 255)
    lights_coords = list(zip(lights_indices[0], lights_indices[1]))
    lights_distances = [np.linalg.norm(
        np.array(light) - np.array(obstacles_center)) for light in lights_coords]
    lights_values = [base[lights_coords[i][0], lights_coords[i][1]]
                     for i in range(len(lights_coords))]
    ld = np.max(lights_distances)
    return lights_distances, ld, lights_values


def get_dark_regions(base, comp_thresh, obstacles, obstacles_center):
    # Get furthest white pixel from obstacles center in darks image
    # darks_indices = np.where(darks == 255)
    darks_indices = np.where(comp_thresh == 0)
    darks_coords = list(zip(darks_indices[0], darks_indices[1]))
    darks_distances = [np.linalg.norm(
        np.array(dark) - np.array(obstacles_center)) for dark in darks_coords]
    darks_values = [base[darks_coords[i][0], darks_coords[i][1]]
                    for i in range(len(darks_coords))]
    dd = np.max(darks_distances)
    return darks_distances, dd, darks_values


def bin(lights_distances, darks_distances, lights_values, darks_values, ld, dd):
    furthest_r = max(ld, dd)
    step = furthest_r / NBINS

    bins = [n * step for n in range(NBINS)]
    print(bins)

    li = {i: [] for i in bins}
    da = {i: [] for i in bins}

    for i in range(0, len(lights_distances), 1):
        print("light ", i, " out of ", len(
            lights_distances), ": Distance ", lights_distances[i], " Value ", lights_values[i])
        hash = lights_distances[i]
        j = 1
        while hash > bins[j-1]:
            j += 1
            if j == len(bins):
                break
        bin = bins[j-1]
        li[bin].append(lights_values[i])

    for i in range(0, len(darks_distances), 1):
        print("dark ", i, " out of ", len(
            darks_distances), ": Distance ", darks_distances[i], ", Value ", darks_values[i])
        hash = darks_distances[i]
        j = 1
        while hash > bins[j-1]:
            j += 1
            if j == len(bins):
                break
        bin = bins[j-1]
        da[bin].append(darks_values[i])

    lval = {d: np.mean(li[d]) for d in li.keys()}
    dval = {d: np.mean(da[d]) for d in da.keys()}

    ldivd = {d: lval[d] / dval[d] for d in li.keys()}

    ds = ldivd.keys()
    vs = ldivd.values()
    return ds, vs


def scatter(ds, vs, timestr):
    plt.scatter(ds, vs)
    plt.xlabel("Distance from obstacle, pixels")
    plt.ylabel("Ratio")
    plt.title("Brightness Ratio vs Distance")
    plt.savefig("graph/out_{}.png".format(timestr))
    plt.clf()


def write_img(img, name, outfolder):
    cv2.imwrite(os.path.join(outfolder, name) + ".png", img)


def main():
    timestr = time.strftime("%m%d%Y_%H%M%S")
    outfolder = "run_{}".format(timestr)
    os.mkdir(outfolder)

    base = cv2.imread('in/base_clean.png', cv2.IMREAD_GRAYSCALE)
    obstacles = cv2.imread('in/obs_clean.png', cv2.IMREAD_GRAYSCALE)
    comp_thresh = cv2.imread(
        'out/thresh_05092022_014021.png', cv2.IMREAD_GRAYSCALE)
    # darks = cv2.imread('in/darks.png', cv2.IMREAD_GRAYSCALE)
    # lights = cv2.imread('in/lights.png', cv2.IMREAD_GRAYSCALE)

    thresh = gr.threshold(base)
    lines = gr.get_light_lines(thresh)
    lights = gr.draw_lines(lines, base)

    obstacles_center = get_obstacle_centers(obstacles)
    lights_distances, ld, lights_values = get_light_regions(
        base, comp_thresh, obstacles, obstacles_center)
    darks_distances, dd, darks_values = get_dark_regions(
        base, comp_thresh, obstacles, obstacles_center)

    ds, vs = bin(lights_distances, darks_distances,
                 lights_values, darks_values, ld, dd)

    scatter(ds, vs, timestr)


if __name__ == "__main__":
    main()
