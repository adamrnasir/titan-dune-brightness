import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import get_regions as gr


NBINS = 256
NSKIP = 10000


def get_obstacle_regions(img):
    obstacles = cv2.inRange(img, (0, 254, 0), (0, 255, 0))
    return obstacles


def get_external_region(img):
    external = cv2.inRange(img, (254, 0, 0), (255, 0, 0))
    return external


def get_sample_region(img):
    obstacles = cv2.bitwise_not(get_obstacle_regions(img))
    external = cv2.bitwise_not(get_external_region(img))
    mask = cv2.bitwise_and(obstacles, external)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_and(gray, mask), mask


def get_obstacles(obstacles):
    # Detect obstacle contours
    contours, _ = cv2.findContours(obstacles, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Remove contours that are too small
    contours = [c for c in contours if cv2.contourArea(c) > 100]
    # Get leftmost point of each contour
    l = []
    c = []
    for contour in contours:
        l.append(contour[contour[:,:,0].argmin()][0])
        c.append(np.mean(contour, axis=0))
    centers = [x[0] for _, x in sorted(zip(l, c), key=lambda p: p[0][0])]
    lefts = sorted(l, key=lambda x: x[0])
    return lefts, centers


def get_light_regions(base, comp_thresh, obstacle_lefts, obstacle_centers):
    # Get furthest white pixel from obstacles center in lights image
    distmap = np.zeros(base.shape)
    lights_indices = np.where(comp_thresh == 255)
    lights_coords = sorted(list(zip(lights_indices[0], lights_indices[1])), key=lambda x: x[1])
    lights_distances = []
    # lidx = 0
    for i, l in enumerate(lights_coords):
        if i % NSKIP == 0:
            print("Getting distance ", i, " out of ", len(lights_coords))
        light = np.flip(l)
        min_dist = np.inf
        for o in obstacle_centers:
            min_dist = min(min_dist, np.linalg.norm(light - o))
        # if lidx < len(obstacle_lefts) - 1 and light[0] > obstacle_lefts[lidx + 1][0]:
        #     lidx += 1
        # dist = np.linalg.norm(light - obstacle_centers[lidx])
        lights_distances.append(min_dist)
        distmap[light[1], light[0]] = min_dist
    lights_values = [base[lights_coords[i][0], lights_coords[i][1]]
                     for i in range(len(lights_coords))]
    ld = np.max(lights_distances)
    return lights_distances, ld, lights_values, distmap


def bin(lights_distances, darks_distances, lights_values, darks_values, ld, dd):
    furthest_r = max(ld, dd)
    step = furthest_r / NBINS

    bins = [n * step for n in range(NBINS)]
    print(bins)

    li = {i: [] for i in bins}
    da = {i: [] for i in bins}

    for i in range(0, len(lights_distances), 1):
        if i % NSKIP == 0:
            print("Binning light ", i, " out of ", len(
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
        if i % NSKIP == 0:
            print("Binning dark ", i, " out of ", len(
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


def scatter(ds, vs, outfolder):
    plt.scatter(ds, vs)
    plt.xlabel("Distance from obstacle, pixels")
    plt.ylabel("Ratio")
    plt.title("Brightness Ratio vs Distance")
    plt.savefig(os.path.join(outfolder, "graph.png"))
    plt.clf()


def write_img(img, name, outfolder):
    cv2.imwrite(os.path.join(outfolder, name) + ".png", img)


def main():
    timestr = time.strftime("%m%d%Y_%H%M%S")
    outfolder = os.path.join("runs", "run_{}".format(timestr))
    os.mkdir(outfolder)

    base = cv2.imread("data/t8_reproc1_corramb_bidr_nldsar_v2.EQUI4.256PPD.8bit.png")
    # base = cv2.imread("data/base_clean.png")
    obstacles = get_obstacle_regions(base)
    sample, mask = get_sample_region(base)
    cv2.imwrite(os.path.join(outfolder, "sample.png"), sample)
    cv2.imwrite(os.path.join(outfolder, "obstacles.png"), obstacles)

    thresh = gr.threshold(sample)
    cv2.imwrite(os.path.join(outfolder, "thresh.png"), thresh)

    obstacle_lefts, obstacle_centers = get_obstacles(obstacles)

    lights_distances, ld, lights_values, distmap = get_light_regions(sample, np.bitwise_and(thresh, mask), obstacle_lefts, obstacle_centers)

    cv2.imwrite(os.path.join(outfolder, "distmap.png"), cv2.normalize(distmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))

    darks_distances, dd, darks_values, darkmap = get_light_regions(sample, np.bitwise_and(np.bitwise_not(thresh), mask), obstacle_lefts, obstacle_centers)
    
    cv2.imwrite(os.path.join(outfolder, "darkmap.png"), cv2.normalize(darkmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))

    ds, vs = bin(lights_distances, darks_distances,
                 lights_values, darks_values, ld, dd)

    scatter(ds, vs, outfolder)


if __name__ == "__main__":
    main()
