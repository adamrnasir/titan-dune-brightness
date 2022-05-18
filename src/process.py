import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import get_regions as gr
import glob


NBINS = 256
NSKIP = 1000


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
    contours = [c for _, c in sorted(zip(l, contours), key=lambda p: p[0][0])]
    return contours, centers


def get_light_regions(base, comp_thresh, obstacle_contours):
    # distmap = np.zeros(base.shape)
    lights_indices = np.where(comp_thresh == 255)
    lights_coords = sorted(list(zip(lights_indices[0], lights_indices[1])), key=lambda x: x[1])
    lights_distances = []
    for i, l in enumerate(lights_coords):
        light = np.flip(l)
        min_dist = np.inf
        for c in obstacle_contours:
            m = map(lambda x: np.linalg.norm(light - x[0]), c[1::min(10,len(c)//50)])
            min_dist = min(min_dist, min(m))
        lights_distances.append(min_dist)
        # distmap[light[1], light[0]] = min_dist
    lights_values = [base[lights_coords[i][0], lights_coords[i][1]]
                     for i in range(len(lights_coords))]
    ld = np.max(lights_distances)
    return lights_distances, ld, lights_values


def hashbin(dists, vals, bins, dict):
    for i in range(0, len(dists), 1):
        hash = dists[i]
        j = 1
        while hash > bins[j-1]:
            j += 1
            if j == len(bins):
                break
        bin = bins[j-1]
        dict[bin].append(vals[i])


def bin(lights_distances, darks_distances, lights_values, darks_values, ld, dd):
    furthest_r = max(ld, dd)
    step = furthest_r / NBINS

    bins = [n * step for n in range(NBINS)]

    li = {i: [] for i in bins}
    da = {i: [] for i in bins}

    hashbin(lights_distances, lights_values, bins, li)
    hashbin(darks_distances, darks_values, bins, da)


    # Get mean bin size
    li_mean = np.mean([len(li[i]) for i in bins]) 
    da_mean = np.mean([len(da[i]) for i in bins])

    # Get bin size standard deviation
    li_std = np.std([len(li[i]) for i in bins])
    da_std = np.std([len(da[i]) for i in bins])

    lval = {d: np.mean(li[d]) for d in li.keys()}
    dval = {d: np.mean(da[d]) for d in da.keys()}

    print("Mean bin size:" + str(li_mean) + " " + str(da_mean))
    print("Std bin size:" + str(li_std) + " " + str(da_std))

    ndev = 1

    ldivd = { d: lval[d] / dval[d] for d in li.keys() 
              if (len(li[d]) > li_mean - ndev * li_std) 
              and (len(da[d]) > da_mean - ndev * da_std)}

    ds = list(ldivd.keys())
    vs = list(ldivd.values())
    return ds, vs


def scatter(img, ds, vs, outfolder):
    plt.scatter(ds, vs)
    plt.xlabel("Distance from obstacle, pixels")
    plt.ylabel("Ratio")
    plt.title("Brightness Ratio vs Distance")
    plt.savefig(os.path.join(outfolder, (img + "_graph.png")))
    plt.clf()


def write_img(img, name, outfolder):
    cv2.imwrite(os.path.join(outfolder, name) + ".png", img)


def process(img, outfolder):
    base = cv2.imread(img)

    obstacles = get_obstacle_regions(base)
    sample, mask = get_sample_region(base)

    thresh = gr.threshold(sample)

    obstacle_contours, _ = get_obstacles(obstacles)
    print("Got obstacles")

    lights_distances, ld, lights_values = get_light_regions(sample, np.bitwise_and(thresh, mask), obstacle_contours)
    print("Got lights")

    darks_distances, dd, darks_values = get_light_regions(sample, np.bitwise_and(np.bitwise_not(thresh), mask), obstacle_contours)
    print("Got darks")

    ds, vs = bin(lights_distances, darks_distances,
                 lights_values, darks_values, ld, dd)
    print("Binned")

    img_name = (os.path.basename(img)).split(".")[0]

    # write to csv file
    with open(os.path.join(outfolder, (img_name + ".csv")), "w") as f:
        for i, d in enumerate(ds):
                f.write(str(d) + "," + str(vs[i]) + "\n")

    scatter(img_name, ds, vs, outfolder)


def main():
    timestr = time.strftime("%m%d%Y_%H%M%S")
    outfolder = os.path.join("runs", "run_{}".format(timestr))
    os.mkdir(outfolder)
    for img in glob.glob('data/test/*.png'):
        process(img, outfolder)
        print("Processed {}".format(img))


if __name__ == "__main__":
    main()
