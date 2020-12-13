import sys
sys.path.append('../../')

import cv2
import proximal
import os
import numpy as np
import imutils
from proximal import *

debug = False

image_dir = "C:/Users/Kyle/Desktop/Test Burst"

base_im = None
ims = []

# Read images
for filename in os.listdir(image_dir):
    if filename.lower().endswith(".jpeg") or filename.endswith(".png"):
        im = cv2.imread(os.path.join(image_dir, filename))
        if base_im is None:
            print("Base im: " + filename)
            base_im = imutils.resize(im, height=300)
        else:
            ims.append(imutils.resize(im, height=300))

# Find warps to base image
orb = cv2.ORB_create(100)
method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
matcher = cv2.DescriptorMatcher_create(method)
keep_percent = 0.3

warps = []
base_grey = cv2.cvtColor(base_im, cv2.COLOR_BGR2GRAY)
(bg_kps, bg_descript) = orb.detectAndCompute(base_grey, None)

for im in ims:
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    (im_kps, im_descript) = orb.detectAndCompute(im_grey, None)

    matches = matcher.match(bg_descript, im_descript, None)

    matches = sorted(matches, key=lambda x:x.distance)
    keep = int(len(matches) * 0.5)
    matches = matches[:keep]

    if debug:
        matchedVis = cv2.drawMatches(base_im, bg_kps, im, im_kps, matches, None)
        matchedVis = imutils.resize(matchedVis, width=700)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    pts_base = np.zeros((len(matches), 2), dtype=float)
    pts_im = pts_base.copy()

    for (i, match) in enumerate(matches):
        pts_base[i] = bg_kps[match.queryIdx].pt
        pts_im[i] = im_kps[match.trainIdx].pt

    H, _ = cv2.findHomography(pts_base, pts_im, method=cv2.RANSAC)
    warps.append(np.array(H, order='C'))

    if debug:
        aligned = cv2.warpPerspective(base_grey, H, (im.shape[1], im.shape[0]))
        im = cv2.merge((aligned, np.zeros_like(base_grey), im_grey))
        im = imutils.resize(im, height=300)
        cv2.imshow("merged", im)
        cv2.waitKey(0)

ims.append(base_im)
warps.append(np.eye(3, order='C'))

print("Aligned Images")

x = Variable(base_im.shape)

prox_vstack_terms = []
imstack = []
for w, im in zip(warps, ims):
    prox_vstack_terms.append(warp(x, w))
    imstack.append(im.astype(float)/255.0)

fn = vstack(prox_vstack_terms)
out = np.zeros((len(ims)*base_im.shape[0]*base_im.shape[1]*base_im.shape[2]))

fn.forward([np.array(imstack)], [out])

data_term = sum_squares(vstack(prox_vstack_terms) - out)

patch_similarity = patch_NLM(x)
tv = norm1(grad(x))

objective = data_term + patch_similarity + tv

p = Problem(objective)
p.solve(verbose=True)

out = x.value * 255
print(out.max())
print(out.min())
out[out < 255] = 0
out[out > 255] = 255

cv2.imshow("out", out.astype(np.uint8))

out_compress = (out - out.min())/(out.max()-out.min())
out_compress = (out_compress*255).astype(np.uint8)
cv2.imshow("out_compress", out_compress)

cv2.waitKey(0)