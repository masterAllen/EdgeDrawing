import os
os.chdir(os.path.dirname(__file__))

import cv2
import numpy as np
import collections
import time

EDGE_VER, EDGE_HOR, EDGE_IMPOSSIBLE = 0, 1, 2

img = cv2.imread('lena.jpg', -1)
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = img.shape

# 1. Blur Image
blursize = 5
blursigma = 1.0
img = cv2.GaussianBlur(img, (blursize, blursize), blursigma)

# 2. Compute Gradient
gradient_thresh = 36
dx = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
dy = np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))

Gradient_value = dx + dy
Gradient_direction = np.where(dx > dy, EDGE_VER, EDGE_HOR)
Gradient_direction[Gradient_value < gradient_thresh] = EDGE_IMPOSSIBLE

# For convenience, we add a one-pixel boundary around the entire structure.
Gradient_direction = np.pad(Gradient_direction, ((1, 1), (1, 1)), mode='constant', constant_values=EDGE_IMPOSSIBLE)
Gradient_value = np.pad(Gradient_value, ((1, 1), (1, 1)), mode='constant', constant_values=0)
rows, cols = rows+2, cols+2

# 3. Get Anchor Points
anchor_thresh = 8
anchor_step = 4

anchors = []
for i in range(2, rows-2, anchor_step):
    for j in range(2, cols-2, anchor_step):
        isanchor = False
        if Gradient_direction[i, j] == EDGE_HOR:
            mask1 = Gradient_value[i, j] - Gradient_value[i-1, j] >= anchor_thresh
            mask2 = Gradient_value[i, j] - Gradient_value[i+1, j] >= anchor_thresh
            isanchor = mask1 and mask2
        elif Gradient_direction[i, j] == EDGE_VER:
            mask1 = Gradient_value[i, j] - Gradient_value[i, j-1] >= anchor_thresh
            mask2 = Gradient_value[i, j] - Gradient_value[i, j+1] >= anchor_thresh
            isanchor = mask1 and mask2

        if isanchor:
            anchors.append((i, j))
        
anchors = sorted(anchors, key=lambda x: Gradient_value[x[0], x[1]], reverse=True)


# 4. Trace Edges from anchors
visited = np.zeros((rows, cols), dtype=bool)

# For dynamic dispaly
edges = np.zeros((rows, cols), dtype=np.uint8)

img = np.zeros((rows, cols, 3), dtype=np.uint8)
result = []

# Use DFS to trace from one anchor point to another anchor point
def visit(x, y):
    global visited, edges

    def getmaxG(p1, p2, p3):
        G1, G2, G3 = Gradient_value[p1], Gradient_value[p2], Gradient_value[p3]
        if G1 >= G2 and G1 >= G3:
            return p1
        elif G3 >= G2 and G3 >= G1:
            return p3
        else:
            return p2

    stacks = collections.deque([((x, y), None)])
    while len(stacks) > 0:
        nowp, fromdirection = stacks.pop()
        nowx, nowy = nowp

        if Gradient_direction[nowx, nowy] == EDGE_IMPOSSIBLE:
            continue
        if visited[nowx, nowy]:
            continue

        visited[nowx, nowy] = True

        # For dynamic display
        edges[nowx, nowy] = 255
        cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
        cv2.imshow('Edges', edges)
        cv2.waitKey(1)

        if Gradient_direction[nowx, nowy] == EDGE_HOR:
            # Go Left
            if fromdirection != 'RIGHT':
                x1, y1 = nowx-1, nowy-1
                x2, y2 = nowx,   nowy-1
                x3, y3 = nowx+1, nowy-1
                newp = getmaxG((x1, y1), (x2, y2), (x3, y3))
                stacks.append((newp, 'LEFT'))

            # Go Right
            if fromdirection != 'LEFT':
                x1, y1 = nowx-1, nowy+1
                x2, y2 = nowx,   nowy+1
                x3, y3 = nowx+1, nowy+1
                newp = getmaxG((x1, y1), (x2, y2), (x3, y3))
                stacks.append((newp, 'RIGHT'))

        elif Gradient_direction[nowx, nowy] == EDGE_VER:
            # Go Up
            if fromdirection != 'DOWN':
                x1, y1 = nowx-1, nowy-1
                x2, y2 = nowx-1, nowy
                x3, y3 = nowx-1, nowy+1 
                newp = getmaxG((x1, y1), (x2, y2), (x3, y3))
                stacks.append((newp, 'UP'))

            # Go Down   
            if fromdirection != 'UP':
                x1, y1 = nowx+1, nowy-1
                x2, y2 = nowx+1, nowy
                x3, y3 = nowx+1, nowy+1 
                newp = getmaxG((x1, y1), (x2, y2), (x3, y3))
                stacks.append((newp, 'DOWN'))


for anchorx, anchory in anchors:
    if not visited[anchorx, anchory]:
        visit(anchorx, anchory)

cv2.imwrite('./edges.png', edges)