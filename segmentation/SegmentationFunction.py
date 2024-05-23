import cv2
import os
import numpy as np
from collections import deque

#To print the bboxes
# for key in cluster_info.keys():
#         point = cluster_info[key][0]
#         width = cluster_info[key][1]
#         height = cluster_info[key][2]
#         cv2.rectangle(region_growing_images[i],(point[0], point[1]),(point[0]+width, point[1]+height),(255,0,0),2)
    

def threshold(image):
    channels = cv2.split(image)
    channels_elaborated = []
    for i in range(len(channels)):
        _, thresholded_channel = cv2.threshold(channels[i], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        channels_elaborated.append(thresholded_channel)
    output_image = cv2.merge(channels_elaborated)
    return output_image

def region_growing(image):
    #apply the threshold
    image = threshold(image)
    rows, cols = image.shape[:2]
    visited = np.zeros((rows, cols), dtype=bool)
    classes = np.ones((rows, cols), dtype=int)
    cluster_info = {}
    current_class = 0
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    for i in range(rows):
        for j in range(cols):
            if not visited[i, j]:
                q = deque([(i, j)])
                visited[i, j] = True
                classes[i, j] = current_class

                min_x, max_x = i, i
                min_y, max_y = j, j
                cluster_pixels = []

                while q:
                    x, y = q.popleft()
                    cluster_pixels.append((x, y))
                    #check up, left, bottom, right pixels
                    for k in range(4):
                        nx, ny = x + dx[k], y + dy[k]
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and np.array_equal(image[nx, ny], image[i, j]):
                            q.append((nx, ny))
                            visited[nx, ny] = True
                            classes[nx, ny] = current_class
                            min_x, max_x = min(min_x, nx), max(max_x, nx)
                            min_y, max_y = min(min_y, ny), max(max_y, ny)

                #Convert cluster_pixels to a numpy array at the end
                cluster_pixels = np.array(cluster_pixels)

                width = max_x - min_x + 1
                height = max_y - min_y + 1
                count = cluster_pixels.shape[0]

                if width==1 or height==1:
                    for pair in cluster_pixels:
                        x,y = pair
                        classes[x,y] = -1

                #cluster info contains the upper left corner,
                # the width and the height of the rectangle
                if count <= 10000 and count >= 200:
                    cluster_info[current_class] = [(min_y, min_x), width, height]
                    current_class += 1
                else:
                    for pair in cluster_pixels:
                        x,y = pair
                        classes[x,y] = -1

    return classes, cluster_info