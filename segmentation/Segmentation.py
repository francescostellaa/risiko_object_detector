import cv2
import os
import numpy as np
from collections import deque
        
def main():
    folder_path = "../real_images/images"
    filename = "000040.jpg"
    images = []
    #images = loading_images(folder_path)
    path = os.path.join(folder_path, filename)
    image = cv2.imread(path)
    cv2.imshow("RGB", image)
    output_image = cv2.resize(image, (900, 600), interpolation=cv2.INTER_LINEAR)
    normalized_images = []
    normalized_images.append(output_image)
    # Normalizza le immagini
    # normalized_images = []
    # for img in images:
    #     new_width = 900
    #     new_height = 600
    #     output_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    #     normalized_images.append(output_image)
    cv2.imshow("First step", normalized_images[0])
    
    # Apply threshold to the images
    segm_images = []
    for img in normalized_images:
        channels = cv2.split(img)
        channels_elaborated = []
        for i in range(len(channels)):
            _, thresholded_channel = cv2.threshold(channels[i], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            channels_elaborated.append(thresholded_channel)
        cv2.merge(channels_elaborated)
        output_image = cv2.merge(channels_elaborated)
        segm_images.append(output_image)

    # Eseguiamo il region growing sull'immagine
    classes, cluster_info = region_growing(segm_images[0])
    print("Number of classes:", len(classes))

    region_growing_images = []
    region_growing_images.append(image_with_colors(classes, segm_images[0]))

    for key in cluster_info.keys():
        point = cluster_info[key][0]
        width = cluster_info[key][1]
        height = cluster_info[key][2]
        cv2.rectangle(region_growing_images[-1],(point[0], point[1]),(point[0]+width, point[1]+height),(255,0,0),2)
    
    cv2.imshow("Rectangles",region_growing_images[-1])

    #gradient_magnitude = cv2.magnitude(derivative_x, derivative_y)
    #cv2.imshow("Double Derivative Image", gradient_magnitude)

    cv2.waitKey(0)
    # Salvare le immagini
    folder_save_path = "output_segmentation"
    for img in region_growing_images:
        save_path = os.path.join(folder_save_path, f"image_{i}.jpg")
        cv2.imwrite(save_path, img)
    

def region_growing(image):
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
                if count <= 10000 and count >= 300:
                    cluster_info[current_class] = [(min_y, min_x), width, height]
                    current_class += 1
                else:
                    for pair in cluster_pixels:
                        x,y = pair
                        classes[x,y] = -1

    print(f"Number of clusters: {current_class}")
    return classes, cluster_info

def image_with_colors(classes, original_image):
    color_map = {}
    
    max_class = max(max(row) for row in classes)
    rng = np.random.default_rng(0)
    
    for i in range(max_class + 1):
        color_map[i] = rng.integers(0, 256, size=3, dtype=np.uint8)
    
    result = original_image.copy()
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):

            if classes[i, j] != -1:
                result[i, j] = color_map[classes[i][j]]
    return result
        
        
if __name__ == "__main__":
    main()
