import cv2
import os
import numpy as np
from collections import deque
        
def main():
    folder_path = "../real_images/images"
    filename = "000000.jpg"
    images = []
    #images = loading_images(folder_path)
    path = os.path.join(folder_path, filename)
    image = cv2.imread(path)
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
    #classes, cluster_size, cluster_info = region_growing(segm_images[0])
    classes, cluster_info = region_growing(segm_images[0])
    print("Number of classes:", len(classes))
    #print("Cluster size: ", len(cluster_size))
    #print("Cluster info:", cluster_info)
    for key in cluster_info.keys():
        print(cluster_info[key]["height"])
    region_growing_images = []
    region_growing_images.append(image_with_colors(classes, segm_images[0]))
    #cv2.imshow("Region growing after small cluster ssuppression", region_growing_images[-1])

    #filter_region_growing(cluster_size)
    #print("Cluster info:", cluster_info[35])
    #print("Cluster size: ", len(cluster_size))

    # Filtro derivata seconda per contorni
    # derivative_x = cv2.Sobel(segm_images[0], cv2.CV_32F, 2, 0)
    # derivative_y = cv2.Sobel(segm_images[0], cv2.CV_32F, 0, 2)
    for key in cluster_info.keys():
        point = cluster_info[key]["upper_left"]
        width = cluster_info[key]["width"]
        height = cluster_info[key]["height"]
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
    

# def region_growing(image):
#     rows, cols = image.shape[:2]
#     visited = np.zeros((rows, cols), dtype=bool)
#     classes = np.full((rows, cols), -1, dtype=int)
#     #cluster_size = number of points for each cluster
#     cluster_size = {}
#     cluster_details = {}
#     #curent class = number of clusters
#     current_class = 0
#     dx = [1, 0, -1, 0]
#     dy = [0, 1, 0, -1]
    
#     for i in range(rows):
#         for j in range(cols):
#             if not visited[i, j]:
#                 # q = deque()
#                 # q.append((i, j))
#                 q = deque([(i, j)])
#                 visited[i, j] = True
#                 classes[i, j] = current_class
                
#                 count = 1
                
#                 #def mask
#                 mask = []
#                 mask.append((i,j))

#                 # max_x, min_x = i, i
#                 # max_y, min_y = j, j
#                 max_x = float(-1)
#                 max_y = float(-1)
#                 min_x = float("inf")
#                 min_y = float("inf")

#                 while q:
#                     x, y = q.popleft()
                    
#                     for k in range(4):
#                         nx, ny = x + dx[k], y + dy[k]
                        
#                         #Check boarders, if it's visited and if all channels of the neighboor are
#                         #are equivalent to the ones of the current class
#                         if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and np.all(image[nx, ny] == image[i, j]):
#                             q.append((nx, ny))
#                             visited[nx, ny] = True
#                             classes[nx, ny] = current_class
#                             min_x, max_x = min(min_x, nx), max(max_x, nx)
#                             min_y, max_y = min(min_y, ny), max(max_y, ny)
#                             count += 1

#                             # mask.append((nx, ny))

#                             # if nx > max_x: max_x = nx
#                             # if nx < min_x: min_x = nx
#                             # if ny > max_y: max_y = ny
#                             # if ny < min_y: min_y = ny

#                 #width = max_y - min_y + 1
#                 width = max_y - min_y
#                 if width==0:
#                     print("Width = 0")
#                 #height = max_x - min_x + 1
#                 height = max_x - min_x
#                 if height==0:
#                     print("height = 0")

#                 # Note the size of this cluster
#                 cluster_size[current_class] = count
#                 if width == 0 or height == 0 or count > 40000 or count < 130:
#                 #if count > 35000 or count < 600 :  # If it does not meet the requirements
#                     #or width/height<0.7 or height/width<0.7
#                     for nx, ny in mask:
#                         classes[nx, ny] = -1
#                 else:
#                     cluster_details[current_class] = {
#                         "upper_left": (min_x, min_y),
#                         "width": width,
#                         "height": height
#                     }
#                     current_class += 1  # Advance to the next class

#     print(f"The number of clusters is: {current_class}")
#     return classes, cluster_size, cluster_details



def region_growing(image):
    rows, cols = image.shape[:2]
    visited = np.zeros((rows, cols), dtype=bool)
    classes = -np.ones((rows, cols), dtype=int)
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

                width = max_x - min_x + 1
                height = max_y - min_y + 1
                count = len(cluster_pixels)

                if width==1:
                    for px, py in cluster_pixels:
                        classes[px, py] = -1
                if height==1:
                    for px, py in cluster_pixels:
                        classes[px, py] = -1

                # Debug information
                print(f"Cluster {current_class}: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}, width={width}, height={height}, count={count}")

                if count <= 30000 and count >= 200:
                    cluster_info[current_class] = {
                        "upper_left": (min_y, min_x),
                        "width": width,
                        "height": height,
                    }
                    current_class += 1
                else:
                    for px, py in cluster_pixels:
                        classes[px, py] = -1

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
            # else:
            #     result[i, j] = (0,0,0)
    return result
    

#def filter_region_growing(cluster_name_clust_size):
def filter_region_growing(cluster_size):
    filtered_clusters = {}
    #print(f"NUMERO CLASSI INIZIALI: {len(cluster_size)}")
    
    for id_cluster, count in cluster_size.items():
        if 15 < count < 200:
            filtered_clusters[id_cluster] = count
    
    cluster_size.clear()
    cluster_size.update(filtered_clusters)
    #print(f"NUMERO CLASSI FINALI: {len(cluster_size)}")
        
        
if __name__ == "__main__":
    main()
