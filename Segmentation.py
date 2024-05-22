import cv2
import os
import numpy as np

from typing import List, Tuple, Dict


class Pixel:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        
def main():
    folder_path = "C:/Users/giuli/OneDrive/Desktop/Risiko Py/Risiko_images"

    images = []
    images = loading_images(folder_path)

    # Normalizza le immagini
    normalized_images = []
    for img in images:
        new_width = 900
        new_height = 600
        output_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        normalized_images.append(output_image)

    # Segmenta le immagini
    segm_images = []
    for img in normalized_images:
        channels = cv2.split(img)
        channels_elaborated = np.zeros((3,600))
        #print(type(channels_elaborated))
        for i in range(len(channels)):
            _, channels_elaborated = cv2.threshold(channels[i], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        output_image = cv2.merge(channels)
        segm_images.append(output_image)

    # Eseguiamo il region growing sull'immagine
    classes, cluster_size = region_growing(segm_images[1])
    print_image_with_colors(classes, segm_images[1])

    filter_region_growing(cluster_size)

    # Filtro derivata seconda per contorni
    derivative_x = cv2.Sobel(segm_images[1], cv2.CV_32F, 2, 0)
    derivative_y = cv2.Sobel(segm_images[1], cv2.CV_32F, 0, 2)

    gradient_magnitude = cv2.magnitude(derivative_x, derivative_y)
    show_image("Double Derivative Image", gradient_magnitude)

    double_derivative_8bit = cv2.convertScaleAbs(gradient_magnitude)

    # Salvare le immagini
    folder_save_path = "C:/Users/Alessandro Di Frenna/OneDrive/Desktop/clustering_risiko/salved_images/"
    for i, img in enumerate(segm_images):
        save_path = os.path.join(folder_save_path, f"image_{i}.jpg")
        cv2.imwrite(save_path, img)
        print(f"Immagine {i + 1} salvata con successo")
    
    
def region_growing(image):
    rows, cols = image.shape[:2]
    visited = [[False] * cols for _ in range(rows)]
    classes = [[-1] * cols for _ in range(rows)]
    cluster_size = {}
    
    current_class = 0
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j]:
                q = []
                q.append((i, j))
                visited[i][j] = True
                classes[i][j] = current_class
                
                count = 1
                
                while not len(q)==0:
                    pixel = q[0]
                    q.pop(0)
                    
                    for k in range(4):
                        nx, ny = pixel[0] + dx[k], pixel[1] + dy[k]
                        
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and np.all(image[nx, ny] == image[i, j]):
                            q.append((nx, ny))
                            visited[nx][ny] = True
                            classes[nx][ny] = current_class
                            count += 1
                
                cluster_size[str(current_class)] = count
                current_class += 1
    
    print(f"IL NUMERO DI CLUSTER E: {current_class}")
    return classes, cluster_size


def process_general(function, input_images: list[np.ndarray]):
    for image in input_images:
        function(image)
        
def print_image_with_colors(classes, original_image):
    color_map = {}
    
    max_class = max(max(row) for row in classes)
    rng = np.random.default_rng(0)
    
    for i in range(max_class + 1):
        color_map[i] = rng.integers(0, 256, size=3, dtype=np.uint8)
    
    result = original_image.copy()
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = color_map[classes[i][j]]
    
    show_image("Region Growing Result", result)
    

def filter_region_growing(cluster_name_clust_size):
    filtered_clusters = {}
    print(f"NUMERO CLASSI INIZIALI: {len(cluster_name_clust_size)}")
    
    for id_cluster, count in cluster_name_clust_size.items():
        if 15 < count < 200:
            filtered_clusters[id_cluster] = count
    
    cluster_name_clust_size.clear()
    cluster_name_clust_size.update(filtered_clusters)
    
    print(f"NUMERO CLASSI FINALI: {len(cluster_name_clust_size)}")

def loading_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)
            if image is not None:
                images.append(image)
            else:
                print(f"Error loading image file: {file_path}")
    return images

def check_loading_images(image_paths):
    images = []
    try:
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f"Error loading image file: {image_path}")
            images.append(image)
    except RuntimeError as e:
        print(f"Exception caught: {e}")
        exit(1)
    return images

def show_image(name_image, image):
    cv2.imshow(name_image, image)
    cv2.waitKey(0)
    cv2.destroyWindow(name_image)


def saving_image(folder_path, salved_images):
    for i, image in enumerate(salved_images):
        file_path = os.path.join(folder_path, f"image_{i}.jpg")
        cv2.imwrite(file_path, image)
        print(f"immagine {i + 1} salvata con successo")
        
        
if __name__ == "__main__":
    main()
