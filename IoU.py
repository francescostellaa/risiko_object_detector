
#Convert boxes from label format to find the upper left and
#and lower right corner of the bounding box
def convert_bbox(center_x, center_y, width_scale, height_scale):
    x1 = float(center_x - (width_scale / 2))
    y1 = float(center_y - (height_scale / 2))
    x2 = float(center_x + (width_scale / 2))
    y2 = float(center_y + (height_scale / 2))
    return [x1, y1, x2, y2]

#Calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    # Calculate intersection
    x1_inter = float(max(box1[0], box2[0]))
    y1_inter = float(max(box1[1], box2[1]))
    x2_inter = float(min(box1[2], box2[2]))
    y2_inter = float(min(box1[3], box2[3]))

    #calculate the area of the intersection    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    #calculate the area of each box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    #area of the union
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = float(inter_area / union_area) if union_area != 0 else 0
    return iou

#Match each box with the most similar of the true labels
def eval_bboxes(pred_bboxes, true_bboxes, true_labels):
    matched_labels = []
    for pred_bbox in pred_bboxes:
        best_iou = 0
        best_label = -1
        for i, true_bbox in enumerate(true_bboxes):
            iou = calculate_iou(pred_bbox, true_bbox)
            #If they don't overlap for at least 70% of their 
            #area, they are not the same. Can be changed
            if iou > 0.70:
                if iou > best_iou:
                    best_iou = iou
                    best_label = true_labels[i]
        matched_labels.append(best_label)
    return matched_labels
