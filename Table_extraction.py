from PIL import Image
import cv2
import numpy as np
import pandas as pd 
import numpy as np
import tensorflow as tf
from paddleocr import PaddleOCR, draw_ocr
import os

import cv2
import layoutparser as lp
from pdf2image import convert_from_path






os.makedirs('pages', exist_ok=True)

#loading model
model=lp.PaddleDetectionLayoutModel(config_path='lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config',threshold=0.5,
                                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                                    enforce_cpu=False,
                                    enable_mkldnn=True)

# Text Detection & recognition
# ocr=PaddleOCR(lang='en',det_model_dir='/root/.paddleocr/whl/det/en/en_PP-OCRv4_det_infer')
ocr=PaddleOCR(lang='en')



def read_document():
    images=convert_from_path('khalis2019.pdf')
    for i in range(len(images)):
        images[i].save('pages/page'+str(i)+'.jpg','JPEG')

def layout_detection_mapping(image):
    layout=model.detect(image)
    # Assuming lp.draw_box() returns an image in a format compatible with OpenCV
    show_img = lp.draw_box(image, layout, box_width=3)
    # Convert the Pillow Image to a NumPy array
    show_img = np.array(show_img) # Convert to NumPy array

    # Check and convert the image to BGR if necessary
    if show_img.ndim == 3 and show_img.shape[-1] == 3:
        show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite('result.jpg', show_img)
    return layout

def table_extraction_function(layout):
    # Finding total number of table for the given page
    tables=[l for l in layout if l.type == "Table"]
    
    if len(tables)>=1:
        print(f'Found {len(tables)} Tables in the given page')
        for no,table in enumerate(tables):
            x_1=int(table.block.x_1)
            y_1=int(table.block.y_1)
            x_2=int(table.block.x_2)
            y_2=int(table.block.y_2)
            cv2.imwrite(f'extracted_table_image/extracted_img_{no}.jpg',image[y_1:y_2,x_1:x_2])
    else:
        print(f'Found {0} Tables in the given page')
        return 


def get_images_from_folder(folder_path):
    images = []
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                filepath = os.path.join(folder_path, filename)
                try:
                    images.append(filepath)
                except IOError:
                    print(f"Unable to open image: {filepath}")
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
    return images

def image_extraction(image_path):
    output=ocr.ocr(image_path)
    boxes=[line[0] for line in output[0]]
    texts=[line[1][0] for line in output[0]]
    probabilities=[line[1][1] for line in output[0]]
    return boxes,texts,probabilities
    

def intersection(box_1, box_2):
    return [box_2[0], box_1[1],box_2[2], box_1[3]]

def iou(box_1, box_2):

    x_1 = max(box_1[0], box_2[0])
    y_1 = max(box_1[1], box_2[1])
    x_2 = min(box_1[2], box_2[2])
    y_2 = min(box_1[3], box_2[3])

    inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
    if inter == 0:
        return 0

    box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
    box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

    return inter / float(box_1_area + box_2_area - inter)


if __name__=="__main__":
    #Read the document (only PDF)

    #Read and convert pdf into pages
    read_document()

    #Read page name or image name
    file_name='pages/page2.jpg'    
    image=cv2.imread(file_name)
    image=image[...,::-1]

    
    #detect layout
    layout=layout_detection_mapping(image)

    table_extraction_function(layout)

    images=get_images_from_folder('extracted_table_image')

    
    for no,image_path in enumerate(images):
        image_cv=cv2.imread(image_path)
        image_height=image_cv.shape[0]
        image_width=image_cv.shape[1]
        image_channels=image_cv.shape[2]
        [boxes,texts,probabilities]=image_extraction(image_path)
        image_boxes=image_cv.copy()
        ext=0
        for box,text in zip(boxes,texts):
            cv2.rectangle(image_boxes,(int(box[0][0]),int(box[0][1]),int(box[2][0])-ext,int(box[3][1])-ext),(0,255,255),1)
            cv2.putText(image_boxes,text,(int(box[0][0]),int(box[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(222,0,0),1)
        cv2.imwrite(f'detection_table/Detections_image_boxes_{no}.jpg',image_boxes)

# Reconstructions           

        im=image_cv.copy()
        vert_boxes,horiz_boxes=[],[]

        for box in boxes:
            x_h,x_v=0,int(box[0][0])
            y_h,y_v=int(box[0][1]),0
            width_h,width_v=image_width,int(box[2][0]-box[0][0])
            height_h,height_v=int(box[2][1]-box[0][1]),image_height

            horiz_boxes.append([x_h,y_h,x_h+width_h,y_h+height_h])
            vert_boxes.append([x_v,y_v,x_v+width_v,y_v+height_v])

            cv2.rectangle(im,(x_h,y_h),(x_h+width_h,y_h+height_h),(0,255,0),1)
            cv2.rectangle(im,(x_v,y_v),(x_v+width_v,y_v+height_v),(255,0,0),1)

        # cv2.imwrite(f'HV_image_boxes_{no}.jpg',im)


        horiz_out = tf.image.non_max_suppression(
            horiz_boxes,
            probabilities,
            max_output_size = 1000,
            iou_threshold=0.1,
            score_threshold=float('-inf'),
            name=None
        )
        horiz_lines = np.sort(np.array(horiz_out))
        im_nms = image_cv.copy()

        for val in horiz_lines:
            cv2.rectangle(im_nms, (int(horiz_boxes[val][0]),int(horiz_boxes[val][1])), (int(horiz_boxes[val][2]),int(horiz_boxes[val][3])),(0,0,255),1)
        ## House varying fit
        # cv2.imwrite(f'HV_table_detection/im_nms_{no}.jpg',im_nms)

        vert_out = tf.image.non_max_suppression(
            vert_boxes,
            probabilities,
            max_output_size = 1000,
            iou_threshold=0.1,
            score_threshold=float('-inf'),
            name=None
        )
        vert_lines = np.sort(np.array(vert_out))
        for val in vert_lines:
            cv2.rectangle(im_nms, (int(vert_boxes[val][0]),int(vert_boxes[val][1])), (int(vert_boxes[val][2]),int(vert_boxes[val][3])),(255,0,0),1)
        cv2.imwrite(f'HV_table_detection/im_nms_{no}.jpg',im_nms)

        out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]
        unordered_boxes = []
        for i in vert_lines:
            unordered_boxes.append(vert_boxes[i][0])
        ordered_boxes = np.argsort(unordered_boxes)

        for i in range(len(horiz_lines)):
            for j in range(len(vert_lines)):
                resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]] )

                for b in range(len(boxes)):
                    the_box = [boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
                    if(iou(resultant,the_box)>0.1):
                        out_array[i][j] = texts[b]

        import pandas as pd
        pd.DataFrame(out_array).to_csv(f'CSV_dataout/sample{no}.csv')
        print('Finished')