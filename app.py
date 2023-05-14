import cv2
from modules.object_detection import ObjectDetecion
from modules.classifier import Classifier
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append("./ObjectTracking/")
from lib import VisTrack
from sort import Sort
detector = ObjectDetecion()
classifier = Classifier()




lo_vai = ''

classes = ['loang mau', 'xuoc', 'rach', 'rut soi']

chieu_dai_thuc_te = 27
chieu_rong_thuc_te = 48


width_screen = 0
height_screen = 0
ratio_x = 0
ratio_y = 0


dict_result = {}

sort = Sort(max_age=10, min_hits=3, iou_threshold=0.1)
vt = VisTrack()
def get_center(x_min,y_min,x_max,y_max):
    center_x = (x_max-x_min)/2.0
    center_y = (y_max-y_min)/2.0
    return int(center_x),int(center_y)

def find_boxes(boxes,box_tracker):
    box_tracker = np.array(box_tracker)
    for idx,box in enumerate(boxes):
        box = np.array(box)
        if cosine_similarity(box.reshape(1, -1),box_tracker.reshape(1, -1))>0.9:
            return idx
    return -1
    


def predict(image):
    labels, boxes, scores = detector.detect(image)

    detections_in_frame = len(boxes)
    if detections_in_frame == 0:
        boxes = np.empty((0, 5))
    dets = np.hstack((boxes, scores[:, np.newaxis]))
    res = sort.update(dets)

    boxes_track = res[:, :-1]
    boces_ids = res[:, -1].astype(int)

    for index_box, box in enumerate(boxes_track):
        box = list(map(int, box))
        label = int(labels[find_boxes(boxes,box)])
        
        x_min, y_min, x_max, y_max = box
        x = (x_max-x_min)*ratio_x
        y = (y_max - y_min)*ratio_y
        id = boces_ids[index_box]
        cv2.putText(image, str(id), (x_max,y_min), 0, 1, (255, 0, 0), 2)
        if 0<x<7.5 or 0<y<7.5:
            penatlty = 1
        if 7.5<=x<15.0 or 7.5<=y<15.0:
            penatlty = 2
        if 15.0<=x<23.0 or 15.0<=y<23.0:
            penatlty = 3
        if x>=23.0:
            penatlty= 4
        crop = image[y_min:y_max,x_min:x_max]
        path = './export/'+lo_vai+'/'+str(id)+'.png'
        cv2.imwrite(path,crop)
        dict_result[id] = {'type':classes[label],'width':x,'height':y,'squared':x*y,'penalty':penatlty,'path':path}
        print(dict_result)
        
        
    for idx, box in enumerate(boxes):
        box = list(map(int, box))

        x_min, y_min, x_max, y_max = box
        x = (x_max-x_min)*ratio_x
        y = (y_max - y_min)*ratio_y
        s = x * y
        cv2.putText(image,str(x)[:3]+' cm',(x_min,y_max+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        cv2.putText(image,str(y)[:3]+' cm',(x_max,y_max),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        cv2.putText(image, classes[int(labels[idx])], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 1)
        image = cv2.rectangle(image, (x_min, y_min),
                (x_max, y_max), (0, 0, 255), 3)
    return image
from datetime import datetime
details = ['type','width','height','squared','penalty','path']
def write_file():
    
    dien_tich_loang_mau = 0
    dien_tich_xuoc = 0
    dien_tich_rach = 0
    dien_tich_rut_soi = 0
    
    tong_loang_mau = 0
    tong_xuoc = 0
    tong_rach = 0
    tong_rut_soi = 0
    
    f = open(lo_vai+'.csv','a')
    f.write('Lô vải: '+lo_vai)
    f.write('\nNgày: '+datetime.today().strftime('%Y-%m-%d %H:%M:%S')+'\n')
    f.write('Loại lỗi,Chiều dài,Chiều rộng,Diện tích,Điểm phạt,Đường dẫn ảnh,\n')
    for key in dict_result:
        value =dict_result[key]
        if value['type'] == classes[0]:
            tong_loang_mau +=1
            dien_tich_loang_mau+= value['squared']
        if value['type'] == classes[1]:
            tong_xuoc +=1
            dien_tich_xuoc+= value['squared']
        if value['type'] == classes[2]:
            tong_rach +=1
            dien_tich_rach+= value['squared']
        if value['type'] == classes[3]:
            tong_rut_soi +=1
            dien_tich_rut_soi+= value['squared']    
        for detail in details:
         f.write(str(value[detail])+',')
        f.write('\n')
        
    f.write('Tổng lỗi loang màu: '+str(tong_loang_mau)+'\n')
    f.write('Tổng diện tích loang màu: '+str(dien_tich_loang_mau)+'\n')
    
    f.write('Tổng lỗi xước: '+str(tong_xuoc)+'\n')
    f.write('Tổng diện tích xước: '+str(dien_tich_xuoc)+'\n')
    
    f.write('Tổng lỗi rách: '+str(tong_rach)+'\n')
    f.write('Tổng diện tích rách: '+str(dien_tich_rach)+'\n')
    
    f.write('Tổng lỗi rút sợi: '+str(tong_loang_mau)+'\n')
    f.write('Tổng diện tích rút sợi: '+str(dien_tich_rut_soi)+'\n')
    
    
    
    f.write('Tổng số lỗi: '+str(len(dict_result.keys()))+'\n')
    f.write('Tổng diện tích lỗi: '+ str(dien_tich_rut_soi+dien_tich_loang_mau+dien_tich_rach+dien_tich_rut_soi) +'cm\n')
    f.close()   
            
import os
if __name__ == '__main__':
    lo_vai = input("Nhập lô vải: ")
    if not os.path.exists('./export/'+lo_vai):
        os.mkdir('./export/'+lo_vai)
    
    
    vid = cv2.VideoCapture(0)
    width_screen = vid.get(3)
    height_screen = vid.get(4)
    ratio_x = chieu_rong_thuc_te/width_screen
    ratio_y = chieu_dai_thuc_te/height_screen
    while (True):
        ret, image = vid.read()
        if ret is None:
            break
        image = predict(image)
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
    write_file()
    
    
