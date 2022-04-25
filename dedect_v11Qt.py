# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:44:36 2021

@author: gregor
"""
import sys, os, time, math
import json
import cv2 
import numpy as np
import tensorflow as tf
#from threading import Thread, Lock
from sensecam_control import vapix_control
#from sensecam_control import onvif_control
from datetime import datetime
from darkflow.net.build import TFNet
#from random import randint
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from filters import Cartoonizer
from filters import WarmingFilter
from filters import CoolingFilter
from PyQt5.QtCore import (QDateTime, QObject, QSizeF, QLineF, QPoint, QPointF, QRectF, QFile, QMargins, Qt, QUrl, QAbstractItemModel, QModelIndex, QIODevice, QItemSelectionModel,
                          QThread, QMutex, QTimer, pyqtSignal, pyqtSlot)
from PyQt5.QtGui import (QPalette, QFont, QPolygon, QPolygonF, QPainterPath, QPainter, QBrush, QPen, QColor, QIcon, QImage, QPixmap, QStandardItemModel, QStandardItem)
#from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import (QStyle, QStyleOption, QApplication, QWidget, QPlainTextEdit, QToolTip, QDialog, QDockWidget, QToolBar, QLineEdit, QToolButton, 
                             QCheckBox, QPushButton, QGroupBox, QScrollArea, QTreeView, QAbstractItemView,
                             QFileDialog, QMainWindow, QMessageBox, QStatusBar, QMdiArea, QMdiSubWindow, 
                             QAction, QHBoxLayout, QVBoxLayout, QLabel)

# pyrcc5 dedect.qrc -o dedect_rc.py
import dedect_rc

#I found an answer on another forum. I change the line number 369 in the Python\Lib\site-packages\Pyinstaller\compat.py file:
# out = out.decode(encoding)
# to out = out.decode(encoding, errors='ignore')
# or out = out.decode(encoding, "replace")
# fixed in: pip install "pyinstaller>=4.0.0"
# pyinstaller --onefile dedect_v06Qt.py
# pyinstaller -F --noupx --log-level=WARN --additional-hooks-dir=hooks --onefile --windowed --icon="images\logo.ico" dedect_v08Qt.py
# pyinstaller -F --noupx --log-level=WARN --additional-hooks-dir=hooks --windowed --clean --onefile -i"images\logo.ico" dedect_v08Qt.py

ip = '192.168.1.25'
login = 'myftp'
password = ''

#url http login axis camera
#URL = 'http://' + login + ':' + password + '@' + ip + '/mjpg/1/video.mjpg?'
URL = 'rtsp://' + login + ':' + password + '@' + ip + '/axis-media/media.amp'
#http://admin:654321@192.168.1.32/videostream.cgi

def bb_intersection_selection(boxA,boxList):
    flag = True
    for boxB in boxList:
        if bb_intersection_over_union(boxA,boxB)>0.3:
            flag=False
    return flag

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if (xB - xA + 1) <= 0 or (yB - yA + 1) <= 0:
      return 0
      
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

class MQObject(QObject):
    message = pyqtSignal(str)
    def __init__(self, p_parent=None):
        super(MQObject, self).__init__(p_parent)
        
    ######################################################
    # Functions called by Child Thread
    ######################################################
    ######################################################
    # Nothing is done 
    ######################################################
    def thread_program_filter_none(self, p_frame, p_args=None):
        return p_frame,False

    ######################################################
    # Cartoon Filter 
    ######################################################
    def thread_program_filter_cartoon_run(self, p_frame, p_args=None):
        return Cartoonizer().render(p_frame),False
    
    ######################################################
    # Warming Filter 
    ######################################################
    def thread_program_filter_warming_run(self, p_frame, p_args=None):
        return WarmingFilter().render(p_frame),False
    
    ######################################################
    # Cooling Filter 
    ######################################################
    def thread_program_filter_cooling_run(self, p_frame, p_args=None):
        return CoolingFilter().render(p_frame),False

    ######################################################
    # Kmeans Filter 
    ######################################################
    def thread_program_kmeans_tf_init(self, p_frame, p_args=None):
        img = np.array(p_frame, dtype=np.float32) / 255.0
        #img = p_frame
        # Load Image and transform to a 2D numpy array.
        w, h, d = tuple(img.shape)
        assert d == 3
        image_array = np.reshape(img, (w * h, d))
        #num_points, dimensions = tuple(image_array.shape)
    
        def input_fn():
            return tf.compat.v1.train.limit_epochs(
                tf.convert_to_tensor(image_array, dtype=tf.float32), num_epochs=1)
    
        num_clusters = p_args[0]
        kmeans = tf.compat.v1.estimator.experimental.KMeans(
            num_clusters=num_clusters, use_mini_batch=False)
    
        # train
        num_iterations = 10
        previous_centers = None
        for _ in range(num_iterations):
            kmeans.train(input_fn)
            cluster_centers = kmeans.cluster_centers()
            if previous_centers is not None:
                print ('delta:', cluster_centers - previous_centers)
            previous_centers = cluster_centers
            print ('score:', kmeans.score(input_fn))
        print ('cluster centers:', cluster_centers)
        return [p_frame, kmeans]
    
    def thread_program_kmeans_tf_run(self, p_frame, p_args=None):
        img = np.array(p_frame, dtype=np.float32) / 255.0
        #img = p_frame
        # Load Image and transform to a 2D numpy array.
        w, h, d = tuple(img.shape)
        assert d == 3
        image_array = np.reshape(img, (w * h, d))
        def input_fn():
            return tf.compat.v1.train.limit_epochs(
                tf.convert_to_tensor(image_array, dtype=tf.float32), num_epochs=1)
        # map the input points to their clusters
        cluster_centers = p_args[1].cluster_centers()
        cluster_indices = list(p_args[1].predict_cluster_index(input_fn))
        img = np.zeros((w*h, d), np.uint8)
        for i, point in enumerate(image_array):
            cluster_index = cluster_indices[i]
            center = cluster_centers[cluster_index]
            img[i] = center*255
        #    #print ('point:', point, 'is in cluster', cluster_index, 'centered at', center)
        ret = np.reshape(img,(w,h,d))
        return ret,False
    
    ######################################################
    # Kmeans Filter 
    ######################################################
    def thread_program_kmeans_init(self, p_frame, p_args):
        def recreate_image(codebook, labels, w, h):
            """Recreate the (compressed) image from the code book & labels"""
            d = codebook.shape[1]
            image = np.zeros((w, h, d), np.uint8)
            label_idx = 0
            for i in range(w):
                for j in range(h):
                    image[i][j] = codebook[labels[label_idx]]*255
                    label_idx += 1        
            return image
        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        img = np.array(p_frame, dtype=np.float64) / 255
        #img = p_frame
        # Load Image and transform to a 2D numpy array.
        w, h, d = tuple(img.shape)
        assert d == 3
        image_array = np.reshape(img, (w * h, d))
    
        print("Fitting model on a small sub-sample of the data")
        t0 = time.time()
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=p_args[0], random_state=0).fit(image_array_sample)
        codebook_random = shuffle(image_array, random_state=0)[:p_args[0]]
        print("done in %0.3fs." % (time.time() - t0))
    
        return [p_frame, recreate_image, kmeans, codebook_random, p_args[1], p_args[0]]
    
    def thread_program_kmeans_run(self, p_frame, p_args):
       
        img = np.array(p_frame, dtype=np.float64) / 255
        #img = p_frame
        # Load Image and transform to a 2D numpy array.
        w, h, d = tuple(img.shape)
        assert d == 3
        image_array = np.reshape(img, (w * h, d))
        
        # Get labels for all points
        #print("Predicting color indices on the full image (k-means)")
        #t0 = time.time()
        labels = p_args[2].predict(image_array)
        #print("done in %0.3fs." % (time.time() - t0))
    
        
        #print("Predicting color indices on the full image (random)")
        #t0 = time.time()
        labels_random = pairwise_distances_argmin(p_args[3],
                                                  image_array,
                                                  axis=0)
        #print("done in %0.3fs." % (time.time() - t0))
        if p_args[4]:
            #print('Quantized image ({} colors, K-Means)'.format(p_args[5]))
            img = p_args[1](p_args[2].cluster_centers_, labels, w,h)
        else:
            #print('Quantized image ({} colors, Random)'.format(p_args[5]))
            img = p_args[1](p_args[3], labels_random, w,h)
        return img,False
    
    ######################################################
    # Motion Simple Background 
    ######################################################
    def thread_program_motion_simple_init(self, p_frame, p_args):
        kernelDil = np.ones((p_args[3],p_args[3]))
        fgbg = cv2.createBackgroundSubtractorMOG2(history=p_args[0], varThreshold=p_args[1], detectShadows=p_args[2])
        return [p_frame, fgbg, kernelDil]
    
    def thread_program_motion_simple_run(self, p_frame,p_args):
        msk_motion = p_args[1].apply(p_frame)
        msk_motion = cv2.morphologyEx(msk_motion,cv2.MORPH_OPEN,p_args[2],1)
        # compute connected components for motion mask
        msk_motion_cc = cv2.connectedComponentsWithStats(msk_motion, 8, cv2.CV_32S)
        cc_n     = msk_motion_cc[0]
        cc_stats = msk_motion_cc[2]
        if cc_n>1:
            msk_motion_lx1 = min(cc_stats[1:,cv2.CC_STAT_LEFT])
            msk_motion_ly1 = min(cc_stats[1:,cv2.CC_STAT_TOP])
            msk_motion_lx2 = max(cc_stats[1:,cv2.CC_STAT_LEFT]+cc_stats[1:,cv2.CC_STAT_WIDTH])
            msk_motion_ly2 = max(cc_stats[1:,cv2.CC_STAT_TOP]+cc_stats[1:,cv2.CC_STAT_HEIGHT])
            cv2.rectangle(p_frame,(msk_motion_lx1,msk_motion_ly1),(msk_motion_lx2,msk_motion_ly2),(0,255,0),3)
        for cc_i in range(1,cc_n):
            msk_motion_lx = cc_stats[cc_i,cv2.CC_STAT_LEFT]
            msk_motion_ly = cc_stats[cc_i,cv2.CC_STAT_TOP]
            msk_motion_lw = cc_stats[cc_i,cv2.CC_STAT_WIDTH]
            msk_motion_lh = cc_stats[cc_i,cv2.CC_STAT_HEIGHT]
            cv2.rectangle(p_frame,(msk_motion_lx,msk_motion_ly),(msk_motion_lx+msk_motion_lw,msk_motion_ly+msk_motion_lh),(255,0,255),3)
        p_frame[:,:,0] = msk_motion
        if cc_n>1:
            self.message.emit("SM; Motion detected!\n")
            return p_frame,True
        return p_frame,False
    
    ######################################################
    # Motion Lukas-Kanade
    ######################################################
    def thread_program_motion_lk_init(self, p_frame, p_args):
        # Parameters for ShiTomasi corner detection
        feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=7, blockSize=7)
        # Parameters for Lucas Kanade optical flow
        lk_params = dict(
            winSize=(50,50),
            maxLevel=1,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
        # Create random colors
        color = np.random.randint(0, 255, (500, 3))
        # Take first frame and find corners in it
        old_gray = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
    
        return [p_frame, lk_params, color, old_gray, feature_params]
            
    def thread_program_motion_lk_run(self, p_frame, p_args):
        p0 = cv2.goodFeaturesToTrack(p_args[3], mask=None, **p_args[4])
        #grid_y, grid_x = np.mgrid[0:old_gray.shape[0]:1, 0:old_gray.shape[1]:1]
        #p0 = np.stack((grid_x.flatten(),grid_y.flatten()),axis=1).astype(np.float32)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(p_frame)
        new_gray = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(p_args[3], new_gray, p0, None, **p_args[1])
        #flow = np.reshape(p1 - p0, (p_args[3].shape[0], p_args[3].shape[1], 2))
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # Draw the tracks
        ret_value = False
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            print(new,old)
            #a, b = new.ravel()
            #c, d = old.ravel()
            if np.sqrt(np.sum((new-old)**2)) >= 1.0:
                mask = cv2.line(mask, (new[0], new[1]), (old[0], old[1]), p_args[2][i].tolist(), 2)
                ret_value = True
                self.message.emit("LK; Motion detected!\n")
            p_frame = cv2.circle(p_frame, (new[0], new[1]), 5, p_args[2][i].tolist(), -1)
        # Display the demo
        img = cv2.add(p_frame, mask)
        # Update the previous frame and previous points
        p_args[3] = new_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        return img,ret_value
    
    ######################################################
    # Motion Dence-Flow
    ######################################################
    def thread_program_motion_df_init(self, p_frame, p_args):
        if p_args[0]:
            method = cv2.calcOpticalFlowFarneback
            params = [0.5, 5, 15, 3, 5, 1.2, 0]  # default Farneback's algorithm parameters [0.5, 3, 10, 6, 7, 1.5, 0]
        else:
            method = cv2.optflow.calcOpticalFlowSparseToDense
            params = [16, 128, 0.05, True, 500.0, 1.5]
        #method = cv2.calcOpticalFlowDenseRLOF
        #params = []
        # crate HSV & make Value a constant
        hsv = np.zeros_like(p_frame, np.float32)
        hsv[..., 1] = 1.0
        # Preprocessing for exact method
        if p_args[0]:
            old_frame = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
        else:
            old_frame = p_frame.copy()
        #kernel = np.ones((5,5),np.float32)/25
        #old_frame = cv2.filter2D(old_frame,-1,kernel)
        #old_frame = cv2.blur(old_frame,(5,5))
        #old_frame = cv2.GaussianBlur(old_frame,(5,5),0)
        #old_frame = cv2.medianBlur(old_frame,5)
        #old_frame = cv2.bilateralFilter(old_frame,9,75,75)
        return [p_frame, method, params, old_frame, hsv, p_args[0]]        
    
    def thread_program_motion_df_run(self, p_frame, p_args):
        # Read the next frame
        # Preprocessing for exact method
        if p_args[5]:
            new_frame = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
        else:
            new_frame = p_frame.copy()
        #kernel = np.ones((5,5),np.float32)/25
        #new_frame = cv2.filter2D(new_frame,-1,kernel)
        #new_frame = cv2.blur(new_frame,(5,5))
        #new_frame = cv2.GaussianBlur(new_frame,(5,5),0)
        #new_frame = cv2.medianBlur(new_frame,5)
        #new_frame = cv2.bilateralFilter(new_frame,9,75,75)
        # Calculate Optical Flow
        flow = p_args[1](p_args[3], new_frame, None, *p_args[2])
        #print(flow.shape)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        # Use Hue and Value to encode the Optical Flow
        #p_args[4][..., 0] = ang * 180 / np.pi / 2
        #p_args[4][..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # set hue according to the angle of optical flow
        p_args[4][..., 0] = ang * ((1 / 360.0) * (180 / 255.0))
        # set value according to the normalized magnitude of optical flow
        p_args[4][..., 2] = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX, -1)
        # multiply each pixel value to 255
        hsv_8u = np.uint8(p_args[4] * 255.0)
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)
        img = cv2.add(p_frame, bgr)
        # Update the previous frame
        p_args[3] = new_frame
        if False:
            self.message.emit("DF; Motion detected!\n")
            return img,True
        return img,False
    
    ######################################################
    # Child Thread Dedection and Tracking
    ######################################################
    def thread_program_dedection_tf_init(self, p_frame, p_args):
     
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=p_args[0])
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
        #options = {"metaLoad": "built_graph/traffic3-tiny-yolo-voc.meta", "pbLoad": "built_graph/traffic3-tiny-yolo-voc.pb", "threshold": 0.1, "gpu": 1.0}
        #options = {"metaLoad": "built_graph/faces-tiny-yolo-voc.meta", "pbLoad": "built_graph/faces-tiny-yolo-voc.pb", "threshold": 0.5, "gpu": 1.0}
        #options = {"metaLoad": "built_graph/faces-tiny-yolo-voc-faces11.meta", "pbLoad": "built_graph/faces-tiny-yolo-voc-faces11.pb", "threshold": 0.5, "gpu": 1.0}
        #options = {"model": "cfg/yolo-face.cfg", "load": "bin/yolo-face_final.weights", "threshold": 0.1, "gpu": 1.0}
        #options = {"model": "../darkflow/cfg/tiny-yolo-voc.cfg", "load": "../darkflow/bin/tiny-yolo-voc.weights", "threshold": p_args[1], "gpu": 1.0}
        #options = {"metaLoad": "../darkflow/built_graph/tiny-yolo-voc-2c.meta", "pbLoad": "../darkflow/built_graph/tiny-yolo-voc-2c.pb", "threshold": 0.5, "gpu": 1.0}
        #options = {"model": "../darkflow/cfg/v1/yolo-tiny.cfg", "load": "../darkflow/bin/yolo-tiny.weights", "threshold": 0.5, "gpu": 1.0}
        options = {"model": "../darkflow/cfg/yolo.cfg", "load": "../darkflow/bin/yolo.weights", "threshold": p_args[1], "gpu": 1.0}
        tfnet = TFNet(options)
        return [p_frame, tfnet]
    
    def thread_program_dedection_tf_run(self, p_frame, p_args):
        result = p_args[1].return_predict(p_frame)
        data = json.loads(str(result).replace("\'","\""))
        # run throught result and show within image
        count = 0
        labelsL = []
        bboxesL = []
        mystr = ""
        for key in data:
            count += 1
            label = key['label']
            if label:
                confidence = key['confidence']
                topleft_x = key['topleft']['x']
                topleft_y = key['topleft']['y']
                bottomright_x = key['bottomright']['x']
                bottomright_y = key['bottomright']['y']
                mystr += "TF;The predicted values are: {},{},{},{},{},{},{}\n".format(label, count, confidence,topleft_x,topleft_y,bottomright_x,bottomright_y)
                #print("The predicted values are: {},{},{},{},{},{},{}".format(label, count, confidence,topleft_x,topleft_y,bottomright_x,bottomright_y))
                msk_motion_r1 = (topleft_x,topleft_y,bottomright_x-topleft_x,bottomright_y-topleft_y)
                cv2.rectangle(p_frame,(msk_motion_r1[0],msk_motion_r1[1]),(msk_motion_r1[0]+msk_motion_r1[2],msk_motion_r1[1]+msk_motion_r1[3]),(0,255,0),3)
#                labelsL.append([label,confidence])
#                bboxesL.append(msk_motion_r1)
                self.message.emit(mystr)
#        for i in range(len(bboxesL)):
#            if i<len(bboxesL)-1:
#                if bb_intersection_selection(bboxesL[i], bboxesL[i+1:]) == True:
#                    cv2.rectangle(p_frame,(bboxesL[i][0],bboxesL[i][1]),(bboxesL[i][0]+bboxesL[i][2],bboxesL[i][1]+bboxesL[i][3]),(0,255,0),1)
#                    font = cv2.FONT_HERSHEY_PLAIN
#                    cv2.putText(p_frame,labelsL[i][0],(bboxesL[i][0],bboxesL[i][1]-20), font, 1,(0,255,0),1,cv2.LINE_AA)
#                    cv2.putText(p_frame,str(i)+":"+str(labelsL[i][1]),(bboxesL[i][0],bboxesL[i][1]-10), font, 1,(0,255,0),1,cv2.LINE_AA)
#            else:
#                cv2.rectangle(p_frame,(bboxesL[i][0],bboxesL[i][1]),(bboxesL[i][0]+bboxesL[i][2],bboxesL[i][1]+bboxesL[i][3]),(0,255,0),3)
#                font = cv2.FONT_HERSHEY_PLAIN
#                cv2.putText(p_frame,labelsL[i][0],(bboxesL[i][0],bboxesL[i][1]-20), font, 1,(0,255,0),1,cv2.LINE_AA)
#                cv2.putText(p_frame,str(i)+":"+str(labelsL[i][1]),(bboxesL[i][0],bboxesL[i][1]-10), font, 1,(0,255,0),1,cv2.LINE_AA)
        
#        if len(bboxesL) > 0:
#            return p_frame,True
        return p_frame,False

class MQThreadVideoCapture(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    #change_pixmap_signal_with_image = pyqtSignal(np.ndarray)

    def __init__(self, p_url = 0, p_FPS = 25):
        super(MQThreadVideoCapture, self).__init__()
        self._run_flag = False
        self._url = p_url
        self._iFPS = 1/p_FPS
        #self._read_lock = Lock()
        self._read_lock = QMutex()
        # try open to see if credentials are ok
        cap = cv2.VideoCapture(self._url)
        if not cap.isOpened():
            raise IOError
        cap.release()
        
    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(self._url)
        if not cap.isOpened():
            return
        #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._run_flag = True
        while self._run_flag:
            #self._read_lock.acquire()
            self._read_lock.lock()
            self._ret, self._cv_img = cap.read()
            self._read_lock.unlock()
            #self._read_lock.release()
            if self._ret:
                self.change_pixmap_signal.emit(self._cv_img)
                #self.change_pixmap_signal_with_image.emit(self._cv_img)
            #self.usleep(1000*100)
            #self.msleep(1000)
        # shut down capture system
        cap.release()

    def read(self):
        frame = None
        if self._run_flag:
            #self._read_lock.acquire()
            self._read_lock.lock()
            frame = self._cv_img
            self._read_lock.unlock()
            #self._read_lock.release()
        return frame
    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class MQThreadVideoFilter(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray,bool)
    #change_pixmap_signal_with_image = pyqtSignal(np.ndarray)

    def __init__(self, p_func_init=None, p_func_run=None, *args,**kwargs):
        super(MQThreadVideoFilter, self).__init__()
        self._run_flag = False
        self._read_lock = QMutex()
        self._cv_img = False
        #self._cv_bak = False
        self._func_init = p_func_init
        self._func_run = p_func_run
        self._args = args[0]
        
    def run(self):
        self._run_flag = True
        self._cv_bak = self._cv_img.copy()
        while self._run_flag:
            lastargs = self._func_init(self._cv_bak, self._args)
            break
        while self._run_flag:
            #self._read_lock.lock()
            img,flag = self._func_run(self._cv_img.copy(), lastargs)
            #self._read_lock.unlock()
            self.change_pixmap_signal.emit(img,flag)
    
    def setImage(self, p_img):
        #self._read_lock.lock()
        self._cv_img = p_img
        #self._read_lock.unlock()
        if self._run_flag == False and not self.isRunning():
            self.start()
        
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

# A Find tool bar (bottom area)
class MQToolBarURL(QToolBar):

    start_camera = pyqtSignal(bool, str, str, str, str)
    start_capture = pyqtSignal(bool)
    
    def __init__(self, p_name, p_parent=None):
        super(MQToolBarURL, self).__init__(p_name, p_parent)
        self._url = QLabel("URL:")
        self.addWidget(self._url)

        self._line_edit = QLineEdit()
        self._line_edit.setClearButtonEnabled(True)
        self._line_edit.setPlaceholderText("rtsp://...")
        self._line_edit.setText(URL)
        self._line_edit.returnPressed.connect(self._open_cam)
        self.addWidget(self._line_edit)

        self._open_cam_button = QToolButton()
        self._open_cam_button.setText('On')
        style_icons = ':/images/'
        self._open_cam_button.setIcon(QIcon(style_icons + 'play.png'))
        self._open_cam_button.setToolTip("Start video")
        self._open_cam_button.setStatusTip("Start video stream")
        self._open_cam_button.clicked.connect(self._open_cam)
        self.addWidget(self._open_cam_button)

        self._close_cam_button = QToolButton()
        self._close_cam_button.setText('Off')
        self._close_cam_button.setIcon(QIcon(style_icons + 'pause.png'))
        self._close_cam_button.setToolTip("Stop video")
        self._close_cam_button.setStatusTip("Stop video stream")
        self._close_cam_button.clicked.connect(self._close_cam)
        self.addWidget(self._close_cam_button)

        self._video_capture_checkbox = QCheckBox('Video Capture')
        self._video_capture_checkbox.setToolTip("Start/Stop Video Capture")
        #self._video_capture_checkbox.setStatusTip("Start/Stop Video Capture to current mp4 File")
        self._video_capture_checkbox.clicked.connect(self._video_capture)
        self.addWidget(self._video_capture_checkbox)
        
    def _open_cam(self):
        url = QUrl(self._line_edit.text())
        url_str = self._line_edit.text()
        self.start_camera.emit(True, url_str, url.host(), url.userName(), url.password())
        
    def _close_cam(self):
        self.start_capture.emit(False)
        self._video_capture_checkbox.setChecked(False)
        self.start_camera.emit(False, None, None, None, None)

    def _video_capture(self, p_flag):
        self.start_capture.emit(p_flag)
        
    def setStartCapture(self, p_flag):
        self._video_capture_checkbox.setChecked(p_flag)
    
    def mousePressEvent(self, e):
        e.accept()
            
    def wheelEvent(self, e):
        e.accept()
        
class TreeModel(QStandardItemModel):
    def __init__(self, p_parent=None):
        super(TreeModel, self).__init__(p_parent)

        self.mqObject = MQObject(self)
        self.setHorizontalHeaderLabels(["Methods", "Description/Parameter"])

        self._lookupTable = [[True, 'Dedection', 'Dedection of humans, cars, ...', False, 
                              'Dedection of humans, cars, ...', 'Dedection of humans, cars, ...', False, None, None],
                             [False, 'NN-TF', 'Deep Neuronal Networks using tensorflow to dedect humans, cars, ...', False, 
                              '0.6, 0.5', 'Parameters: 1: using percent of graphics card memory, 2: detection threshold - the lower the value the more dedections are found, but with less accuracy!', True, self.mqObject.thread_program_dedection_tf_init, self.mqObject.thread_program_dedection_tf_run],
                             [True, 'K-Means', 'Reduction of colors by k-Means ...', False,
                              'Reduction of colors by k-Means ...', 'Reduction of colors by k-Means ...', False, None, None],
                             [False, 'K-Means-TF', 'K-Means implemented using tensorflow', False, 
                              '2, None', 'Parameters: 1: number of classes, 2: none', True, self.mqObject.thread_program_kmeans_tf_init, self.mqObject.thread_program_kmeans_tf_run],
                             [False, 'K-Means-Software', 'K-Means implemented using prozessor only', False, 
                              '2, None', 'Parameters: 1: number of classes, 2: none', True, self.mqObject.thread_program_kmeans_init, self.mqObject.thread_program_kmeans_run],
                             [True, 'Motion', 'Different motion dedection filters ...', False,
                              'Different motion dedection filters ...', 'Different motion dedection filters ...', False, None, None],
                             [False, 'Background Differences', 'The background is extracted such that foreground motion can be dedected', False, 
                              '600, 200, 0, 5', 'Parameters: 1:, 2:, 3:, 4:', True, self.mqObject.thread_program_motion_simple_init, self.mqObject.thread_program_motion_simple_run],
                             [False, 'Lukas-Kanade', 'Features are dedected, like corners and traced within different images', False, 
                              'None', 'Parameters: 1: none', True, self.mqObject.thread_program_motion_lk_init, self.mqObject.thread_program_motion_lk_run],
                             [False, 'Dense Motion', 'Every Pixel is traced within following images and the direction of the flow is color coded', False, 
                              'True, None', 'Parameters: 1: method, 2: none', True, self.mqObject.thread_program_motion_df_init, self.mqObject.thread_program_motion_df_run],
                             [True, 'Pixel Filtering', 'Simple pixel filtering or convolution based filtering ...', False,
                              'Simple pixel filtering or convolution based filtering ...', 'Simple pixel filtering or convolution based filtering ...', False, None, None],
                             [False, 'Cartoon Filter', 'Simple pixel filtering or convolution based filtering ...', False, 
                              'None', 'Parameters: 1: none', True, self.mqObject.thread_program_filter_none, self.mqObject.thread_program_filter_cartoon_run],
                             [False, 'Warming Filter', 'Simple pixel filtering or convolution based filtering ...', False, 
                              'None', 'Parameters: 1: none', True, self.mqObject.thread_program_filter_none, self.mqObject.thread_program_filter_warming_run],
                             [False, 'Cooling Filter', 'Simple pixel filtering or convolution based filtering ...', False, 
                              'None', 'Parameters: 1: none', True, self.mqObject.thread_program_filter_none,self.mqObject.thread_program_filter_cooling_run],
                             [True, 'Recording Options', 'When a method is running, you can start recording as long as the filter method require this ..,', False, 
                              'When a method is running, you can start recording as long as the filter method require this ...', 'When a method is running, you can start recording as long as the filter method require this ...', False, None, None],
                             [False, 'Start Recording', 'As long as the method require it, recording will be done, at least n milliseconds ...', False, 
                              '"out.mp4", 6000', 'Parameters: 1: filename of the video, 2: minimum milliseconds of recording', True, None, None]]
        cap = False
        capn = 0
        for item in self._lookupTable:
            if item[0]==True and cap == False:
                #print('make root method')
                parentItem = self.invisibleRootItem()
                cap = True
            else:
                #print('make child method')
                if cap == True:
                    parentItem = self.item(capn)
                    capn = capn + 1
                    cap = False
            itemL = QStandardItem(item[1])
            itemL.setStatusTip(item[2])
            itemL.setToolTip(item[2])
            itemL.setEditable(item[3])
            assert(item[3]==False)
            itemR = QStandardItem(item[4])
            itemR.setStatusTip(item[5])
            itemR.setToolTip(item[5])
            itemR.setEditable(item[6])
            assert(item[6] == True or item[6] == False)
            parentItem.appendRow([itemL, itemR])
    
    def getMethods(self, p_str):
        for item in self._lookupTable:
            if item[0]==False and item[1]==p_str:
                return item[7],item[8]
        return self.mqObject.thread_program_filter_none, self.mqObject.thread_program_filter_none

    def getStandardMP4FileName(self):
        return str(eval(self._lookupTable[14][4])[0])
            
class MQDockWidgetFilterOptions(QDockWidget):
    change_filter_signal = pyqtSignal(str, object, object, object)
    def __init__(self, p_name, p_parent=None):
        super(MQDockWidgetFilterOptions, self).__init__(p_name, p_parent)
        #QDockWidget.__init__(self, p_name, p_parent)
        self.setAllowedAreas(Qt.TopDockWidgetArea | Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)   
        self._mainWidget = QWidget(self)
        #self._mainWidget.setObjectName("centralwidget")
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        self._treeView = QTreeView()
        self._treeView.setAlternatingRowColors(True)
        self._treeView.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._treeView.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._treeView.setAnimated(True)
        self._treeView.setAllColumnsShowFocus(True)
        #self._treeView.setRootIsDecorated(False)
        self._treeView.doubleClicked.connect(self.selectedFilterAndArgs)
        vbox.addWidget(self._treeView)
        self._mainWidget.setLayout(vbox)
        self.setWidget(self._mainWidget)
        
        self._model = TreeModel()
        self._treeView.setModel(self._model)       
        self._treeView.expandToDepth(2)
        for column in range(self._model.columnCount(QModelIndex())):
            self._treeView.resizeColumnToContents(column)
            
    def selectedFilterAndArgs(self, index):
        model = self._treeView.model()
        #print(index.row(), index.column())
        if not index.isValid():
            return None
        #print(index.row(), index.column())
        if index.column() != 0 or model.rowCount(index)>0:
            return None
        method = index.parent().child(index.row(),index.column()+0).data()
        para = index.parent().child(index.row(),index.column()+1).data()
        minit,mrun = model.getMethods(method)
        #print("Methode:", method)
        #print("Parameter:", para)
        self.change_filter_signal.emit(method, minit, mrun, eval(para))
     
    def getStandardMP4Filename(self):
        return self._treeView.model().getStandardMP4FileName()

class MQWidgetConsole(QPlainTextEdit):
    def __init__(self, p_parent=None):
        super(MQWidgetConsole, self).__init__(p_parent)
        self.document().setMaximumBlockCount(1000)
        self._counter = 0
        p = self.palette()
        p.setColor(QPalette.Base, Qt.white)
        p.setColor(QPalette.Text, Qt.black)
        self.setPalette(p)
    
    def putData(self, data):
        self._counter+=1
        self.insertPlainText(str(self._counter)+" : "+QDateTime.currentDateTime().toString("yyyy.MM.dd - hh:mm:ss ")+data)
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

class MQDockWidgetConsole(QDockWidget):
    def __init__(self, p_name, p_parent=None):
        super(MQDockWidgetConsole, self).__init__(p_name, p_parent)
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)   
        self._mainWidget = MQWidgetConsole(self)
        self.setWidget(self._mainWidget)
        
class MQWidgetPTZPath(QWidget):
    ctrlPointsChanged = pyqtSignal()
    ctrlPointMessage = pyqtSignal(str,int)
    ctrlPointCurrentPos = pyqtSignal(int) 
    def __init__(self, p_parent = None):
        super(MQWidgetPTZPath, self).__init__(p_parent)
        #QWidget.__init__(p_parent)
        self.setWindowTitle("MQWidgetPTZPath")
        #self.setStyleSheet("background-color:green;")
        self.setMouseTracking(True)
        # set Background Styles does not work
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(230,230,230,150))
        self.setPalette(palette)
        
        # axis spec.
        self._mAxisThickness = 8
        self._mArrowThickness = self._mAxisThickness/2
        self._mAxisPen = QPen(QColor(0, 0, 0, 150), self._mAxisThickness)
        self._mAxisPenArrow = QPen(QColor(0, 0, 0, 150), self._mArrowThickness)
        self._mAxisAlign = 0
        self._mAxisName = 'y'
        self._mDrawAreaRect = self.rect()
        
        # set the two boundary points
        self._mPoints = QPolygonF()
        self._mPointCurrent = None
        self._mPoint = None
	    # scale
        self._mPointScale = QPointF(10000, 180)
        self._mPointsLock = QPolygon()
        self._mPointsReset()
        
        self._mPointActiveIndex = -1
        self._mPointsEditable = True
        self._mPointsEnabled = True

    	# control points spec.		
        self._mPointSortType = 0 #SortXDir
        self._mPointShape = 0 # ShapeRectangle==0, ShapeCircle==1
        self._mPointSize = QSizeF(20.0, 20.0)
        self._mPointPen = QPen(QColor(0, 0, 0, 255), 1)
        self._mPointTextPen = QPen(QColor(0, 0, 0, 255), 1)
        self._mPointBrush = QBrush(QColor(100, 100, 100, 150))
        self._mPointsPath = None
        # for PTZ camera
        self._mPointcT = 0.0
        self._mPointdT = 0.0
        self._mPointcX = 0.0
        self._mPointdX = 0.0
        
    	# connection between control points spec.
        self._mConnectionType = 3 #ConnectionLinear=2, ConnectionQuadratic=3
        self._mConnectionPen = QPen(QColor(255, 0, 0, 150), 4)
        
        # timer
        self._timer = None

    def getCurrentPos(self):
        if self._mPoint != None:
            return self._mPoint
        return None
    
    def setCtrlPoint(self, p_x, p_y):
        #print(p_x, p_y)
        # no scaling of x value, since it is in window coordinates
        # y value has to be transformed into window coordinates
        y = self._mPointYScaleFromCoord(p_y)
        p = QPointF(p_x, y)
        if p.x() <= self._mPoints[0].x()+self._mPointSize.width()/2:
            self._mPoints[0] = self._mPointClip(p, self._mDrawAreaRect, self._mPointsLock[0].x())
            return
        if p.x() >= self._mPoints[self._mPoints.size()-1].x()-self._mPointSize.width()/2:
            self._mPoints[self._mPoints.size()-1] = self._mPointClip(p, self._mDrawAreaRect, self._mPointsLock[self._mPoints.size()-1].x())
            return
        self._mPointInsert(p)
        self._mPointActiveIndex = -1
        
    def setDimensionsAxis(self, p_dim, p_axisalign, p_name):
        self._mPointScale = p_dim
        self._mAxisAlign = p_axisalign
        self._mAxisName = p_name
        self._mPointsReset()
        
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.parentWidget().mousePressEvent(event)
            return
        clickPos = event.pos()
        # find the current active point
        # nearest to the user's mouse click position 
        self._mPointActiveIndex = -1
        self._mPointFind(clickPos)
    
        if event.button() == Qt.LeftButton:
            self._mPointInsert(clickPos)
        if event.button() == Qt.MidButton:
            self._mPointRemove(clickPos)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._mPointActiveIndex >= 0:
            self._mPointActiveIndex = -1
            self.ctrlPointsChanged.emit()
    
    def mouseMoveEvent(self, event):
        movePos = event.pos()
        # find the current active point
        # nearest to the user's mouse move position 
        if self._mPointActiveIndex == -1:
            self._mPointFind(movePos)
            if self._mPointActiveIndex >= 0:
                assert (self._mPointActiveIndex >= 0 and self._mPointActiveIndex < self._mPoints.size())
                pp = self._mPointScaleToCoord(self._mPoints[self._mPointActiveIndex])
                self.ctrlPointMessage.emit("Over Point: "+
                                           str(self._mPointActiveIndex)+":"+
                                           str(pp.x())+"x"+
                                           str(pp.y()), 4000)
                QToolTip.showText(event.globalPos(), "Over Point: "+
                                  str(self._mPointActiveIndex)+":"+
                                  str(pp.x())+"x"+
                                  str(pp.y()), self, self.rect())
                self._mPointActiveIndex = -1
                event.accept()
                return
        QToolTip.hideText()
        if self._mPointActiveIndex >= 0:
            assert (self._mPointActiveIndex >= 0 and self._mPointActiveIndex < self._mPoints.size())
            self._mPoints[self._mPointActiveIndex] = movePos
            self._mPoints[self._mPointActiveIndex] = self._mPointClip(self._mPoints[self._mPointActiveIndex], self._mDrawAreaRect, self._mPointsLock[self._mPointActiveIndex].x())
            self._mPointsSort()
            pp = self._mPointScaleToCoord(self._mPoints[self._mPointActiveIndex])
            self.ctrlPointMessage.emit("Moving Point: "+
                                       str(self._mPointActiveIndex)+":"+
                                       str(pp.x())+"x"+
                                       str(pp.y()), 4000)
            self.ctrlPointsChanged.emit()
            self.repaint()

    def paintEvent(self, event):
        #print("paint!")
        p = QPainter(self)
        #p.begin(self) # not neccessary to call begin and end, when Constructor QPainter(self) is called
        self._mAxisDraw(p)
        self._mPointsDraw(p)
        #p.end()
    
    def resizeEvent(self, event):
        #print("resize!")
        if event.size().width() <=0 or event.size().height()<=0:
            return
        # compute new drawing area
        n = self.rect()
        # compute streching factors
        sx = n.width()/self._mDrawAreaRect.width()
        sy = n.height()/self._mDrawAreaRect.height()
        # set new positions of the points
        for index in range(0,self._mPoints.size(),1):
            pp = QPointF((self._mPoints[index].x())*sx, (self._mPoints[index].y())*sy)
            #pp = QPointF((self._mPoints[index].x()-self._mDrawAreaRect.x())*sx+n.x(),
            #             (self._mPoints[index].y()-self._mDrawAreaRect.y())*sy+n.y())
            #self._mPoints[index] = pp
            self._mPoints[index] = self._mPointClip(pp, n, self._mPointsLock[index].x())
        if self._mPointCurrent != None:
            self._mPointCurrent = QPointF((self._mPointCurrent.x())*sx, (self._mPointCurrent.y())*sy)
        self._mDrawAreaRect = n
        self._mPointComputeParam()
        self.ctrlPointsChanged.emit()
        
    def _mPointComputeParam(self):
        #self._mPoint = None
        #self._mPointCurrent = None
        self._mPointcT = 0.0
        self._mPointcX = 0.0
        pm = self._mPointScaleToCoord(self._mPoints[self._mPoints.size()-1])
        self._mPointdT = pm.x() / (self._mDrawAreaRect.width())
        self._mPointdX = 1.0 / (self._mDrawAreaRect.width())
        self.ctrlPointMessage.emit("Parameter: dT="+
                                   str(round(self._mPointdT,2))+" ms : dX="+
                                   str(round(self._mPointdX,4)), 60000)
        
    def _mPointCurrentPos(self):
        if self._mPointcX >= 1.0: #or self._mPointcT > pm.x()
            self._mPoint = None
            self._mPointCurrent = None
            self._mPointcT = 0.0
            self._mPointcX = 0.0
            self._mPointComputeParam()
        self._mPointCurrent = self._mPointsPath.pointAtPercent(self._mPointcX)
        #self._mPointCurrent = self._mPointClip(self._mPointCurrent, self._mDrawAreaRect)
        self._mPoint = self._mPointScaleToCoord(self._mPointCurrent)
        #print(self._mPoint, self._mPointCurrent, self._mPointcT, self._mPointcX, self._mPointdT, self._mPointdX)
        #pm = self._mPointScaleToCoord(self._mPoints[self._mPoints.size()-1])
        self._mPointcT += self._mPointdT
        self._mPointcX += self._mPointdX
        self.ctrlPointCurrentPos.emit(self._mPoint.y())
        self.repaint()
    
    def _mPointFind(self, pos):
        # a point already selected
        if self._mPointActiveIndex >= 0:
            return
    	# find the current active point
        # nearest to a position in the widget
        assert(self._mPointActiveIndex == -1)
        distance = -1.0
        for index in range(0,self._mPoints.size(),1):
            d = QLineF(pos, self._mPoints[index]).length()
            if (distance<0 and d < self._mPointSize.width()) or d<distance:
                distance = d
                self._mPointActiveIndex = index
    
    def _mPointInsert(self, pos):
        if self._mPointActiveIndex >= 0:
            self.ctrlPointsChanged.emit()
            return
        # find where to insert a new point in the array 
        # according to a position in the widget
        assert(self._mPointActiveIndex == -1);
        for index in range(0,self._mPoints.size(),1):
            if self._mPoints[index].x() > pos.x():
                self._mPointActiveIndex = index
                break
        assert(self._mPointActiveIndex >= 0 and self._mPointActiveIndex < self._mPoints.size())
        self._mPoints.insert(self._mPointActiveIndex, pos)
        self._mPointsLock.insert(self._mPointActiveIndex, QPoint(0x00,0))
        #self._mPoints[self._mPointActiveIndex] = self._mPointClip(self._mPoints[self._mPointActiveIndex], self._mDrawAreaRect, self._mPointsLock[self._mPointActiveIndex].x())
        pp = self._mPointScaleToCoord(self._mPoints[self._mPointActiveIndex])
        self.ctrlPointMessage.emit("Insert Point: "+
                                   str(self._mPointActiveIndex)+":"+
                                   str(pp.x())+"x"+
                                   str(pp.y()), 4000)
        self.ctrlPointsChanged.emit()
        self.repaint()
    
    def _mPointRemove(self, pos):
        if self._mPointActiveIndex < 0:
            self._mPointFind(pos)
        if self._mPointActiveIndex >= 0:
            assert (self._mPointActiveIndex >= 0 and self._mPointActiveIndex < self._mPointsLock.size())
            assert (self._mPointActiveIndex >= 0 and self._mPointActiveIndex < self._mPoints.size())
            if self._mPointsLock[self._mPointActiveIndex].x() == 0:
                pp = self._mPointScaleToCoord(self._mPoints[self._mPointActiveIndex])
                self.ctrlPointMessage.emit("Remove Point: "+
                                           str(self._mPointActiveIndex)+":"+
                                           str(pp.x())+"x"+
                                           str(pp.y()), 4000)
                self._mPoints.remove(self._mPointActiveIndex)
                self._mPointsLock.remove(self._mPointActiveIndex)
            else:
                self._mPointsReset()
            self._mPointActiveIndex = -1;
            self.ctrlPointsChanged.emit()        
            self.repaint()
            

    #def _QPointF_XL(self, p1, p2):
    #    return p1.x() < p2.x()        
        
    def _mPointsSort(self):
        def get_QPointF_XL1(p):
            return p.x()
        if self._mPointActiveIndex >= 0:
            assert(self._mPointActiveIndex >= 0 and self._mPointActiveIndex < self._mPoints.size())
            pp = self._mPoints[self._mPointActiveIndex]
        #qSort(self._mPoints.begin(), self._mPoints.end(), self._QPointF_XL)
        ppSorted = sorted(self._mPoints, key=get_QPointF_XL1)
        for index in range(0,self._mPoints.size(),1):
            self._mPoints[index] = ppSorted[index]
        # identify current point, i.e. compensate for changed order
        if self._mPointActiveIndex >= 0:
            for index in range(0,self._mPoints.size(),1):
                if self._mPoints[index] == pp:
                    self._mPointActiveIndex = index
                    break
    
    def _mPointBoundingRect(self, p_point):
        w = self._mPointSize.width()
        h = self._mPointSize.height()
        x = p_point.x() - w / 2.0
        y = p_point.y() - h / 2.0
        return QRectF(x, y, w, h)

    def _mPointClip(self, p, clip, lock=0x00):
        pp = p
        if p.x() < clip.left() or lock & 0x01:
            pp.setX(clip.left())
        elif p.x() > clip.right() or lock & 0x02:
            pp.setX(clip.right())
        if p.y() < clip.top() or lock & 0x04:
            pp.setY(clip.top())
        elif p.y() > clip.bottom() or lock & 0x08:
            pp.setY(clip.bottom())
        return pp
  
    def _mPointYScaleFromCoord(self, p_y):
        y = p_y
        h = self._mDrawAreaRect.height()
        if self._mAxisAlign == 0:
            y = -y*h/self._mPointScale.y()+h
        elif self._mAxisAlign == 1:
            y = (-y+math.ceil(self._mPointScale.y()/2))*h/self._mPointScale.y()
        elif self._mAxisAlign == 2:
            y = -y*h/self._mPointScale.y()
        assert (y >= 0 and y <= h)
        return y
       
    def _mPointScaleToCoord(self, point):
        assert (self._mDrawAreaRect.width() > 0)
        assert (self._mDrawAreaRect.width() == (self._mDrawAreaRect.right()-self._mDrawAreaRect.left()+1)) 
        assert (self._mDrawAreaRect.height() > 0)
        assert (self._mDrawAreaRect.height() == (self._mDrawAreaRect.bottom()-self._mDrawAreaRect.top()+1))
        #print(point, self._mDrawAreaRect.left(), self._mDrawAreaRect.right()+1)
        #print(point, self._mDrawAreaRect.top(), self._mDrawAreaRect.bottom()+1)
        assert (point.x() >= self._mDrawAreaRect.left() and
                point.x() <= self._mDrawAreaRect.right()+1)
        assert (point.y() >= self._mDrawAreaRect.top() and
                point.y() <= self._mDrawAreaRect.bottom()+1.001)
        assert (self._mDrawAreaRect.x() == self._mDrawAreaRect.left())
        assert (self._mDrawAreaRect.y() == self._mDrawAreaRect.top())
        # transform point
        w = self._mDrawAreaRect.width()
        h = self._mDrawAreaRect.height()
        p = point.toPoint()
        #p = point
        if self._mAxisAlign == 0:
            #p.setX(self._mPointScale.x() * (0+(p.x()-self._mDrawAreaRect.x())) / (w-1))
            #p.setY(self._mPointScale.y() * (h-(p.y()-self._mDrawAreaRect.y())) / (h-1))
            p.setX(self._mPointScale.x() * (0+point.x()) / (w))
            p.setY(self._mPointScale.y() * (h-point.y()) / (h))
            #print("Align0:",p)
            assert (p.x() >= 0 and p.x() <= self._mPointScale.x())
            assert (p.y() >= 0 and p.y() <= self._mPointScale.y())
        elif self._mAxisAlign == 1:
            #p.setX(self._mPointScale.x() * (0+(point.x()-self._mDrawAreaRect.x())) / (w-1))
            #p.setY(self._mPointScale.y() * (h-(point.y()-self._mDrawAreaRect.y())) / (h-1) - math.floor(self._mPointScale.y()/2))
            p.setX(+(self._mPointScale.x() * (point.x()) / (w)))
            p.setY(-(self._mPointScale.y() * (point.y()) / (h) - math.ceil(self._mPointScale.y()/2)))
            #print("Align1:",p)
            assert (p.x() >= 0 and p.x() <= self._mPointScale.x())
            assert (p.y() >= -self._mPointScale.y()/2-1 and p.y() <=  self._mPointScale.y()/2+1)
        elif self._mAxisAlign == 2:
            #p.setX(self._mPointScale.x() * (0+(p.x()-self._mDrawAreaRect.x())) / (w-1))
            #p.setY(self._mPointScale.y() * (h-(p.y()-self._mDrawAreaRect.y())) / (h-1))
            p.setX(self._mPointScale.x() * (0+point.x()) / (w))
            p.setY(self._mPointScale.y() * (0-point.y()) / (h))
            #print("Align0:",p)
            assert (p.x() >= 0 and p.x() <= +self._mPointScale.x())
            assert (p.y() <= 0 and p.y() >= -self._mPointScale.y())
        #print(p)
        return p
    
    def _mPointsReset(self):
        self._mPoints.clear()
        self._mPointsLock.clear()
        self._mPoint = None
        self._mPointCurrent = None
        # set the two boundary points
        r = self._mDrawAreaRect
        if self._mAxisAlign == 0:
            p1 = QPointF(r.left(), round(r.bottom(),0)+1)
            p2 = QPointF(r.right(), round(r.bottom(),0)+1)
        elif self._mAxisAlign == 1:
            p1 = QPointF(r.left(), round(r.bottom()/2,0)+1)
            p2 = QPointF(r.right(), round(r.bottom()/2,0)+1)
        elif self._mAxisAlign == 2:
            p1 = QPointF(r.left(), round(r.top(),0)+0)
            p2 = QPointF(r.right(), round(r.top(),0)+0)
        self._mPoints.append(p1)
        self._mPoints.append(p2)
        #print(self._mPoints)
        #  lock them to the appropriate boundary
        assert (self._mPoints.size() == 2)
        #self._mPointsLock.resize(self._mPoints.size())
        #self._mPointsLock.fill(0)
        self._mPointsLock.append(QPoint(0x01,0)) # hex(0X01) LockToLeft
        self._mPointsLock.append(QPoint(0x02,0)) # hex(0X02) LockToRight
        self._mPointComputeParam()

    def _mPointsDraw(self, p):
        # draw into widget using a painter
        p.setRenderHint(QPainter.Antialiasing)
    	# draw connection path between the points
        if self._mConnectionPen.style() != Qt.NoPen and self._mConnectionType != 0:
            p.setPen(self._mConnectionPen)
            if self._mConnectionType == 2: # Linear
                #p.drawPolyline(self._mPoints) or
                path = QPainterPath(self._mPoints[0])
                for index in range(1,self._mPoints.size(),1):
                    path.lineTo(self._mPoints[index-0])
				# draw polygon path
                p.drawPath(path)
            elif self._mConnectionType == 3: # Quadratic
                path = QPainterPath(self._mPoints[0])
	            # consider first control point twice
                #p1 = self._mPoints[0]
                #p2 = self._mPoints[0]
                #bc = QPointF(p1.x() + (p2.x() - p1.x()) / 2, 
                #             p1.y() + (p2.y() - p1.y()) / 2)
                #path.moveTo(bc)
                #path.quadTo(p1, bc)
                for index in range(1,self._mPoints.size(),1):
                    p1 = self._mPoints[index-1]
                    p2 = self._mPoints[index-0]
                    bc = QPointF(p1.x() + (p2.x() - p1.x()) / 2,
                                 p1.y() + (p2.y() - p1.y()) / 2)
                    path.quadTo(p1, bc)       
                # consider last control point twice
                p1 = self._mPoints[self._mPoints.size()-1]
                p2 = self._mPoints[self._mPoints.size()-1]
                bc = QPointF(p1.x() + (p2.x() - p1.x()) / 2,
                             p1.y() + (p2.y() - p1.y()) / 2)
                path.quadTo(p1, bc)
				# draw polygon path
                p.drawPath(path)
            
            # draw points itself
            for index in range(0,self._mPoints.size(),1):
                bounds = self._mPointBoundingRect(self._mPoints[index])
                p.setPen(self._mPointPen);
                if self._mPointShape == 0:
                    p.setBrush(self._mPointBrush)
                    p.drawRect(bounds)
                    p.setPen(self._mConnectionPen)
                    p.drawPoint(self._mPoints[index])
                elif self._mPointShape == 1:
                    p.setBrush(self._mPointBrush)
                    p.drawEllipse(bounds)
                    p.setPen(self._mConnectionPen)
                    p.drawPoint(self._mPoints[index])
                p.setPen(self._mPointTextPen)
                hh = self._mPointScaleToCoord(self._mPoints[index])
                pos = self._mPoints[index]
                if index < self._mPoints.size()-1:
                    pos.setY(pos.y()+10)
                    pos.setX(pos.x()+10)
                    p.drawText(pos, str(index)+":"+str(hh.x())+"x"+str(hh.y()))
                else:
                    pos.setY(pos.y()+10)
                    pos.setX(pos.x()-110)
                    p.drawText(pos, str(index)+":"+str(hh.x())+"x"+str(hh.y()))
            # draw current point
            if self._mPointCurrent != None:
                bounds = self._mPointBoundingRect(self._mPointCurrent)
                p.setPen(self._mPointPen);
                if self._mPointShape == 0:
                    p.setBrush(self._mPointBrush)
                    p.drawRect(bounds)
                    p.setPen(self._mConnectionPen)
                    p.drawPoint(self._mPointCurrent)
                elif self._mPointShape == 1:
                    p.setBrush(self._mPointBrush)
                    p.drawEllipse(bounds)
                    p.setPen(self._mConnectionPen)
                    p.drawPoint(self._mPointCurrent)
                p.setPen(self._mPointTextPen)
                #hh = self._mPointScaleToCoord(self._mPointCurrent)
                hh = self._mPoint
                pos = QPointF(self._mPointCurrent.x()+10, self._mPointCurrent.y()+10)
                p.drawText(pos, "c:"+str(hh.x())+"x"+str(hh.y()))
            self._mPointsPath = path
            #print(self._mPointScaleToCoord(path.pointAtPercent(0.0)),
            #      self._mPointScaleToCoord(path.pointAtPercent(0.5)),
            #      self._mPointScaleToCoord(path.pointAtPercent(1.0)))
            
    def _mAxisDraw(self, p):
        r = self._mDrawAreaRect
        # x-axis
        if self._mAxisAlign == 0: # axis bottom window
            p.setPen(self._mAxisPen)
            p.drawLine(r.left(), r.bottom()-self._mAxisThickness, r.right()-self._mAxisThickness-2, r.bottom()-self._mAxisThickness)
            p.setPen(self._mAxisPenArrow)
            p.drawLine(r.right()-16, r.bottom()-self._mAxisThickness-8, r.right(), r.bottom()-self._mAxisThickness)
            p.drawLine(r.right()-16, r.bottom()-self._mAxisThickness+8, r.right(), r.bottom()-self._mAxisThickness)
            p.drawText(r.right()-54, r.bottom()-30, "x/ms")
        elif self._mAxisAlign == 1: # axis middle window
            p.setPen(self._mAxisPen)
            p.drawLine(r.left(), r.bottom()/2, r.right()-self._mAxisThickness-2, r.bottom()/2)
            p.setPen(self._mAxisPenArrow)
            p.drawLine(r.right()-16, r.bottom()/2-8, r.right(), r.bottom()/2)
            p.drawLine(r.right()-16, r.bottom()/2+8, r.right(), r.bottom()/2)
            p.drawText(r.right()-54, r.bottom()/2-20, "x/ms")
        elif self._mAxisAlign == 2: # axis top window
            p.setPen(self._mAxisPen)
            p.drawLine(r.left(), r.top()+self._mAxisThickness+1, r.right()-self._mAxisThickness-2, r.top()+self._mAxisThickness+1)
            p.setPen(self._mAxisPenArrow)
            p.drawLine(r.right()-16, r.top()+self._mAxisThickness-8+1, r.right(), r.top()+self._mAxisThickness+1)
            p.drawLine(r.right()-16, r.top()+self._mAxisThickness+8+1, r.right(), r.top()+self._mAxisThickness+1)
            p.drawText(r.right()-54, r.top()+35, "x/ms")
        
        # y-axis
        if self._mAxisAlign == 0 or self._mAxisAlign == 1:
            p.setPen(self._mAxisPen)
            p.drawLine(r.left()+self._mAxisThickness, r.bottom(), r.left()+self._mAxisThickness, r.top()+self._mAxisThickness+2)
            p.setPen(self._mAxisPenArrow)
            p.drawLine(r.left()+self._mAxisThickness-8, r.top()+16, r.left()+self._mAxisThickness, r.top())
            p.drawLine(r.left()+self._mAxisThickness+8, r.top()+16, r.left()+self._mAxisThickness, r.top())
            p.drawText(r.left()+self._mAxisThickness+20, r.top()+24, self._mAxisName)
        elif self._mAxisAlign == 2:
            p.setPen(self._mAxisPen)
            p.drawLine(r.left()+self._mAxisThickness, r.top(), r.left()+self._mAxisThickness, r.bottom()-self._mAxisThickness-2)
            p.setPen(self._mAxisPenArrow)
            p.drawLine(r.left()+self._mAxisThickness, r.bottom(), r.left()+self._mAxisThickness-8, r.bottom()-16)
            p.drawLine(r.left()+self._mAxisThickness, r.bottom(), r.left()+self._mAxisThickness+8, r.bottom()-16)
            p.drawText(r.left()+self._mAxisThickness+20, r.bottom()-24, self._mAxisName)
        
class MQDockWidgetPTZPath(QDockWidget):
    ctrlPointCurrent = pyqtSignal(int,int,int)
    ctrlSetPoint = pyqtSignal(int)
    def __init__(self, p_name, p_parent=None):
        super(MQDockWidgetPTZPath, self).__init__(p_name, p_parent)
        self.setAllowedAreas(Qt.TopDockWidgetArea | Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)   
        self._mainWidget = QWidget(self)
        self._widgetPTZPan = MQWidgetPTZPath(self._mainWidget)
        self._widgetPTZTilt = MQWidgetPTZPath(self._mainWidget)
        self._widgetPTZZoom = MQWidgetPTZPath(self._mainWidget)   
        self._statusBar = QStatusBar(self._mainWidget)
        self._statusBar.showMessage("Ready")
        self._statusBar.setMaximumHeight(40)
        self.setFocusPolicy(Qt.ClickFocus)
        vbox = QVBoxLayout()
        vbox.addWidget(self._widgetPTZPan)
        vbox.addWidget(self._widgetPTZTilt)
        vbox.addWidget(self._widgetPTZZoom)
        vbox.addWidget(self._statusBar)
        vbox.setSpacing(10)
        vbox.setContentsMargins(QMargins(0,0,0,0))
        self._mainWidget.setLayout(vbox)
        self.setWidget(self._mainWidget)
        self._widgetPTZPan.ctrlPointMessage.connect(self._statusBar.showMessage)
        self._widgetPTZTilt.ctrlPointMessage.connect(self._statusBar.showMessage)
        self._widgetPTZZoom.ctrlPointMessage.connect(self._statusBar.showMessage)
        self._timer = None
        self._mPeriodicMove = False
        
    def setDimensionsAxisPan(self, p_dim, p_axisalign):
        self._widgetPTZPan.setDimensionsAxis(p_dim, p_axisalign, "y/deg - pan")
    
    def setDimensionsAxisTilt(self, p_dim, p_axisalign):
        self._widgetPTZTilt.setDimensionsAxis(p_dim, p_axisalign, "y/deg - tilt")
        
    def setDimensionsAxisZoom(self, p_dim, p_axisalign):
        self._widgetPTZZoom.setDimensionsAxis(p_dim, p_axisalign, "y/fac - zoom")
        
    def setColorPan(self, p_color):
        self._widgetPTZPan._mConnectionPen = QPen(p_color, 4)
    
    def setColorTilt(self, p_color):
        self._widgetPTZTilt._mConnectionPen = QPen(p_color, 4)
        
    def setColorZoom(self, p_color):
        self._widgetPTZZoom._mConnectionPen = QPen(p_color, 4)
    
    def setCtrlPoint(self, p_x, p_p, p_t, p_z):
        self._widgetPTZPan.setCtrlPoint(p_x, p_p)
        self._widgetPTZTilt.setCtrlPoint(p_x, p_t)
        self._widgetPTZZoom.setCtrlPoint(p_x, p_z)
        
    def wheelEvent(self, event):
        dy = event.angleDelta().y()
        s = self._widgetPTZPan._mPointScale
        s.setX(s.x()+dy if s.x()+dy>0 else 0)
        self._widgetPTZPan._mPointScale = s
        self._widgetPTZPan._mPointComputeParam()
        #self._widgetPTZPan.ctrlPointsChanged.emit()
        s = self._widgetPTZTilt._mPointScale
        s.setX(s.x()+dy if s.x()+dy>0 else 0)
        self._widgetPTZTilt._mPointScale = s
        self._widgetPTZTilt._mPointComputeParam()
        #self._widgetPTZTilt.ctrlPointsChanged.emit()
        s = self._widgetPTZZoom._mPointScale
        s.setX(s.x()+dy if s.x()+dy>0 else 0)
        self._widgetPTZZoom._mPointScale = s
        self._widgetPTZZoom._mPointComputeParam()
        #self._widgetPTZZoom.ctrlPointsChanged.emit()
        self.repaint()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.ctrlSetPoint.emit(event.pos().x())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            if self._timer == None:
                self._timer = QTimer(self)
                #self._widgetPTZPan._mPointComputeParam()
                #self._widgetPTZTilt._mPointComputeParam()
                #self._widgetPTZZoom._mPointComputeParam()
                #self._timer.timeout.connect(self._widgetPTZPan._mPointCurrentPos)
                #self._timer.timeout.connect(self._widgetPTZTilt._mPointCurrentPos)
                #self._timer.timeout.connect(self._widgetPTZZoom._mPointCurrentPos)
                self._timer.timeout.connect(self._mPointCurrentPos)
                assert (self._widgetPTZPan._mPointdT == self._widgetPTZTilt._mPointdT)
                self._timer.start(self._widgetPTZPan._mPointdT)
                self._statusBar.showMessage("Timer started: Camera is moving according to the path!")
            else:
                self._timer.stop()
                #self._timer.timeout.disconnect(self._widgetPTZPan._mPointCurrentPos)
                #self._timer.timeout.disconnect(self._widgetPTZTilt._mPointCurrentPos)
                #self._timer.timeout.disconnect(self._widgetPTZZoom._mPointCurrentPos)
                self._timer.timeout.disconnect(self._mPointCurrentPos)
                self._timer = None
                #self._widgetPTZPan._mPointCurrent = None
                #self._widgetPTZTilt._mPointCurrent = None
                #self._widgetPTZZoom._mPointCurrent = None
                #self.repaint()
                self._statusBar.showMessage("Timer stopped: Camera is stopped at current path position, press s to start!")
            event.accept()
        if event.key() == Qt.Key_P:
            self._mPeriodicMove = not self._mPeriodicMove
            s = "True" if self._mPeriodicMove else "False"
            self._statusBar.showMessage("Periodic movement: "+ s)
            
    def _mPointCurrentPos(self):
        if self._mPeriodicMove == True:
            self._widgetPTZPan._mPointCurrentPos()
            self._widgetPTZTilt._mPointCurrentPos()
            self._widgetPTZZoom._mPointCurrentPos()
            pc = self._widgetPTZPan.getCurrentPos()
            tc = self._widgetPTZTilt.getCurrentPos()
            zc = self._widgetPTZZoom.getCurrentPos()
            #print(pc.y(), tc.y(), zc.y())
            self.ctrlPointCurrent.emit(pc.y(), tc.y(), zc.y())
        else:
            if self._widgetPTZPan._mPointcX < 1.0:
                self._widgetPTZPan._mPointCurrentPos()
                self._widgetPTZTilt._mPointCurrentPos()
                self._widgetPTZZoom._mPointCurrentPos()
                pc = self._widgetPTZPan.getCurrentPos()
                tc = self._widgetPTZTilt.getCurrentPos()
                zc = self._widgetPTZZoom.getCurrentPos()
                #print(pc.y(), tc.y(), zc.y())
                self.ctrlPointCurrent.emit(pc.y(), tc.y(), zc.y())
            else:
                self._timer.stop()
                #self._timer.timeout.disconnect(self._widgetPTZPan._mPointCurrentPos)
                #self._timer.timeout.disconnect(self._widgetPTZTilt._mPointCurrentPos)
                #self._timer.timeout.disconnect(self._widgetPTZZoom._mPointCurrentPos)
                self._timer.timeout.disconnect(self._mPointCurrentPos)
                self._timer = None
                self._widgetPTZPan._mPoint = None
                self._widgetPTZPan._mPointCurrent = None
                self._widgetPTZTilt._mPoint = None
                self._widgetPTZTilt._mPointCurrent = None
                self._widgetPTZZoom._mPoint = None
                self._widgetPTZZoom._mPointCurrent = None
                self._widgetPTZPan._mPointComputeParam()
                self._widgetPTZTilt._mPointComputeParam()
                self._widgetPTZZoom._mPointComputeParam()
                self._statusBar.showMessage("Timer stopped: Camera is stopped at the end position, restart only from origin!")
        
class MQWidgetVideoDisplay(QWidget):
    def __init__(self, p_parent = None, p_ptz=None):
        super(MQWidgetVideoDisplay, self).__init__(p_parent)
        #QWidget.__init__(p_parent)
        self.setWindowTitle("MQWidgetVideoDisplay")
        # create the label that holds the image
        self._image_label = QLabel()
        self._image_label.setText("Video Stream Off")
        self._image_label.setScaledContents(True)
        self._image_label.setAlignment(Qt.AlignCenter)
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.setContentsMargins(QMargins(0,0,0,0))
        vbox.addWidget(self._image_label)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)
        #self.layout().setContentsMargins(0,0,0,0)
        #self.layout().setContentsMargins(QMargins(0,0,0,0))
        self._ptz = p_ptz
        
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            #print("Mouse Coords Left:", e.pos().x(), 'x', e.pos().y())
            if self._ptz != None:
                #pa,ta,za = po,to,zo = self._ptz.get_ptz()
                #print(po,to,zo)
                self._ptz.relative_move((e.pos().x()-self.width()/2)/12.0,
                                        -(e.pos().y()-self.height()/2)/12.0, 0, 50)
                #pc,tc,zc = self._ptz.get_ptz()
                #while pc!=pa or tc!=ta or zc!=za:
                #    pa,ta,za = pc,tc,zc
                #    pc,tc,zc = self._ptz.get_ptz()
                #for i in range(100):
                #    (pan,tilt,zoom) = self._ptz.get_ptz()
                #    print("pan/tilt/zoom", pan,tilt,zoom)
            e.accept()
        if e.button() == Qt.RightButton:
            #print("Mouse Coords Right:", e.pos().x(), 'x', e.pos().y())
            #zoom = 1.0
            if self._ptz != None:
                self._ptz.center_move(e.pos().x(), e.pos().y(), 20)
                #(pan,tilt,zoom) = self._ptz.get_ptz()
                #print("pan/tilt/zoom", pan,tilt,zoom)
            e.accept()
            
    def wheelEvent(self, e):
        if self._ptz != None:
            self._ptz.area_zoom(e.pos().x(), e.pos().y(), e.angleDelta().y(), 30)
        #print(e.angleDelta().y())
        #if self._ptz != None:
        #    (pan,tilt,zoom) = self._ptz.get_ptz()
        #    print("pan/tilt/zoom", pan,tilt,zoom)
        e.accept()
    
    def setText(self, p_text):
        self._image_label.setText(p_text)
    
    def setPTZ(self, p_ptz=None):
        self._ptz = p_ptz
      
    def moveAbsolutePan(self, p_pan):
        if self._ptz != None:
            #print("Move Pan:", p_pan)
            self._ptz.absolute_move(p_pan, 0, 0, 100)
            
    def moveAbsoluteTilt(self, p_tilt):
        if self._ptz != None:
            #print("Move Tilt:", p_tilt)
            self._ptz.absolute_move(0, p_tilt, 0, 100)
    
    def moveAbsoluteZoom(self, p_zoom):
        if self._ptz != None:
            #print("Move Zoom:", p_zoom)
            self._ptz.absolute_move(0, 0, p_zoom, 100)
    
    def moveAbsolutePTZ(self, p_pan, p_tilt, p_zoom):
        if self._ptz != None:
            #print("Move Pan/Tilt/Zoom:", p_pan, p_tilt, p_zoom)
            self._ptz.absolute_move(p_pan, p_tilt, p_zoom, 100)
    
    def getCurrentPTZ(self):
        if self._ptz == None:
            return 0,0,0
        return self._ptz.get_ptz()
        
    #@pyqtSlot(object)
    def update_image(self, cv_img):
        qformat = QImage.Format_Indexed8
        if len(cv_img.shape) == 3:
            if cv_img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        qt_img = QImage(cv_img, cv_img.shape[1], cv_img.shape[0], cv_img.strides[0], qformat)
        qt_img = qt_img.rgbSwapped()
        #qt_img = qt_img.rgbSwapped().scaled(self._image_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self._image_label.setPixmap(QPixmap.fromImage(qt_img))
        #self._image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self._image_label.size(),
        #                                                             Qt.KeepAspectRatio,
        #                                                             Qt.SmoothTransformation))
        #self._image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.size(),
        #                                                             Qt.KeepAspectRatio))

class MQDockWidgetFilterView(QDockWidget):
    change_smart_record = pyqtSignal(bool) 
    def __init__(self, p_name, p_parent=None, p_func_init=None, p_func_run=None, *args,**kwargs):
        super(MQDockWidgetFilterView, self).__init__(p_name, p_parent)
        #QDockWidget.__init__(self, p_name, p_parent)
        self.setAllowedAreas(Qt.TopDockWidgetArea | Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._mainWidget = QWidget(self)
        self._widgetDisplay = MQWidgetVideoDisplay(self._mainWidget)
        #self.setMaximumWidth(p_parent.width()/3)
        #self.setMaximumHeight(p_parent.height()/3)
        self.setMaximumWidth(768)
        self.setMaximumHeight(512)
        #self._widgetDisplay.resize(p_parent.width()/3, p_parent.width()/3)
        self._widgetDisplay.setText('Video Filter Starting')
        #self._widgetDisplay.layout().setContentsMargins(QMargins(0,0,0,0))
        #self.setWidget(self._widgetDisplay)
        self._statusBar = QStatusBar(self._mainWidget)
        self._statusBar.showMessage("Ready")
        #self._statusBar.layout().setContentsMargins(QMargins(0,0,0,0))
        self._statusBar.setMaximumHeight(40)
        vbox = QVBoxLayout()
        vbox.addWidget(self._widgetDisplay)
        vbox.addWidget(self._statusBar)
        vbox.setContentsMargins(QMargins(0,0,0,0))
        # set the vbox layout as the widgets layout
        self._mainWidget.setLayout(vbox)
        self.setWidget(self._mainWidget)
        self._thread = MQThreadVideoFilter(p_func_init, p_func_run, args[0], kwargs)
        self._parent = p_parent
        self._parent._thread.change_pixmap_signal.connect(self._thread.setImage)
        self._thread.change_pixmap_signal.connect(self.update_image)
        self._FC = 0
        self._FPS = 25
        self._start_time = time.time()

    def update_image(self, p_img, p_flag=False):
        self._widgetDisplay.update_image(p_img)
        if p_flag == True:
            self.change_smart_record.emit(p_flag)
        self._FC = self._FC + 1
        if self._FC % self._FPS == 0:
            fps = round(self._FPS / (time.time() - self._start_time),2)
            #print("FPS Main: ", fps) # FPS = 1 / time to process loop
            self._start_time = time.time() # start time of the loop
            if self._statusBar.currentMessage() == '' or self._statusBar.currentMessage().startswith('FPS:'):
                self._statusBar.showMessage(f"FPS: {fps}")
    
    def close(self):
        self._parent._thread.change_pixmap_signal.disconnect(self._thread.setImage)
        self._thread.change_pixmap_signal.disconnect(self.update_image)
        self._thread.stop()
        
    def mousePressEvent(self, e):
        e.accept()
            
    def wheelEvent(self, e):
        e.accept()
        
class MQMainWindow(QMainWindow):
    def __init__(self, p_parent=None):
        super(MQMainWindow, self).__init__(p_parent)
        
        self.setWindowTitle("WindowMain VideoControl")
        self.resize(1920, 1280)
        
        self._workspace = QMdiArea(self)
        self._workspace_win = QMdiSubWindow(self._workspace)
        self._widgetDisplay = MQWidgetVideoDisplay(self._workspace_win)
        self._widgetDisplay.setWindowTitle("Window VideoDisplay")
        self._workspace_win.setWidget(self._widgetDisplay)
        self._workspace_win.resize(self.width()/2, self.height()/2)
        self._workspace.addSubWindow(self._workspace_win)
        self.setCentralWidget(self._workspace)
        
        
        self._dock_filter_view = None
        self._dock_filter_options = None
        self._dock_ptz_path = None
        self._dock_console = None
        self.create_actions()
        self.create_menus()
        self.create_tool_bars()
        self.create_status_bar()
        self.create_dock_windows()
        
        self._stream_out = None
        self._recording = False
        self._smart_recording = True
        self._displaying = False
        self._control = False
        self._thread = None
        self._timer = None
        #self._timer_run = False
        self._X = None
        self._filename_mp4_standard = self._dock_filter_options.getStandardMP4Filename()
        self._filename_mp4 = self._filename_mp4_standard
        self._filename_png = 'output.png'
        self._FPS = 25
        self._FC = 0
        self._start_time = time.time()

    def mousePressEvent(self, e):
        if not self._displaying or not self._control:
            e.accept()
            return
            
    def wheelEvent(self, e):
        if not self._displaying or not self._control:
            e.accept()
            return
    
    def resizeEvent(self, e):
        self.centerWorkspaceWindow()
        #if self._dock_filter_view != None:
        #    self._dock_filter_view.setMaximumWidth(self.width()/3)
        #    self._dock_filter_view.setMaximumHeight(self.height()/3)
        #self._widgetDisplay._image_label.resize(self.width()/3, self.height()/3)
    
    def closeEvent(self, e):
        self.start_record(False)
        self.start_stream(False,None,None,None,None)
        self._dock_filter_options.close()
        self._dock_ptz_path.close()
        if self._dock_filter_view != None:
            self._dock_filter_view.close()
            self._dock_filter_view = None
        if self._widgetDisplay != None:
            self._widgetDisplay.close()
            self._widgetDisplay = None
        self._workspace.removeSubWindow(self._workspace_win)
        self._workspace_win.close()
        self._workspace.close()
        super().closeEvent(e)
        e.accept()
    
    def close(self):
        self.start_record(False)
        self.start_stream(False,None,None,None,None)
        self._dock_filter_options.close()
        self._dock_ptz_path.close()
        if self._dock_filter_view != None:
            self._dock_filter_view.close()
            self._dock_filter_view = None
        if self._widgetDisplay != None:
            self._widgetDisplay.close()
            self._widgetDisplay = None
        self._workspace.removeSubWindow(self._workspace_win)
        self._workspace_win.close()
        self._workspace.close()
        super().close()        
        
    def start_stream(self, p_flag, p_url=None, p_ip=None, p_login=None, p_password=None):
        if p_flag == True:
            if not self._workspace_win.isVisible():
                self._workspace_win.show()
                self._widgetDisplay.show()
            if self._displaying:
                return
            # create the video capture thread and vapix ptz
            if not (p_ip == '' or p_login == '' or p_password == ''):
                try:
                    self._thread = MQThreadVideoCapture(p_url, self._FPS)
                except IOError:
                    self.statusBar().showMessage("Wrong credentials to open Video Stream", 4000)
                    return
                else:
                    self._X = vapix_control.CameraControl(p_ip, p_login, p_password)
                    self._control = True
                    if self._widgetDisplay != None:
                        self._widgetDisplay.setPTZ(self._X)
            else:
                if p_url != '':
                    self._thread = MQThreadVideoCapture(int(p_url), self._FPS)
                else:
                    return
                self._control = False
            # connect its signal to the update_image slot
            self._thread.change_pixmap_signal.connect(self.updateWorkspaceWindowSize)
            self._thread.change_pixmap_signal.connect(self.update_image)
            # start the thread
            self._thread.start()
            self._displaying = True
        else:
            if not self._displaying:
                return
            self.start_record(False)
            if self._control == True:
                self._X.stop_move()
                if self._widgetDisplay != None:
                    self._widgetDisplay.setPTZ(None)
                self._control = False
            if self._dock_filter_view != None:
                self._view_menu.removeAction(self._dock_filter_view.toggleViewAction())
                #self._thread.change_pixmap_signal.disconnect(self._dock_filter_view.update_image)
                self.removeDockWidget(self._dock_filter_view)
                self._dock_filter_view.close()
                self._dock_filter_view = None
            self._thread.stop()
            self._displaying = False
            if self._widgetDisplay != None:
                self._widgetDisplay.setText("Video Stream Off")
            
    def start_record(self, p_flag):
        if self._displaying and self._thread.isRunning():
            if p_flag == True:
                if self._recording:
                    return
                img = self._thread.read()
                self.statusBar().showMessage(f"Write mp4 Video Stream to {self._filename_mp4}", 4000)
                self._stream_out = cv2.VideoWriter(self._filename_mp4, cv2.VideoWriter_fourcc(*'MP4V'), 25, (img.shape[1],img.shape[0]))
                self._recording = True
                self._tool_bar.setStartCapture(True)
            else:
                if self._recording:
                    self.statusBar().showMessage(f"Stopped writing mp4 Video Stream to {self._filename_mp4}", 4000)
                    self._stream_out.release()
                    self._recording = False
                    self._tool_bar.setStartCapture(False)
                    self.set_video_filename(self._filename_mp4_standard)
                    if self._timer != None and self._timer.isActive():
                        self._timer.stop()
                        self._timer = None
        else:
            self.statusBar().showMessage(f"Need to start display first, to writing mp4 Video Stream to {self._filename_mp4}", 4000)
            self._tool_bar.setStartCapture(False)
           
    def pause_record(self,p_flag=False):
        if p_flag and self._recording:
            self.statusBar().showMessage(f"Timer: Restarted writing mp4 Video Stream to {self._filename_mp4}", 4000)
        elif not p_flag and self._recording:
            self.statusBar().showMessage(f"Timer: Paused writing mp4 Video Stream to {self._filename_mp4}", 4000)
        self._smart_recording = p_flag
        
    def image_save(self, p_filename):
        if  not self._displaying:
            return
        cv_img = self._thread.read()
        cv2.imwrite(p_filename, cv_img)
        
    def set_video_filename(self, p_filename):
        self._filename_mp4 = p_filename
    
    def centerWorkspaceWindow(self):
        if not self._workspace_win.isVisible():
           self._workspace_win.show()
           self._widgetDisplay.show()
        if self._workspace_win.isMaximized() or self._workspace_win.isMinimized():
            self._workspace_win.showNormal()     
        if self._workspace_win.isVisible():
            center = self._workspace.viewport().rect().center()
            geo = self._workspace_win.geometry()
            geo.moveCenter(center)
            self._workspace_win.setGeometry(geo)
        
    def updateWorkspaceWindowSize(self, cv_img = None):
        if not self._workspace_win.isVisible():
            self._workspace_win.show()
            self._widgetDisplay.show()
        if self._workspace_win.isVisible():
            self._workspace_win.resize(cv_img.shape[1], cv_img.shape[0])
            self.centerWorkspaceWindow()
        self._thread.change_pixmap_signal.disconnect(self.updateWorkspaceWindowSize)
                        
    #@pyqtSlot()
    def update_image(self, cv_img = None):
        if  not self._displaying:
            return
        self._FC = self._FC + 1
        if self._FC % self._FPS == 0:
            fps = round(self._FPS / (time.time() - self._start_time),2)
            #print("FPS Main: ", fps) # FPS = 1 / time to process loop
            self._start_time = time.time() # start time of the loop
            if self.statusBar().currentMessage() == '' or self.statusBar().currentMessage().startswith('FPS:'):
                self.statusBar().showMessage(f"FPS: {fps}", 1000)
        #cv_img = self._thread.read()
        if self._recording and self._smart_recording:
            self._stream_out.write(cv_img)
        if self._widgetDisplay != None:
            self._widgetDisplay.update_image(cv_img)
        
    def save(self):
        if not self._displaying:
            return
        dialog = QFileDialog(self, "Choose a file name")
        dialog.setMimeTypeFilters({"image/jpeg", # will show "JPEG image (*.jpeg *.jpg *.jpe)
                                   "image/png",  # will show "PNG image (*.png)"
                                   "video/mp4",
                                   "application/octet-stream" # will show "All files (*)"
                                   })
        dialog.setDefaultSuffix("png")
        dialog.selectFile(self._filename_png)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        
        if dialog.exec() != QDialog.Accepted:
            return

        filename = dialog.selectedFiles()[0]
        file = QFile(filename)
        if not file.open(QFile.WriteOnly | QFile.Text):
            reason = file.errorString()
            QMessageBox.warning(self, "WindowMain Video",
                    f"Cannot write file {filename}:\n{reason}.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
            self.image_save(filename)
        if filename.endswith('.mp4'):
            self.set_video_filename(filename)
        QApplication.restoreOverrideCursor()

        self.statusBar().showMessage(f"Saved '{filename}'", 4000)

    def about(self):
        QMessageBox.about(self, "Camera Viewer / Recorder",
                "The <b>Camera Viewer and Recorder</b> ...")

    def create_actions(self):
        icon = QIcon.fromTheme('document-save', QIcon(':/images/save.png'))
        self._save_act = QAction(icon, "&Save", self, shortcut="Ctrl+S",
                statusTip="Save the current Image or set Video capture filename", triggered=self.save)

        icon = QIcon.fromTheme('document-save', QIcon(':/images/quit.png'))
        self._quit_act = QAction(icon, "&Quit", self, shortcut="Ctrl+Q",
                statusTip="Quit the application", triggered=self.close)

        self._about_act = QAction("&About", self,
                statusTip="Show the application's About Box",
                triggered=self.about)

        self._about_qt_act = QAction("About &Qt", self,
                statusTip="Show the Qt library's About Box",
                triggered=QApplication.instance().aboutQt)

    def create_menus(self):
        self._file_menu = self.menuBar().addMenu("&File")
        #self._file_menu.addSeparator()
        self._file_menu.addAction(self._save_act)
        self._file_menu.addAction(self._quit_act)

        self._view_menu = self.menuBar().addMenu("&View")
        #self.menuBar().addSeparator()

        self._help_menu = self.menuBar().addMenu("&Help")
        self._help_menu.addAction(self._about_act)
        self._help_menu.addAction(self._about_qt_act)

    def create_tool_bars(self):
        self._file_tool_bar = self.addToolBar("File")
        self._file_tool_bar.addAction(self._quit_act)
        self._file_tool_bar.addAction(self._save_act)
        self._tool_bar = MQToolBarURL("Camera", self)
        self._tool_bar.start_camera.connect(self.start_stream)
        self._tool_bar.start_capture.connect(self.start_record)
        self.addToolBar(Qt.TopToolBarArea, self._tool_bar)
        
    def create_status_bar(self):
        self.statusBar().showMessage("Ready")

    def create_dock_windows(self):
        self._dock_filter_options = MQDockWidgetFilterOptions("Filter/Options", self)
        self._view_menu.addAction(self._dock_filter_options.toggleViewAction())
        self._dock_filter_options.change_filter_signal.connect(self.setFilter)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._dock_filter_options)
        self._dock_ptz_path = MQDockWidgetPTZPath("PTZ/Path", self)
        self._view_menu.addAction(self._dock_ptz_path.toggleViewAction())
        self._dock_ptz_path.setDimensionsAxisPan(QPointF(60000, 360), 1)
        self._dock_ptz_path.setColorPan(QColor(255, 0, 0, 150))
        self._dock_ptz_path.setDimensionsAxisTilt(QPointF(60000, 90), 2)
        self._dock_ptz_path.setColorTilt(QColor(0, 255, 0, 150))
        self._dock_ptz_path.setDimensionsAxisZoom(QPointF(60000, 9999), 0)
        self._dock_ptz_path.setColorZoom(QColor(0, 0, 255, 150))
        self.addDockWidget(Qt.LeftDockWidgetArea, self._dock_ptz_path)
        self.resizeDocks([self._dock_filter_options, self._dock_ptz_path], [768,768], Qt.Horizontal)
        self.resizeDocks([self._dock_filter_options, self._dock_ptz_path], [512,512], Qt.Vertical)
        self._dock_console = MQDockWidgetConsole("Filter/Console", self)
        self._view_menu.addAction(self._dock_console.toggleViewAction())
        self._dock_filter_options._model.mqObject.message.connect(self._dock_console._mainWidget.putData)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._dock_console)
        #self._dock_ptz_path._widgetPTZPan.ctrlPointCurrentPos.connect(self._widgetDisplay.moveAbsolutePan)
        #self._dock_ptz_path._widgetPTZTilt.ctrlPointCurrentPos.connect(self._widgetDisplay.moveAbsoluteTilt)
        #self._dock_ptz_path._widgetPTZZoom.ctrlPointCurrentPos.connect(self._widgetDisplay.moveAbsoluteZoom)
        self._dock_ptz_path.ctrlPointCurrent.connect(self._widgetDisplay.moveAbsolutePTZ)
        self._dock_ptz_path.ctrlSetPoint.connect(self.setCtrlPoint)
    
    def setCtrlPoint(self, p_x):
        if self._widgetDisplay != None:
            p,t,z = self._widgetDisplay.getCurrentPTZ()
            self._dock_ptz_path.setCtrlPoint(p_x, p, t, z)
        
    def setFilter(self, p_name, p_func_init, p_func_run, p_args):
        if not self._displaying:
            self.statusBar().showMessage("Need to start display first", 4000)
            return
        
        if p_name == 'Start Recording':
            if self._dock_filter_view != None and not self._recording and self._timer == None:    
                if p_args[0].endswith('.mp4'):
                    strlist = str(p_args[0]).split('.',1)
                    dateTime = str(datetime.now()).replace(' ', '_').replace(':','').split('.',1)
                    #print(strlist[0]+'_'+dateTime[0]+'.'+strlist[1])
                    self.set_video_filename(strlist[0]+'_'+dateTime[0]+'.'+strlist[1])
                    if p_args[1] > 0:
                        self._timer = QTimer(self)
                        self._timer.timeout.connect(self.pause_record)
                        self._timer.start(p_args[1])
                        self.start_record(True)
                        self.statusBar().showMessage(f"Started smart writing mp4 Video Stream to {self._filename_mp4}", 4000)
                return
            else:
                if self._timer != None and self._timer.isActive():
                    self._timer.stop()
                    self._timer = None
                    self.statusBar().showMessage(f"Stopped smart writing mp4 Video Stream to {self._filename_mp4}", 4000)
                    self.start_record(False)
                    self._smart_recording = True
                else:
                    if self._recording == True:
                        self.statusBar().showMessage(f"Already writing mp4 Video Stream to {self._filename_mp4}", 4000)
                    else:
                        self.statusBar().showMessage(f"Need to start a filter for smart writing mp4 Video Stream to {self._filename_mp4}", 4000)
                return

        if self._dock_filter_view != None:
            self.start_record(False)
            self._view_menu.removeAction(self._dock_filter_view.toggleViewAction())
            self._dock_filter_view.change_smart_record.disconnect(self.pause_record)
            self.removeDockWidget(self._dock_filter_view)
            self._dock_filter_view.close()
            self._dock_filter_view = None
            self.resizeDocks([self._dock_filter_options, self._dock_ptz_path], [768,768], Qt.Horizontal)
            self.resizeDocks([self._dock_filter_options, self._dock_ptz_path], [512,512], Qt.Vertical)
            return
        
        self._dock_filter_view = MQDockWidgetFilterView("Filter/Video - "+p_name, self, 
                                                        p_func_init,
                                                        p_func_run,
                                                        p_args)
        self._dock_filter_view._statusBar.showMessage(f"Filter: {p_name} starting ...", 4000)
        self._view_menu.addAction(self._dock_filter_view.toggleViewAction())
        self._dock_filter_view.change_smart_record.connect(self.pause_record)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._dock_filter_view)
        self.resizeDocks([self._dock_filter_options, self._dock_ptz_path, self._dock_filter_view], [768,768,768], Qt.Horizontal)
        self.resizeDocks([self._dock_filter_options, self._dock_ptz_path, self._dock_filter_view], [256,256,256], Qt.Vertical)
        self.centerWorkspaceWindow()
        #self._workspace.addSubWindow(self._dock_filter_view)
        
if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(':/images/logo.png'))
    with open("dedect.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)
    main_win = MQMainWindow()
    main_win.setWindowIcon(QIcon(':/images/logo.png'))
    main_win.show()
    sys.exit(app.exec())