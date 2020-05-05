# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:11:54 2020

@author: prasa
"""

from xml.etree.ElementTree import Element, SubElement, Comment
import xml.etree.cElementTree as ET
#from ElementTree_pretty import prettify
import cv2
import os
from pathlib import Path
from shutil import move
import re
import numpy as np

    
    
sourcepath = 'sample/'
dest_path = 'sample/'

ids = []
for file in os.listdir(sourcepath): #Save all images in a list
    filename = os.fsdecode(file)
    if filename.endswith('.jpg'):
        ids.append(filename[:-4])

for fname in ids[:]: 
        myfile = os.path.join(dest_path,fname +'.xml')
        myfile = Path(myfile)
#    if not myfile.exists(): #if file is not existing 
        txtfile = os.path.join(sourcepath, fname + '.txt') #Read annotation of each image from txt file
        f = open(txtfile,"r")
        imgfile = os.path.join(sourcepath, fname +'.jpg')
        img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED) #Read image to get image width and height
        w, h = img.shape[:2]
        
        top = Element('annotation')
        child = SubElement(top,'folder')
        child.text = 'open_images_volume'

        child_filename = SubElement(top,'filename')
        child_filename.text = fname +'.jpg'

        child_path = SubElement(top,'path')
        child_path.text = sourcepath + fname +'.jpg'

        child_source = SubElement(top,'source')
        child_database = SubElement(child_source, 'database')
        child_database.text = 'Unknown'

        child_size = SubElement(top,'size')
        child_width = SubElement(child_size,'width')
        child_width.text = str(img.shape[1])

        child_height = SubElement(child_size,'height')
        child_height.text = str(img.shape[0])

        child_depth = SubElement(child_size,'depth')
        if len(img.shape) == 3: 
            child_depth.text = str(img.shape[2])
        else:
            child_depth.text = '3'
        child_seg = SubElement(top, 'segmented')
        child_seg.text = '0'
        for x in f:     #Iterate for each object in a image. 
            x = list(x.split(','))
#            show(x[0], imgfile)
            x_min,  x_max, y_min, y_max = float(x[1]), float(x[2]), float(x[3]) ,float(x[4]) 
            
            x_ = int(x_min * img.shape[1])
            y_ = int(y_min* img.shape[0])
            
            x2 = int(x_max * img.shape[1])
            y2 =  int(y_max * img.shape[0])
            
            img = cv2.rectangle(img, (x_,y_), (x2,y2), (0,0,255), 2)

#            
            child_obj = SubElement(top, 'object')

            child_name = SubElement(child_obj, 'name')
            child_name.text = x[0] #name

            child_pose = SubElement(child_obj, 'pose')
            child_pose.text = 'Unspecified'

            child_trun = SubElement(child_obj, 'truncated')
            child_trun.text = '0'

            child_diff = SubElement(child_obj, 'difficult')
            child_diff.text = '0'

            child_bndbox = SubElement(child_obj, 'bndbox')


            child_xmin = SubElement(child_bndbox, 'xmin')
            child_xmin.text = str(x_) #xmin

            child_ymin = SubElement(child_bndbox, 'ymin')
            child_ymin.text = str(y_) #ymin

            child_xmax = SubElement(child_bndbox, 'xmax')
            child_xmax.text = str(x2) #xmax

            child_ymax = SubElement(child_bndbox, 'ymax')
            child_ymax.text = str(y2) #ymax
            
#        cv2.imshow('image', img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()     
#        cv2.imwrite('output/out{}.jpg'.format(fname),img)

        tree = ET.ElementTree(top)
        save = fname+'.xml'
        tree.write(save)
        move(fname+'.xml', myfile)
        print('done')
  
