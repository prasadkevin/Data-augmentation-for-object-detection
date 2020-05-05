# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:35:12 2020

@author: prasad
## augmentation of images along with bounding boxes 
"""

#import caffe
import numpy as np
import cv2
import os
import glob, random
import os.path
from os import path
import imgaug.augmenters as iaa
from data_aug.data_aug import *
from data_aug.bbox_util import *

import xml.etree.ElementTree as ET

scale=(0, 60)
aug1 = iaa.Add((-40, 40))
aug2 = iaa.AddElementwise((-40, 40), per_channel=0.5)
aug3 = aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))

aug4 = iaa.AdditiveLaplaceNoise(scale=(0, 0.2*255))
aug5 = iaa.AdditivePoissonNoise(scale)
aug6 = iaa.Multiply((0.5, 1.5), per_channel=0.5)

aug10 = iaa.Dropout(p=(0, 0.2))
aug11 = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.2))
aug12 = iaa.Dropout2d(p=0.2, nb_keep_channels=0)
aug13 = iaa.ImpulseNoise(0.2)
aug14 = iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.6))
aug15 = iaa.CoarsePepper(0.05, size_percent=(0.01, 0.2))
aug16 = iaa.Invert(0.25, per_channel=0.4)
aug17 = iaa.Invert(0.1)
aug18 = iaa.Solarize(0.2, threshold=(32, 128))
aug19 = iaa.JpegCompression(compression=(70, 99))
aug20 = iaa.GaussianBlur(sigma=(0.0, 3.0)) # blur
aug21 = iaa.AverageBlur(k=((5, 11), (1, 3)))

aug22 = iaa.MotionBlur(k=10)

aug23 = iaa.BlendAlpha(
    (0.0, 0.5),
    iaa.Affine(rotate=(-20, 20)),
    per_channel=0.5)

aug25 = iaa.BlendAlphaMask(
    iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),
    iaa.Clouds()
)

aug26 = iaa.BlendAlphaElementwise(
    (0.0, 0.5),
    iaa.Affine(rotate=(-20, 20)),
    per_channel=0.5)

aug27 = iaa.BlendAlphaElementwise(
    (0.0, 0.2),
    foreground=iaa.Add(20),
    background=iaa.Multiply(0.4))
aug28 = iaa.BlendAlphaElementwise([0.25, 0.75], iaa.MedianBlur(13))

aug29 = iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(0.6))
aug30 = iaa.BlendAlphaSimplexNoise(
    iaa.EdgeDetect(0.2),
    upscale_method="nearest")
#aug31 = iaa.Cutout(fill_mode="constant", cval=(0, 255),
#                 fill_per_channel=0.5)
aug32 = iaa.BlendAlphaHorizontalLinearGradient(
    iaa.TotalDropout(0.6),
    min_value=0.2, max_value=0.8)

aug33 = iaa.BlendAlphaHorizontalLinearGradient(
    iaa.AveragePooling(11),
    start_at=(0.0, 0.6), end_at=(0.0, 0.6))

aug34 = iaa.BlendAlphaVerticalLinearGradient(
    iaa.TotalDropout(0.6),
    min_value=0.2, max_value=0.8)

aug35 = iaa.BlendAlphaVerticalLinearGradient(
    iaa.AveragePooling(9),
    start_at=(0.0, 0.4), end_at=(0.0, 0.4))

aug36 = iaa.BlendAlpha(
    (0.0, 0.3),
    iaa.Affine(rotate=(-10, 10)),
    per_channel=0.3)


aug38 =  iaa.WithColorspace(
    to_colorspace="HSV",
    from_colorspace="RGB",
    children=iaa.WithChannels(0,iaa.Add((0, 50))))
aug40 = iaa.WithHueAndSaturation(
    iaa.WithChannels(0, iaa.Add((0, 50))))
aug41 = iaa.MultiplyHueAndSaturation((0.5, 1.9), per_channel=True)
aug42 = iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
aug43 = iaa.AddToHue((-50, 50))
aug44 = iaa.AddToSaturation((-50, 50))
aug45 = iaa.Sequential([
    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
    iaa.WithChannels(0, iaa.Add((50, 100))),
    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")])
    
aug46 = iaa.Grayscale(alpha=(0.0, 1.0))
aug47 = iaa.ChangeColorTemperature((1100, 10000))
aug49 = iaa.UniformColorQuantization()
aug50 = iaa.UniformColorQuantizationToNBits()
aug51 = iaa.GammaContrast((0.5, 2.0), per_channel=True)
aug52 = iaa.SigmoidContrast(
    gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)
aug53 = iaa.LogContrast(gain=(0.6, 1.4), per_channel=True)
aug54 = iaa.LinearContrast((0.4, 1.6), per_channel=True)
aug55 = iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)
aug56 = iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization())
aug57 = iaa.HistogramEqualization(
    from_colorspace=iaa.HistogramEqualization.BGR,
    to_colorspace=iaa.HistogramEqualization.HSV)

aug58  = iaa.DirectedEdgeDetect(alpha=(0.0, 0.5), direction=(0.0, 0.5))
aug59 = iaa.Canny(
    alpha=(0.0, 0.3),
    colorizer=iaa.RandomColorsBinaryImageColorizer(
        color_true=255,
        color_false=0
    )
)
    

    ##################### for augment 2

def drawBox(image, bboxes, string_):
    for i in range(len(bboxes)):
        # changed color and width to make it visible
        cv2.rectangle(image, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])), (255, 0, 0), 1 )
#    cv2.imshow(string_, image)
    cv2.imwrite('aug_output/{}'.format(string_), image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


def resize_img_bbox(image, bbox, targetSize):
    # image = cv2.imread("img.jpg", 3)
    
    print(image.shape)

    # Note: flipped comparing to your original code!
    # x_ = image.shape[0]
    # y_ = image.shape[1]
    y_ = image.shape[0]
    x_ = image.shape[1]

    x_scale = targetSize / x_
    y_scale = targetSize / y_
    print(x_scale, y_scale)
    img = cv2.resize(image, (targetSize, targetSize));
    print(img.shape)
    img = np.array(img);

    # original frame as named values
    resized_bboxes = []
    for i in range(len(bbox)):
        (origLeft, origTop, origRight, origBottom) = bboxes[i]
        x = int(np.round(origLeft * x_scale))
        y = int(np.round(origTop * y_scale))
        xmax = int(np.round(origRight * x_scale))
        ymax = int(np.round(origBottom * y_scale))
        resized_bboxes.append([x, y, xmax, ymax])
    return img, resized_bboxes


def aug_imgaug(aug_string, aug, image):

    image2 = image.copy()
    image2 = np.expand_dims(image2, axis=0)
    images_aug = aug(images = image2)
    
#    cv2.imshow(aug_string , images_aug[0])
#    cv2.waitKey(0)
#    cv2.imwrite('aug_output/{}_{}.jpg'.format(aug_string,image_path.split('/')[1].split('.jpg')[0]), images_aug[0])
    return images_aug, aug_string


def augment1(image,bboxes):
    image_array, bboxes_array, aug_string = [] , [], []
    
    aug_imgaug_1, aug_string_1 = aug_imgaug('aug1_',aug1, image)
    aug_imgaug_2, aug_string_2 = aug_imgaug('aug2_',aug2, image)
    aug_imgaug_3, aug_string_3 = aug_imgaug('aug3_',aug3, image)
    aug_imgaug_4, aug_string_4 = aug_imgaug('aug4_',aug4, image)
    aug_imgaug_5, aug_string_5 = aug_imgaug('aug5_',aug5, image)
    aug_imgaug_6, aug_string_6 = aug_imgaug('aug6_',aug6, image)
    aug_imgaug_10, aug_string_10 = aug_imgaug('aug10_',aug10, image)
    aug_imgaug_11, aug_string_11 = aug_imgaug('aug11_',aug11, image)
    aug_imgaug_12, aug_string_12 = aug_imgaug('aug12_',aug12, image)
    aug_imgaug_13, aug_string_13 = aug_imgaug('aug13_',aug13, image)
    aug_imgaug_14, aug_string_14 = aug_imgaug('aug14_',aug14, image)
    aug_imgaug_15, aug_string_15 = aug_imgaug('aug15_',aug15, image)
    aug_imgaug_16, aug_string_16 = aug_imgaug('aug19_',aug19, image)
    aug_imgaug_17, aug_string_17 = aug_imgaug('aug20_',aug20, image)
    aug_imgaug_18, aug_string_18 = aug_imgaug('aug21_',aug21, image)
    aug_imgaug_19, aug_string_19 = aug_imgaug('aug22_',aug22, image)
    aug_imgaug_20, aug_string_20 = aug_imgaug('aug23_',aug23, image)
    aug_imgaug_22, aug_string_22 = aug_imgaug('aug25_',aug25, image)
    aug_imgaug_23, aug_string_23 = aug_imgaug('aug26_',aug26, image)
    aug_imgaug_24, aug_string_24 = aug_imgaug('aug27_',aug27, image)
    aug_imgaug_25, aug_string_25 = aug_imgaug('aug28_',aug28, image)
    aug_imgaug_26, aug_string_26 = aug_imgaug('aug29_',aug29, image)
    aug_imgaug_27, aug_string_27 = aug_imgaug('aug30_',aug30, image)
    aug_imgaug_29, aug_string_29 = aug_imgaug('aug32_',aug32, image)
    aug_imgaug_30, aug_string_30 = aug_imgaug('aug34_',aug34, image)
    aug_imgaug_32, aug_string_32 = aug_imgaug('aug36_',aug36, image)
    aug_imgaug_33, aug_string_33 = aug_imgaug('aug38_',aug38, image)
    aug_imgaug_34, aug_string_34 = aug_imgaug('aug41_',aug41, image)
    aug_imgaug_35, aug_string_35 = aug_imgaug('aug42_',aug42, image)
    aug_imgaug_36, aug_string_36 = aug_imgaug('aug43_',aug43, image)
    aug_imgaug_37, aug_string_37 = aug_imgaug('aug44_',aug44, image)
    aug_imgaug_38, aug_string_38 = aug_imgaug('aug45_',aug45, image)
    aug_imgaug_39, aug_string_39 = aug_imgaug('aug46_',aug46, image)
    aug_imgaug_40, aug_string_40 = aug_imgaug('aug47_',aug47, image)
    aug_imgaug_41, aug_string_41 = aug_imgaug('aug48_',aug49, image)
    aug_imgaug_42, aug_string_42 = aug_imgaug('aug49_',aug50, image)
    aug_imgaug_43, aug_string_43 = aug_imgaug('aug51_',aug51, image)
    aug_imgaug_44, aug_string_44 = aug_imgaug('aug52_',aug52, image)
    aug_imgaug_45, aug_string_45 = aug_imgaug('aug53_',aug53, image)
    aug_imgaug_46, aug_string_46 = aug_imgaug('aug54_',aug54,image)
    aug_imgaug_47, aug_string_47 = aug_imgaug('aug55_',aug55, image)
    aug_imgaug_48, aug_string_48 = aug_imgaug('aug56_',aug56, image)
    aug_imgaug_49, aug_string_49 = aug_imgaug('aug57_',aug57, image)
    aug_imgaug_58, aug_string_58 = aug_imgaug('aug58_',aug58, image)
    aug_imgaug_59, aug_string_59 = aug_imgaug('aug59_',aug59, image)
    
    ## adding all the images list into a single array 
    image_array.extend((aug_imgaug_1, aug_imgaug_2, aug_imgaug_3, aug_imgaug_4, aug_imgaug_5, aug_imgaug_6, aug_imgaug_10, 
                        aug_imgaug_11, aug_imgaug_12, aug_imgaug_13, aug_imgaug_14, aug_imgaug_15, aug_imgaug_16,
                        aug_imgaug_17, aug_imgaug_18, aug_imgaug_19, aug_imgaug_20,  aug_imgaug_22, 
                        aug_imgaug_23, aug_imgaug_24, aug_imgaug_25, aug_imgaug_26, aug_imgaug_27, aug_imgaug_29, 
                        aug_imgaug_30, aug_imgaug_32, aug_imgaug_33, aug_imgaug_34, aug_imgaug_35,
                        aug_imgaug_36, aug_imgaug_37, aug_imgaug_38, aug_imgaug_39, aug_imgaug_40, aug_imgaug_41, 
                        aug_imgaug_42, aug_imgaug_43, aug_imgaug_44, aug_imgaug_45, aug_imgaug_46, aug_imgaug_47,
                        aug_imgaug_48, aug_imgaug_49, aug_imgaug_58, aug_imgaug_59))
    image_array = np.array(image_array)
    image_array = image_array[:, 0, :, :, :] # finaly dimesion is (51,250,250,3) , 51 augmented images
    
    for i in range(len(image_array)):  # to add bboxes of the same size 51
        bboxes_array.append(bboxes)
    bboxes_array = np.array(bboxes_array)
    
    aug_string.extend((aug_string_1 ,aug_string_2 ,aug_string_3 ,aug_string_4 ,aug_string_5 ,aug_string_6 ,aug_string_10 ,
                       aug_string_11 ,aug_string_12 ,aug_string_13 ,aug_string_14 ,aug_string_15 ,aug_string_16 ,aug_string_17, 
                       aug_string_18 ,aug_string_19 ,aug_string_20 ,aug_string_22 ,aug_string_23 ,aug_string_24, 
                       aug_string_25 ,aug_string_26 ,aug_string_27 ,aug_string_29 ,aug_string_30 ,aug_string_32, 
                       aug_string_33 ,aug_string_34 ,aug_string_35 ,aug_string_36 ,aug_string_37 ,aug_string_38 ,aug_string_39, 
                       aug_string_40 ,aug_string_41 ,aug_string_42 ,aug_string_43 ,aug_string_44 ,aug_string_45 ,aug_string_46, 
                       aug_string_47 ,aug_string_48 ,aug_string_49 ,aug_string_58 ,aug_string_59))
    aug_string = np.array(aug_string)
    print('image_array shape>>>> {} /n bboxes_aray shape >>>>> {} /n aug_string shape>>>>>>>> {}'.format(image_array.shape, bboxes_array.shape, aug_string.shape))
    return image_array, bboxes_array, aug_string ## return 51 img array and 51 bboxes array 

################# augmentation starts
def flip(image, bboxes):
    img_, bboxes_ = RandomHorizontalFlip(1)(image.copy(), bboxes.copy())
    return img_, bboxes_


def rotate(img, bboxes):
    
    img_, bboxes_ = RandomShear(0.2)(img.copy(), bboxes.copy())
    return img_, bboxes_


def scale_rotate(img, bboxes):
    seq = Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear()])
    img_, bboxes_ = seq(img.copy(), bboxes.copy())
    return img_, bboxes_


def augment2(image, bboxes, augment_type_counter):

    if augment_type_counter == 0 or augment_type_counter == 4 or augment_type_counter == 7 :   
        return  flip(image, bboxes)        
    elif augment_type_counter == 1 or augment_type_counter == 5 or augment_type_counter == 4 :   
        return rotate(image, bboxes)
    elif augment_type_counter == 2 or augment_type_counter == 6:
        return scale_rotate(image, bboxes)

    
    
def read_bbox(xml_file):
    bboxes = []
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        
        bndbox = member.find('bndbox')

        filename, class_name, xmin, ymin, xmax, ymax = (root.find('filename').text, member.find('name').text,
                                                        bndbox.find('xmin').text, bndbox.find('ymin').text,
                                                        bndbox.find('xmax').text, bndbox.find('ymax').text)
        bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        
    return filename, class_name, bboxes


#################3 main 
out_image_size = 250 # size of the output image
images = glob.glob('test_aug_samples/*.jpg') ## input images path list
noises = ["s&p", "gauss", "speckle", "poisson"]
for img_path in images[:59]:
    image = cv2.imread(img_path)
    head,tail = os.path.split(img_path)
    xml_file = tail[:-4] 
    xml_path = head + '/' + xml_file + ".xml"
    if path.exists(xml_path):
        
        filename, class_name, bboxes = read_bbox(xml_path)
        # resizing image and bounding bboxes
        image, bboxes = resize_img_bbox(image, bboxes, out_image_size)
        # To display the image and bounding boxes
        drawBox(image.copy(), bboxes.copy(), 'resized image and boxx' + tail)
#         applying augmentation 2
        image2, bboxes2, aug_string = augment1(image.copy(), bboxes.copy()) # returns a list of images and bbounding boxes along with aug string
        for i in range(len(image2)):
            print('string>>>>>>>>>>>>>>>>>>>>>>>', aug_string[i])
            drawBox(image2[i].copy(), bboxes2[i].copy(), aug_string[i] + tail)
#            # augmentation 2
       
        bboxes2 = np.array(bboxes2, dtype="float64" )
        
        
        for i in range(40):
            randomized_val = random.randint(0, 43)
            
            image3, bboxes3 = augment2(image2[randomized_val], bboxes2[randomized_val],random.randint(0,2)) # returns a single image and bounding boxes
            print(i)
            drawBox(image3.copy(), bboxes3.copy(),str(i) + tail)





    
    
    
    