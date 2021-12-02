import os

import scipy.io as sio
import tarfile
#import os
import io
import numpy as np
from PIL import Image
from datetime import datetime
from matplotlib import pyplot as plt

img_folder = r'A:\ILSVRC14\ILSVRC2012_img_train'

img_size=256
now = datetime.now()
data_arr_for_mat = np.empty((0,img_size,img_size,3),dtype=np.uint8)
labels_arr_for_mat = []
tar_filenames = os.listdir(img_folder) [:40]    #top 40
cnt_images_planned = len(tar_filenames) * 1300 # 1300 - number of images  per class; actual count may differ if some images can't be added
cntr_images_added = 0

data_arr_for_mat = np.empty((cnt_images_planned,img_size,img_size,3),dtype=np.uint8)

concat_time = 0


#tar_filename = r'A:\ILSVRC14\ILSVRC2012_img_train\n01440764.tar'
for class_id, tar_filename in enumerate(tar_filenames):
    cntr = 0
    #tar_filecontents = tarfile.open(tar_filename)

    with tarfile.open(os.path.join(img_folder,tar_filename)) as tar_filecontents:
        for tarinfo in tar_filecontents.getmembers():
            image = tar_filecontents.extractfile(tarinfo)
            image = image.read()
            image = Image.open(io.BytesIO(image))

            # crop central
            image_min_dim = np.min(image.size)
            x_from, y_from = int((image.size[0]-image_min_dim)/2), int((image.size[1]-image_min_dim)/2)
            x_to, y_to = x_from + image_min_dim, y_from + image_min_dim
            #image_arr = np.asarray(image) [ x_from:x_to, y_from:y_to , :]
            image = image.crop((x_from, y_from, x_to, y_to))

            # to target size
            image = image.resize((img_size,img_size))

            # concatenate to full array
            image_arr = np.asarray(image)
            if len(image_arr.shape)==3:
                #now1 = datetime.now()
                data_arr_for_mat[cntr_images_added,:,:,:] =  image_arr
                #data_arr_for_mat = np.concatenate ( (data_arr_for_mat, np.expand_dims(image_arr,0) ), axis=0 )
                #concat_time = concat_time + (datetime.now() - now1).seconds*1000000 + (datetime.now() - now1).microseconds
                labels_arr_for_mat.append( np.float64(class_id) )
                cntr_images_added += 1
            else:
                print ("file.dom!=3: {}".format(tarinfo.name))
            #break

            cntr +=1
            if cntr%100==0:
                print("Ctr:{},class:{},sec:{}".format(cntr,class_id,(datetime.now() - now).seconds))
                #print("Ctr:{},class:{},sec:{}, concat_time:{}".format(cntr,class_id,(datetime.now() - now).seconds, concat_time/1000000))
                concat_time = 0

plt.imshow(image_arr)
plt.show()
print (image_arr.shape)
print(data_arr_for_mat.shape)
print( np.array(labels_arr_for_mat).reshape(1,-1).shape )

mat_dic = {"labels": np.array(labels_arr_for_mat).reshape(1,-1),
           "data": data_arr_for_mat[:cntr_images_added,:,:,:] }

sio.savemat (r'D:\Labs\BigGAN-tensorflow.MingtaoGuo\BigGAN-tensorflow\dataset\imagenet_{}.mat'.format(img_size), mat_dic)

#    with tarfile.open(r'A:\ILSVRC14\ILSVRC2012_img_train\n01440764.tar') as tf:
#        tarinfo = tf.getmember('n01440764_2708.JPEG')
#        image = tf.extractfile(tarinfo)
#        image = image.read()
#        image = Image.open(io.BytesIO(image))

#mat_filename = r'D:\Labs\BigGAN-tensorflow.MingtaoGuo\BigGAN-tensorflow\dataset\imagenet64'
#mat = sio.loadmat(mat_filename)
#mat.keys()
#Out[56]: dict_keys(['__header__', '__version__', '__globals__', 'labels', 'data'])
#mat['data'].shape
#Out[57]: (50869, 64, 64, 3)