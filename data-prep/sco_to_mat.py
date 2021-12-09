import os
import glob
import scipy.io as sio
import tarfile
#import os
import io
import numpy as np
from PIL import Image
from datetime import datetime
from matplotlib import pyplot as plt

#img_folder = r'A:\ILSVRC14\ILSVRC2012_img_train'
img_folder = r"A:\IsKnown_Images\Aff_NE_Balanced\Bal_v14\Ind-0\*\*\*.jpg"

cnt_classes_to_add = 40

img_size=128

now = datetime.now()
data_arr_for_mat = np.empty((0,img_size,img_size,3),dtype=np.uint8)
labels_arr_for_mat = []
all_filenames = glob.glob(img_folder)
class_names = np.unique ( [ filename.split('\\')[-2] for filename in all_filenames ] ).tolist()
class_names = class_names[:cnt_classes_to_add]

filtered_filename_inds = [ filename.split("\\")[-2] in class_names for filename in all_filenames ]
all_filenames = np.asarray(all_filenames)[ filtered_filename_inds ]

cnt_images_planned = len(all_filenames) # actual count may differ if some images can't be added
cntr_images_added = 0

data_arr_for_mat = np.empty((cnt_images_planned,img_size,img_size,3),dtype=np.uint8)

concat_time = 0

#tar_filename = r'A:\ILSVRC14\ILSVRC2012_img_train\n01440764.tar'
for cntr, filename in enumerate(all_filenames):

    class_id = class_names.index(filename.split("\\")[-2])
    #if class_id>=cnt_classes_to_add:
    #    continue

    image = Image.open(filename)

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
        print ("file.dom!=3: {}".format(filename))
    #break

    cntr +=1
    if cntr%100==0:
        print("Ctr:{},class:{},sec:{}".format(cntr,class_id,(datetime.now() - now).seconds))
        #print("Ctr:{},class:{},sec:{}, concat_time:{}".format(cntr,class_id,(datetime.now() - now).seconds, concat_time/1000000))
        concat_time = 0

print (image_arr.shape)
print(data_arr_for_mat.shape)
print( np.array(labels_arr_for_mat).reshape(1,-1).shape )

mat_dic = {"labels": np.array(labels_arr_for_mat).reshape(1,-1),
           "data": data_arr_for_mat[:cntr_images_added,:,:,:] }

sio.savemat (r'D:\Labs\BigGAN-tensorflow.MingtaoGuo\BigGAN-tensorflow\dataset\sco_{}.mat'.format(img_size), mat_dic)
plt.imshow(image_arr)
plt.show()

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