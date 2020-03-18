import os
import cv2
import numpy as np
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

def crop_resize(img0, size=128, pad=16):
    """
    https://www.kaggle.com/iafoss/image-preprocessing-128x128
    """
    HEIGHT = 137
    WIDTH = 236

    def bbox(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    # inverse
    img = 255 - img0
    # normalize
    img = (img * 255.0 / img.max()).astype(np.uint8)

    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    # resize
    img = cv2.resize(img,(size,size))

    return img

def crop_resize2(img0, size=160, pad=16):
    """
    https://www.kaggle.com/iafoss/image-preprocessing-128x128
    """
    HEIGHT = 137
    WIDTH = 236

    def bbox(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    # inverse
    img = 255 - img0
    # normalize
    img = (img * 255.0 / img.max()).astype(np.uint8)

    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    # resize
    img = cv2.resize(img,(size,size), interpolation=cv2.INTER_AREA)

    return img

def center_resize(img0, h=96, w=168):
    """
    https://www.kaggle.com/iafoss/image-preprocessing-128x128
    """
    HEIGHT = 137
    WIDTH = 236

    def bbox(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    # inverse
    img = 255 - img0
    # normalize
    img = (img * 255.0 / img.max()).astype(np.uint8)

    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    #img = img[ymin:ymax,xmin:xmax]

    d_x = int((WIDTH - (xmax - xmin)) / 2)
    d_y = int((HEIGHT - (ymax - ymin)) / 2)

    back_img = np.zeros_like(img)
    back_img[d_y:d_y + (ymax - ymin), d_x:d_x + (xmax - xmin)] = img[ymin:ymax,xmin:xmax]
    img = back_img

    #remove lo intensity pixels as noise
    img[img < 28] = 0
    #lx, ly = xmax-xmin,ymax-ymin
    #l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    #img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    # resize
    img = cv2.resize(img,(w,h))

    return img

def resize_wrapper(h=96, w=168):
    def resize_f(img0):
        # resize
        img = cv2.resize(img0, (w,h))

        return img
    return resize_f

def add_gray_channel(imgs):
    return imgs[:,:,:,None]

def gray2rgb(image):
    if len(image.shape) == 2:
        return np.repeat(image[:,:,None], 3, axis=2)
    else:
        return np.repeat(image, 3, axis=2)

class Gray2Rgb:
    def __init__(self):
        return
    def apply(self, image):
        return gray2rgb(image)

class GrayToBinary:
    def __init__(self, thresh=50):
        self.thresh = thresh
        return
    def __call__(self, image):
        conv_img = image.copy()
        conv_img[image < self.thresh] = 0
        conv_img[image >= self.thresh] = 255
        return conv_img

class PreprocPipeline:
    def __init__(self):
        self.pl_name = 'pl_base'
        self.data_dir = '../input/train_data'
        self.image_path = os.path.join(self.data_dir, self.pl_name + '_images.npy')

        return

    def save_imgs(self, imgs):
        print('save ', self.image_path)
        np.save(self.image_path, imgs)
        return

    def load_imgs(self):
        print('load ', self.image_path)
        imgs = np.load(self.image_path)
        print('images shape ', imgs.shape)
        return imgs

    def apply_all_image(self, imgs, func):
        pp_imgs = []
        for img in tqdm(imgs):
            pp_imgs.append(func(img))
        pp_imgs = np.array(pp_imgs)
        return pp_imgs

class PreprocPipeline_v1(PreprocPipeline):
    def __init__(self):
        super(PreprocPipeline_v1, self).__init__()
        self.pl_name = 'pl_v1'
        self.data_dir = '../input/train_data'
        self.image_path = os.path.join(self.data_dir, self.pl_name + '_images.npy')

        return

    def preprocessing(self, imgs):
        pp_imgs = self.apply_all_image(imgs, crop_resize)
        pp_imgs = add_gray_channel(pp_imgs)

        return pp_imgs

class PreprocPipeline_v2(PreprocPipeline):
    def __init__(self):
        super(PreprocPipeline_v2, self).__init__()
        self.pl_name = 'pl_v2'
        self.data_dir = '../input/train_data'
        self.image_path = os.path.join(self.data_dir, self.pl_name + '_images.npy')

        return

    def preprocessing(self, imgs):
        pp_imgs = self.apply_all_image(imgs, resize_wrapper(h=96, w=168))
        pp_imgs = add_gray_channel(pp_imgs)

        return pp_imgs

class PreprocPipeline_v3(PreprocPipeline):
    def __init__(self):
        super(PreprocPipeline_v3, self).__init__()
        self.pl_name = 'pl_v3'
        self.data_dir = '../input/train_data'
        self.image_path = os.path.join(self.data_dir, self.pl_name + '_images.npy')

        return

    def preprocessing(self, imgs):
        pp_imgs = self.apply_all_image(imgs, center_resize)
        pp_imgs = add_gray_channel(pp_imgs)

        return pp_imgs

class PreprocPipeline_v4(PreprocPipeline):
    def __init__(self):
        super(PreprocPipeline_v4, self).__init__()
        self.pl_name = 'pl_v4'
        self.data_dir = '../input/train_data'
        self.image_path = os.path.join(self.data_dir, self.pl_name + '_images.npy')

        return

    def preprocessing(self, imgs):
        pp_imgs = self.apply_all_image(imgs, crop_resize2)
        pp_imgs = add_gray_channel(pp_imgs)

        return pp_imgs

class PreprocPipeline_v5(PreprocPipeline):
    def __init__(self):
        super(PreprocPipeline_v5, self).__init__()
        self.pl_name = 'pl_v5'
        self.data_dir = '../input/train_data'
        self.image_path = os.path.join(self.data_dir, self.pl_name + '_images.npy')

        return

    def preprocessing(self, imgs):
        pp_imgs = add_gray_channel(imgs)

        return pp_imgs