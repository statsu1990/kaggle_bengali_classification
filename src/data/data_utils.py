import os
import gc
import numpy as np
import pandas as pd

class Config:
    data_path = '../input/bengaliai-cv19'

def get_image(type_is_train, height=137, width=236, data_idxs=[0, 1, 2, 3]):
    data_type = 'train' if type_is_train else 'test'

    print('read image')
    image_df_list = [pd.read_parquet(os.path.join(Config.data_path, f'{data_type}_image_data_{i}.parquet')) for i in data_idxs]
    images = [df.iloc[:, 1:].values.reshape(-1, height, width) for df in image_df_list]

    del image_df_list
    gc.collect()
    
    images = np.concatenate(images, axis=0)

    print('image shape', images.shape)
    return images

def get_train_label():
    print('read train label')
    path = os.path.join(Config.data_path, 'train.csv')
    df = pd.read_csv(path)
    print('df size ', len(df))
    print(df.head())

    label = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
    return label

def get_label_with_unique(labels):
    """
    Args:
        label = shape (N, 3)
    Returns:
        label = shape (N, 4)
    """
    uniq_labels = np.unique(labels, axis=0)
    uniq_labels_dict = {tuple(lb.tolist()):id for id, lb in enumerate(uniq_labels) }

    labels_with_uniq_labels = []

    for lb in labels:
        lb_ls = lb.tolist()
        labels_with_uniq_labels.append(lb_ls + [uniq_labels_dict[tuple(lb_ls)]])

    labels_with_uniq_labels = np.array(labels_with_uniq_labels, dtype='int64')

    return labels_with_uniq_labels