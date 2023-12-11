import math
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import Dataset
import os
from PIL import Image
from natsort import natsorted
import subsetsum as sb


class OCTDataset(Dataset):

    def __init__(self, data_dir, img_type="L", transform=None, img_paths=None, nst_path=None,
                 nst_prob=0, **kwargs):
        self.data_dir = data_dir.replace("\\", "/")  # root directory
        self.transform = transform  # transform functions
        self.img_type = img_type  # the type of image L, R
        self.nst_prob = nst_prob  # the probability of using NST generated imgs
        self.nst_path = nst_path  # path to the nst file
        self.kwargs = kwargs
        self.img_paths = img_paths

    def __getitem__(self, index):
        img_path, label = self.img_paths[index]
        img_path = img_path.replace("\\", "/")  # fixing the path for windows os
        img_name = img_path.split(self.data_dir)[1].split("/")[-1]
        img_view = self.load_img(img_path)  # return an image
        if self.transform is not None:
            img_view = self.transform(img_view)
        results = (img_view, label)
        return results

    def __len__(self):
        """
        The __len__ should be overridden when the parent is Dataset to return the number of elements of the dataset

        Return:
            - returns the size of the dataset
        """
        return len(self.img_paths)

    def load_img(self, img_path):
        img = Image.open(img_path).convert(self.img_type)
        return img

    def load_nst_img(self, img_name, nst_count=3):
        """
        nst_count (int): number of NST generated images for each OCT
        """
        # Directory where NST images are stored / file name
        directory_path = os.path.join(self.nst_path, img_name.split('-')[0])
        # Construct the prefix of the image names you're looking for
        prefix = img_name[:-5]
        nst_img_paths = [os.path.join(directory_path, prefix + ".jpeg_" + str(i) + ".jpg") for i in range(nst_count)]
        # Randomly select one of the matched files
        selected_img_path = np.random.choice(nst_img_paths)
        # randomly select n_view of them and transform the using the transformation
        img = Image.open(selected_img_path).convert(self.img_type)
        return img


def get_srinivasan_imgs(data_dir: str, **kwargs):
    """

    :param data_dir:
    :param kwargs:
        - ignore_folders (np.array): indices of files to ignore
        - classes (list of tuples): ex: [("NORMAL", 0), ...]
    :return:
    """
    classes = kwargs["classes"]
    img_filename_list = []
    for c in classes:
        img_filename_list += list(filter(lambda k: c[0] in k, os.listdir(os.path.join(data_dir))))

    imgs_path = []
    for img_file in img_filename_list:
        if any(item == int(img_file.replace("AMD", "").replace("NORMAL", "").replace("DME", ""))
               for item in kwargs["ignore_folders"]):
            continue
        folder = os.path.join(data_dir, img_file, "TIFFs/8bitTIFFs")
        imgs_path += [(os.path.join(folder, id), get_class(os.path.join(folder, id), kwargs["classes"]))
                      for id in os.listdir(folder)]
    return imgs_path


def get_kermany_imgs(data_dir: str, **kwargs):
    # make sure the sum of split is 1
    split = kwargs["split"]
    classes = kwargs["classes"]

    split = np.array(split)
    assert (split.sum() == 1)

    img_paths = [[] for _ in split]
    img_filename_list = []
    path = os.listdir(os.path.join(data_dir))
    # filter out the files not inside the classes
    for c in classes:
        img_filename_list += list(filter(lambda k: c[0] in k, path))
    for img_file in img_filename_list:  # iterate over each class
        img_file_path = os.path.join(data_dir, img_file)
        # sort lexicographically the img names in the img_file directory
        img_names = natsorted(os.listdir(img_file_path))
        img_dict = {}
        # patient-wise dictionary
        for img in img_names:
            img_num = img.split("-")[2]  # the number associated to each img
            img_name = img.replace(img_num, "")
            if img_name in img_dict:
                img_dict[img_name] += [img_num]
            else:
                img_dict[img_name] = [img_num]
        selected_keys = set()  # Keep track of images that have already been added
        copy_split = split.copy()
        for i, percentage in enumerate(copy_split):
            # create a list of #visits of clients that has not been selected
            num_visits = [len(img_dict[key]) for key in img_dict if key not in selected_keys]
            total_imgs = sum(num_visits)
            selected_num = math.ceil(total_imgs * (percentage))
            subset = []
            for solution in sb.solutions(num_visits, selected_num):
                # `solution` contains indices of elements in `nums`
                subset = [i for i in solution]
                break
            keys = [key for key in img_dict if key not in selected_keys]
            for idx in subset:
                selected_subset = [(img_file_path + "/" + keys[idx] + count,
                                    get_class(img_file_path + keys[idx], classes)) for count in img_dict[keys[idx]]]
                img_paths[i] += selected_subset
                selected_keys.add(keys[idx])  # Mark this key as selected
            if len(copy_split) > i + 1:
                for j in range(i + 1, len(copy_split)):
                    copy_split[j] += percentage / (len(copy_split) - (i + 1))
    return img_paths


def get_class(img_name, classes: list):
    for c, v in classes:
        if c in img_name:
            return v


def get_oct500_imgs(data_dir: str, **kwargs):
    assert sum(kwargs["split"]) == 1
    classes = kwargs["classes"]
    mode = kwargs["mode"]
    split = kwargs["split"]

    df = pd.read_excel(os.path.join(data_dir, "Text labels.xlsx"))
    if "merge" in kwargs:
        for key, val in kwargs["merge"].items():
            for new_lbls in val:
                filtered_df.loc[filtered_df['Disease'].isin([new_lbls]), 'Disease'] = key

    img_paths = []
    for c in classes:
        temp_path = []
        disease_ids = df[df["Disease"] == c[0]]["ID"].sort_values().tolist()
        train, val, test = split_oct500(disease_ids, split)
        if mode == "train":
            temp_path += get_oct500(train, data_dir, class_label=c[1], filter_img=kwargs["filter_img"])
        elif mode == "val":
            temp_path += get_oct500(val, data_dir, class_label=c[1], filter_img=kwargs["filter_img"])
        elif mode == "test":
            temp_path += get_oct500(test, data_dir, class_label=c[1], filter_img=kwargs["filter_img"])
        img_paths += temp_path
    return img_paths


def split_oct500(total_ids: list, train_val_test: tuple):
    """
    Divides config into train, val, test
    :param total_ids: list of patients ids
    :param train_val_test: (train split, val split, test split) --> the sum should be 1
    """
    train_idx = math.ceil(len(total_ids) * train_val_test[0])
    return total_ids[0: train_idx], \
        total_ids[train_idx + 1: math.ceil(len(total_ids) * train_val_test[1]) + train_idx], \
        total_ids[math.ceil(len(total_ids) * train_val_test[1]) + train_idx + 1:]


def get_oct500(list_ids, data_dir, class_label, filter_img=True):
    img_paths = []
    for idd in list_ids:
        file_path = os.path.join(data_dir, "OCT", str(idd))
        for img in os.listdir(file_path):
            if filter_img and (("6mm" in data_dir and 160 <= int(img[:-4]) <= 240) or
                               ("3mm" in data_dir and 100 <= int(img[:-4]) <= 180)):
                img_paths.append((os.path.join(file_path, img), class_label))
            elif filter_img is False:
                img_paths.append((os.path.join(file_path, img), class_label))
        # img_paths += [(os.path.join(file_path, img), class_label)
        #               for img in os.listdir(file_path)]
    return img_paths


if __name__ == "__main__":
    # read from the .env file
    load_dotenv(dotenv_path="config/.env")
    DATASET_PATH = os.getenv('DATASET_PATH')
    OCTDataset(data_dir=DATASET_PATH + "2/OCTA_6mm/", dataset_func=get_oct500_imgs,
               mode="train", train_val_test=(0.6, 0.2, 0.2))
