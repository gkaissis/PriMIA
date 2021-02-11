import os
from os import path, remove, environ
from preprocessing.data_loader import prepare_data
import numpy as np
from PIL import Image
from tqdm import tqdm

import argparse
import configparser

def save_to_img(PATH, array, isLabel:bool): 
    """Convert splitted and processed arrays to jpg"""
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        
    if isLabel: 
        for i, img_array in tqdm(enumerate(array), total=len(array)):
            formatted = (img_array * 255)
            #img_array[img_array==2]=255
            #img_array[img_array==1]=0
            #formatted = img_array
            img = Image.fromarray(formatted)
            img.save(PATH+f"{i}.jpg")
    else: 
        for i, img_array in tqdm(enumerate(array), total=len(array)):
            formatted = (img_array * 255).astype('uint8')
            img = Image.fromarray(formatted)
            img.save(PATH+f"{i}.jpg")

#class Arguments:
#    """Parameters for training"""
#    def __init__(self, cmd_args):
#        self.data = cmd_args.data
#        self.res = cmd_args.res
#        self.res_z = cmd_args.res_z
#        self.crop_height = cmd_args.crop_height

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    RES = 256
    RES_Z = 64
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the parent directory of the input and label files.",
    )
    parser.add_argument(
        "--res",
        type=int,
        required=True,
        help="Resolution of the scans.",
    )
    parser.add_argument(
        "--res_z",
        type=int,
        required=True,
        help="Number of stacked scans, z-axis in scans.",
    )
    parser.add_argument(
        "--crop_height",
        type=int,
        required=True,
        help="Finds the midpoint of the labels along the z-axis and crops [..., zmiddle-(crop_height//2):zmiddle+(crop_height//2)].",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        help="In case the number of samples should be set to a specific value",
    )

    cmd_args = parser.parse_args()
    #args = Arguments(cmd_args)
   
    print(str(cmd_args))
    print("### start process ###")

    PATH = cmd_args.data
    RES = cmd_args.res
    RES_Z = cmd_args.res_z
    CROP_HEIGHT = cmd_args.crop_height
    # 281 is the whole datset (for liver)
    NUM_SAMPLES = cmd_args.num_samples if cmd_args.num_samples is not None else 281

    # Preprocessing
    print("## preprocessing ##")
    X_train_partial, y_train_partial, X_val, y_val, X_test, y_test = prepare_data(
                                                                                    PATH,
                                                                                    res=RES,
                                                                                    res_z=RES_Z,
                                                                                    num_samples=NUM_SAMPLES
                                                                                )
    print("## converting and saving ##")
    save_to_img(PATH+'/train/inputs/', X_train_partial, False)
    save_to_img(PATH+'/val/inputs/', X_val, False)
    save_to_img(PATH+'/test/inputs/', X_test, False)

    save_to_img(PATH+'/train/labels/', y_train_partial, True)
    save_to_img(PATH+'/val/labels/', y_val, True)
    save_to_img(PATH+'/test/labels/', y_test, True)

