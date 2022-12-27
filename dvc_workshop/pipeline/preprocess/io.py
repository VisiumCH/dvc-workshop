import os
import shutil 
import glob
from typing import List, Iterable
import numpy.typing as npt



def read_images(source_directory: str) -> List[str] : 
    """simple function to showcase reading data from io 

    Args:
        source_directory (str): source iamges directory

    Returns:
        List[str]: list of images 
    """
    return glob.glob(source_directory + "/**/*.jpg",recursive=True)

def save_images(images:Iterable[str],target_directory:str) :
    """simple function to showcase writing data on disk from io 

    Args:
        images (Iterable[str]): Image paths
        target_directory (str): Destination directory
    """
    # check if directory exists
    target_exist = os.path.exists(target_directory)
    if not target_exist:
        # Create a new directory because it does not exist
        os.makedirs(target_directory)
    # copy selected images
    for img in images :
        shutil.copy(img,target_directory)


