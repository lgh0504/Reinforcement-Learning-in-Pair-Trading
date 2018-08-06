import os, zipfile
import pandas as pd
from typing import List


def target_zip(ccy:str, year:int, months:List[int]) -> List[str]:
    target_files = ['-'.join([ccy, year, str(month).zfill(2)]) + '.zip' for month in months]
    return target_files


def get_file_paths(folder_path:str, files:List[str]) -> List[str]:
    file_paths = []
    for file in files:
        file_path = folder_path + file
        assert os.path.isfile(file_path), 'Error:{file} not found in {folder}'.format(file=file, folder=folder_path)
        file_paths.append(file_path)
    return file_paths


def extract_fx_close(file_path:str, freq_in_min:int) -> pd.DataFrame:
    with zipfile.ZipFile(file_path) as zip:
        unzipped_file_name = zipfile.ZipFile.namelist(zip)[0]
        with zip.open(unzipped_file_name) as file:
            for line in file:
                print(line)