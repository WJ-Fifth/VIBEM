# -*- coding: utf-8 -*-
# By Jinwu Wang u7354172

from lib.dataset import Dataset3D
from lib.core.config import SSP3D_DIR

class ssp3d(Dataset3D):
    def __init__(self, set, seqlen, overlap=0.75, debug=False):
        db_name = 'ssp3d'

        is_train = False
        overlap = overlap if is_train else 0.
        print('SSP_3D Dataset overlap ratio: ', overlap)
        super(ssp3d, self).__init__(
            set=set,
            folder=SSP3D_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')