# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

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