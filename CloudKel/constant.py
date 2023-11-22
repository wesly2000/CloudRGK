# This file defines some frequently used constants.

import numpy as np

__ELEMENT_TYPE__ = np.int64 # Currently we use np.int64 for feature matrix and encryption.
__INDEX_TYPE__ = np.int32 # Used for encrypting indices, since indices are re
                          # reletively small(Seldom smaller than 65565), one can 
                          # set this constant to np.int8 or np.int16 for storage
                          # concern.

recover_steps = {np.int16: 2,np.int32: 4, np.int64: 8}
__RECOVER_STEP__ = recover_steps[__INDEX_TYPE__] # Used for recovering byte array
                                                 # to tuple list.