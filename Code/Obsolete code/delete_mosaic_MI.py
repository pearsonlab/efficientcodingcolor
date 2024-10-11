# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:57:36 2024

@author: David
"""

from MosaicAnalysis import Analysis
import numpy as np
save = '240301-055438_test2'
path = "../../saves/" + save + "/" 

test = Analysis(path)
test(3,5,4)

#restriction = 'torch.mean((result[0,:,:] - result[1,:,:])**2) < 0.2'
restriction = 'torch.mean(result) < 0'
mosaics_to_delete = [1,2]

loss1,MI1,r1 = test.compute_loss(restriction = restriction)
for mosaic in mosaics_to_delete:
    test.delete_mosaic(mosaic)
loss2,MI2,r2 = test.compute_loss(restriction = restriction, skip_read = True)
loss3, MI3, r3 = test.compute_loss(restriction = 'True', skip_read = False)
print("Mosaic that were deleted: ", str(mosaics_to_delete))
print(-76.67, np.mean(loss3), np.mean(loss1), np.mean(loss2))
print(-76.67, np.mean(MI3), np.mean(MI1), np.mean(MI2))


#Notes: If I reverse d for last 2 mosaics, the loss gets way higher 
#because the firing rate constraint. This difference diminishes if I remove the high-var images 