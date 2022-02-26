import numpy as np
import os                
import nibabel as nib    
import imageio           
 
def read_niifile(niifile):
    img = nib.load(niifile)
    img_fdata = img.get_fdata()
    img90 = np.rot90(img_fdata)

    return img90

def save_fig(file):
    fdata = read_niifile(file)
    (y,x,z) = fdata.shape
    for k in range(z):
        silce = fdata[:,:, k]
        imageio.imwrite(os.path.join(output, '{}.jpg'.format(k)),silce)

def findAllFile(base):
    for root, ds, fs, in os.walk(base):
        for f in fs:
            yield f

base = r'data1.nii.gz'
output = r'jpg'
for i in findAllFile(base):
    dir = os.path.join(base, i)
    savepicdir = (os.path.join(output,i))
    save_fig(dir)


save_fig(base)