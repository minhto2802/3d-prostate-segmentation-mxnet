import os
import SimpleITK as sitk
import numpy as np
import pylab as plt
from skimage.util import montage


def extract_itkimage(_itkimage):
    """Return numpy arrays of image, origin and spacing"""
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    scan = sitk.GetArrayFromImage(_itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(_itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(_itkimage.GetSpacing())))

    return scan, origin, spacing


def load_itk(_filename, itkimage_only=False):
    """This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image"""
    _itkimage = sitk.ReadImage(_filename)
    if itkimage_only:
        return _itkimage

    scan, origin, spacing = extract_itkimage(_itkimage)
    return scan, origin, spacing


def get_reference_image(input_folder, case_idx=38):
    """Get reference image"""
    return load_itk('%s/Case%02d.mhd' % (input_folder, case_idx), itkimage_only=True)


def print_metadata(x):
    """Print meta-data"""
    print('%02d ' % i,
          x[0].shape[0], x[0].shape[1], x[0].shape[2],
          x[1][0], x[1][1], x[1][2],
          x[2][0], x[2][1], x[2][2],
          )


def load_and_resample_itk(_filename, _ref_img, is_label=False, force_size=False):
    """Load and resample itkimage to _ref_img"""
    ori_itkimage = load_itk(_filename, itkimage_only=True) if isinstance(_filename, str) else _filename

    # Compute size of resampled image
    ref_spacing = np.array(_ref_img.GetSpacing())
    orig_size = np.array(ori_itkimage.GetSize(), dtype=np.int)
    orig_spacing = np.array(ori_itkimage.GetSpacing())
    if not force_size:
        new_size = orig_size * (orig_spacing / ref_spacing)
        new_size = list(np.round(new_size).astype(np.int))  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]
    else:
        new_size = _ref_img.GetSize()
    # Setup resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    # We need both direction and origin to correctly resample the original image
    resampler.SetOutputDirection(ori_itkimage.GetDirection())  # direction from original image
    resampler.SetOutputOrigin(ori_itkimage.GetOrigin())  # origin from original image
    resampler.SetOutputSpacing(ref_spacing)
    resampler.SetSize(new_size)

    # Perform Resampling
    resampled_itkimage = resampler.Execute(ori_itkimage)
    return resampled_itkimage


def montage_volume(_x, _l=None, save_to_file=False, _set='train', case_id=0):
    """Visualize image and contoured label (if available)"""
    plt.figure(1, (12, 12), frameon=False)
    plt.imshow(montage(_x), cmap='gray')
    if _l is not None:
        plt.contour(montage(_l), cmap='gray')
    plt.axis('off')

    if save_to_file:
        _folder = 'inputs/figures/resampled/%s' % _set
        _filename = '%s/Case%02d.png' % (_folder, case_id)
        os.makedirs(_folder) if not os.path.exists(_folder) else None

        plt.savefig(_filename, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()


def inspect_reversed_resampled():
    """This function use variables in the main namespace"""
    ori = load_itk(filename, itkimage_only=True)
    ori_rev = load_and_resample_itk(resampled_x, ori, force_size=False, is_label=False)  # reversed resampled image
    ori, ori_rev = extract_itkimage(ori)[0], extract_itkimage(ori_rev)[0]
    if not (set == 'test'):
        ori1 = load_itk(filename_label, itkimage_only=True)
        ori_rev1 = load_and_resample_itk(resampled_l, ori1, force_size=False, is_label=True)  # reversed resampled label
        ori1, ori_rev1 = extract_itkimage(ori1)[0], extract_itkimage(ori_rev1)[0]
    else:
        ori1, ori_rev1 = None, None

    plt.figure(1, (20, 10), frameon=True)
    plt.subplot(121), montage_volume(ori, ori1)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(122), montage_volume(ori_rev, ori_rev1)
    plt.title('Reversed Resampled')
    plt.axis('off')
    _folder = 'inputs/figures/inspect_reversed_resampled/%s' % set
    _filename = '%s/Case%02d.png' % (_folder, i)
    os.makedirs(_folder) if not os.path.exists(_folder) else None
    plt.savefig(_filename, bbox_inches='tight', transparent=False, pad_inches=0)
    plt.close()


if __name__ == "__main__":
    sets = {
        'training': 50,
        # 'test': 30,
    }
    to_inspect_reversed_resampled = False
    to_save_resampled_fig = False
    verbose = False
    ref_idx = 12  # None for not resampling
    ref_img = get_reference_image(input_folder='inputs/raw/training', case_idx=ref_idx) if ref_idx else None
    viz_resampled = True

    for set, n in sets.items():
        for i in range(n):
            input_folder = 'inputs/raw/%s' % set
            output_folder = 'inputs/resampled/%s' % set
            filename = '%s/Case%02d.mhd' % (input_folder, i)
            filename_label = '%s/Case%02d_segmentation.mhd' % (input_folder, i)
            print(filename)

            if verbose:
                print_metadata(load_itk(filename))

            if ref_idx and (
                    (i != ref_idx) or (set == 'test')):  # This condition is safe for PROMISE2012 for n_test <= 30
                # Resample the image to the reference image
                resampled_x = load_and_resample_itk(filename, ref_img)
                if set == 'training':
                    resampled_l = load_and_resample_itk(filename_label, ref_img, is_label=True)
                    if to_inspect_reversed_resampled:
                        inspect_reversed_resampled()
                        continue
                    resampled_x, resampled_l = extract_itkimage(resampled_x)[0], extract_itkimage(resampled_l)[0]
                    img = np.concatenate((resampled_x[np.newaxis], resampled_l[np.newaxis]))
                else:
                    resampled_l = None
                    if to_inspect_reversed_resampled:
                        inspect_reversed_resampled()
                        continue
                    resampled_x = extract_itkimage(resampled_x)[0]
                    img = resampled_x[np.newaxis]
            else:
                resampled_x = load_itk(filename)[0]  # resampled image as reference image
                resampled_l = load_itk(filename_label)[0]  # resampled image as reference image
            if to_save_resampled_fig:
                montage_volume(resampled_x, resampled_l, save_to_file=True,
                               _set=set, case_id=i) if viz_resampled else None

            os.makedirs(output_folder) if not os.path.exists(output_folder) else None
            np.save('%s/Case%02d.npy' % (output_folder, i), img)
