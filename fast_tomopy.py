#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 08:44:11 2015

@author: lbluque
"""

import sys
import os
import argparse
import numpy as np
import tomopy
import h5py
import logging
from datetime import datetime


logger = logging.getLogger('fast_tomopy')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


def write_als_832h5(rec, input_file_name, file_data, group_data, output_file, step=1):

    logger = logging.getLogger('fast_tomopy.write_als_832h5')

    dx, dy, dz = rec.shape

    # This should really come from the group in the input H5 file and not the name of the H5 file.
    group_name = os.path.split(input_file_name)[-1].split('.')[0]

    dataset_data = {'date': '', 'dim1': 1, 'dim2': dy, 'dim3': dz, 'name':
                    'sample0', 'title': 'image'}
    data = np.empty((1, dy, dz), dtype='float32')

    with h5py.File(output_file, 'w') as f:
        # Write file level metadata
        logger.info('Writing output file level metadata')
        for key in file_data.keys():
            f.attrs.create(key, file_data[key])

        # Create group and write group level metadata
        logger.info('Writing output group level metadata')
        f.create_group(group_name)
        g = f[group_name]
        for key in group_data.keys():
            g.attrs.create(key, group_data[key])

        # Create and write reconstructed slices to group
        logger.info('Writing reconstructed slices')
        for i in range(dx):
            ind = step*(i + 1)
            dname = group_name + '_0000_{0:0={1}d}'.format(ind, 4) + '.tif'
            data[0] = rec[i, :, :]
            g.create_dataset(dname, data=data)
            for key in dataset_data.keys():
                g[dname].attrs.create(key, dataset_data[key])

        time = datetime.now()
        f.attrs['stage_date'] = time.strftime('%Y-%m-%dT%H%MZ')

    return


def fast_tomo_recon(argv):
    """
    Reconstruct subset slices (sinograms) equally spaced within tomographic
    dataset
    """

    logger = logging.getLogger('fast_tomopy.fast_tomo_recon')

    # Parse arguments passed to function
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='path to input raw '
                        'dataset', required=True)
    parser.add_argument('-o', '--output-file', type=str, help='full path to h5 output '
                        'file', default=os.path.join(os.getcwd(), "fast-tomopy.h5"))
    parser.add_argument('-sn', '--sino-num', type=int, help='Number of slices '
                        'to reconstruct', default=5)
    parser.add_argument('-a', '--algorithm', type=str, help='Reconstruction'
                        ' algorithm', default='gridrec',
                        choices=['art', 'bart', 'fbp', 'gridrec', 'mlem',
                                 'ospml_hybrid', 'ospml_quad', 'pml_hybrid',
                                 'pml_quad', 'sirt'])
    parser.add_argument('-c', '--center', type=float, help='Center of rotation',
                        default=None)
    parser.add_argument('-fn', '--filter-name', type=str, help='Name of filter'
                        ' used for reconstruction',
                        choices=['none', 'shepp', 'cosine', 'hann', 'hamming',
                                 'ramlak', 'parzen', 'butterworth'],
                        default='butterworth')
    parser.add_argument('-rr', '--ring-remove', type=str, help='Ring removal '
                        'method', choices=['Octopus', 'Tomopy-FW', 'Tomopy-T'],
                        default='Tomopy-FW')
    parser.add_argument('-lf', '--log-file', type=str, help='log file name',
                        default='fast-tomopy.log')

    args = parser.parse_args()

    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if os.path.isdir(os.path.dirname(args.output_file)) is False:
        raise IOError(2, 'Directory of output file does not exist', args.output_file)

    # Read file metadata
    logger.info('Reading input file metadata')
    fdata, gdata = tomopy.read_als_832h5_metadata(args.input)
    proj_total = int(gdata['nangles'])
    last = proj_total - 1
    sino_total = int(gdata['nslices'])
    ray_total = int(gdata['nrays'])
    px_size = float(gdata['pxsize'])/10  # cm

    # Set parameters for sinograms to read
    step = sino_total // (args.sino_num + 2)
    start = step
    end = step*(args.sino_num + 1)
    sino = (start, end, step)

    # Read full first and last projection to determine center of rotation
    if args.center is None:
        logger.info('Reading full first and last projection for COR')
        first_last, flats, darks, floc = tomopy.read_als_832h5(args.input,
                                                               ind_tomo=(0, last))
        first_last = tomopy.normalize_nf(first_last, flats, darks, floc)
        args.center = tomopy.find_center_pc(first_last[0, :, :],
                                            first_last[1, :, :], tol=0.1)
        logger.info('Detected center: %f', args.center)

    # Read and normalize raw sinograms
    logger.info('Reading raw data')
    tomo, flats, darks, floc = tomopy.read_als_832h5(args.input, sino=sino)
    logger.info('Normalizing raw data')
    tomo = tomopy.normalize_nf(tomo, flats, darks, floc)

    # Remove stripes from sinograms (remove rings)
    logger.info('Preprocessing normalized data')
    if args.ring_remove == 'Tomopy-FW':
        logger.info('Removing stripes from sinograms with %s',
                    args.ring_remove)
        tomo = tomopy.remove_stripe_fw(tomo)
    elif args.ring_remove == 'Tomopy-T':
        logger.info('Removing stripes from sinograms with %s',
                    args.ring_remove)
        tomo = tomopy.remove_stripe_ti(tomo)

    # Pad sinograms with edge values
    npad = int(np.ceil(ray_total*np.sqrt(2)) - ray_total)//2
    tomo = tomopy.pad(tomo, 2, npad=npad, mode='edge')
    args.center += npad # account for padding

    filter_name = np.array(args.filter_name, dtype=(str, 16))
    theta = tomopy.angles(proj_total, 270, 90)

    logger.info('Reconstructing normalized data')
    # Reconstruct sinograms
    rec = tomopy.recon(tomo, theta, center=args.center, emission=False,
                       algorithm=args.algorithm, filter_name=filter_name)
    rec = tomopy.circ_mask(rec[:, npad:-npad, npad:-npad], 0)
    rec = rec/px_size

    # Remove rings from reconstruction
    if args.ring_remove == 'Octopus':
        logger.info('Removing rings from reconstructions with %s',
                    args.ring_remove)
        thresh = float(gdata['ring_threshold'])
        thresh_max = float(gdata['upp_ring_value'])
        thresh_min = float(gdata['low_ring_value'])
        theta_min = int(gdata['max_arc_length'])
        rwidth = int(gdata['max_ring_size'])
        rec = tomopy.remove_rings(rec, center_x=args.center, thresh=thresh,
                                  thresh_max=thresh_max, thresh_min=thresh_min,
                                  theta_min=theta_min, rwidth=rwidth)

    # Write reconstruction data to new hdf5 file
    fdata['stage'] = 'fast-tomopy'
    fdata['stage_flow'] = '/raw/' + fdata['stage']
    fdata['stage_version'] = 'fast-tomopy-0.1'
    # WHAT ABOUT uuid ????? Who asigns this???
    del fdata['uuid']  # I'll get rid of it altogether then...

    gdata['Reconstruction_Type'] = 'tomopy-gridrec'
    gdata['ring_removal_method'] = args.ring_remove
    gdata['rfilter'] = args.filter_name

    logger.info('Writing reconstructed data to h5 file')
    write_als_832h5(rec, args.input, fdata, gdata, args.output_file, step)

    return

if __name__ == '__main__':
    fast_tomo_recon(sys.argv[1:])
