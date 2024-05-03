#!/usr/bin/python
import io
import multiprocessing
import os
import shutil
import sys
import tempfile
from cmath import sqrt
from io import BytesIO
# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from pathlib import Path

import click
import exifread
import magic
import numpy as np
import scipy.stats
from PIL import Image, ImageCms
from PIL.ExifTags import TAGS
from imageio.v3 import imread, imiter as imsave
from scipy import ndimage

from aletheialib.exiftool import ExifTool
from aletheialib.jpeg import JPEG

exifread.logger.disabled = True


# -- EXIF --

# {{{ exif()


def exif(filename) -> dict:
    """Return the EXIF data of an image."""
    with ExifTool() as exif:
        return exif.get_metadata(filename)


# }}}


def metadata_diff(cover, stego) -> dict:
    """Return the differences between the metadata of two images."""
    cover_data = exif(cover)
    stego_data = exif(stego)
    diff = {}
    for key in cover_data.keys() | stego_data.keys():
        if cover_data.get(key) != stego_data.get(key):
            diff[key] = (cover_data.get(key), stego_data.get(key))

    return diff


# -- SAMPLE PAIR ATTACK --

# {{{ spa()
"""
    Sample Pair Analysis attack. 
    Return Beta, the detected embedding rate.
"""


def spa(filename, channel=0):
    return spa_image(imread(filename), channel)


def spa_image(image, channel=0):
    if channel != None:
        width, height, channels = image.shape
        I = image[:, :, channel]
    else:
        I = image
        width, height = I.shape

    r = I[:-1, :]
    s = I[1:, :]

    # we only care about the lsb of the next pixel
    lsb_is_zero = np.equal(np.bitwise_and(s, 1), 0)
    lsb_non_zero = np.bitwise_and(s, 1)
    msb = np.bitwise_and(I, 0xFE)

    r_less_than_s = np.less(r, s)
    r_greater_than_s = np.greater(r, s)

    x = np.sum(np.logical_or(np.logical_and(lsb_is_zero, r_less_than_s),
                             np.logical_and(lsb_non_zero, r_greater_than_s)).astype(int))

    y = np.sum(np.logical_or(np.logical_and(lsb_is_zero, r_greater_than_s),
                             np.logical_and(lsb_non_zero, r_less_than_s)).astype(int))

    k = np.sum(np.equal(msb[:-1, :], msb[1:, :]).astype(int))

    if k == 0:
        click.echo("ERROR")
        sys.exit(0)

    a = 2 * k
    b = 2 * (2 * x - width * (height - 1))
    c = y - x

    bp = (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    bm = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    beta = min(bp.real, bm.real)
    return beta


# }}}


# -- RS ATTACK --

# {{{ solve()
def solve(a, b, c):
    sq = np.sqrt(b ** 2 - 4 * a * c)
    return (-b + sq) / (2 * a), (-b - sq) / (2 * a)


# }}}

# {{{ smoothness()
def smoothness(I):
    return (np.sum(np.abs(I[:-1, :] - I[1:, :])) +
            np.sum(np.abs(I[:, :-1] - I[:, 1:])))


# }}}

# {{{ groups()
def groups(I, mask):
    grp = []
    m, n = I.shape
    x, y = np.abs(mask).shape
    for i in range(m - x):
        for j in range(n - y):
            yield I[i:(i + x), j:(j + y)]


# }}}

class RSAnalysis(object):
    def __init__(self, mask):
        self.mask = mask
        self.cmask = - mask
        self.cmask[(mask > 0)] = 0
        self.abs_mask = np.abs(mask)

    def call(self, group):
        flip = (group + self.cmask) ^ self.abs_mask - self.cmask
        return np.sign(smoothness(flip) - smoothness(group))


# {{{ difference()
def difference(I, mask):
    pool = Pool(multiprocessing.cpu_count())
    analysis = pool.map(RSAnalysis(mask).call, groups(I, mask))
    pool.close()
    pool.join()

    counts = [0, 0, 0]
    for v in analysis:
        counts[v] += 1

    N = sum(counts)
    R = float(counts[1]) / N
    S = float(counts[-1]) / N
    return R - S


# }}}

# {{{ rs()
def rs(filename, channel=0):
    return rs_image(np.asarray(imread(filename), channel))


def rs_image(image, channel=0):
    if channel != None:
        I = image[:, :, channel]
    else:
        I = image
    I = I.astype(int)

    mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    d0 = difference(I, mask)
    d1 = difference(I ^ 1, mask)

    mask = -mask
    n_d0 = difference(I, mask)
    n_d1 = difference(I ^ 1, mask)

    p0, p1 = solve(2 * (d1 + d0), (n_d0 - n_d1 - d1 - 3 * d0), (d0 - n_d0))
    if np.abs(p0) < np.abs(p1):
        z = p0
    else:
        z = p1

    return z / (z - 0.5)


# }}}

# -- CALIBRATION --

# {{{ calibration()
def H_i(dct, k, l, i):
    dct_kl = dct[k::8, l::8].flatten()
    return sum(np.abs(dct_kl) == i)


def H_i_all(dct, i):
    dct_kl = dct.flatten()
    return sum(np.abs(dct_kl) == i)


def beta_kl(dct_0, dct_b, k, l):
    h00 = H_i(dct_0, k, l, 0)
    h01 = H_i(dct_0, k, l, 1)
    h02 = H_i(dct_0, k, l, 2)
    hb0 = H_i(dct_b, k, l, 0)
    hb1 = H_i(dct_b, k, l, 1)
    return (h01 * (hb0 - h00) + (hb1 - h01) * (h02 - h01)) / (h01 ** 2 + (h02 - h01) ** 2)


def calibration_f5(path):
    """ it uses jpeg_toolbox """
    import jpeg_toolbox as jt

    tmpdir = tempfile.mkdtemp()
    predfile = os.path.join(tmpdir, 'img.jpg')
    os.system("convert -chop 4x4 " + path + " " + predfile)
    im_jpeg = jt.load(path)
    impred_jpeg = jt.load(predfile)
    shutil.rmtree(tmpdir)

    beta_list = []
    for i in range(im_jpeg["jpeg_components"]):
        dct_b = im_jpeg["coef_arrays"][i]
        dct_0 = impred_jpeg["coef_arrays"][i]
        b01 = beta_kl(dct_0, dct_b, 0, 1)
        b10 = beta_kl(dct_0, dct_b, 1, 0)
        b11 = beta_kl(dct_0, dct_b, 1, 1)
        beta = (b01 + b10 + b11) / 3
        if beta > 0.05:
            click.echo(f"Hidden data found in channel {i}: {beta}")
        else:
            click.echo(f"No hidden data found in channel {i}")


def calibration_chisquare_mode(path):
    """ it uses jpeg_toolbox """
    import jpeg_toolbox as jt

    tmpdir = tempfile.mkdtemp()
    predfile = os.path.join(tmpdir, 'img.jpg')
    os.system("convert -chop 4x4 " + path + " " + predfile)
    im_jpeg = jt.load(path)
    impred_jpeg = jt.load(predfile)
    shutil.rmtree(tmpdir)

    beta_list = []
    for i in range(im_jpeg["jpeg_components"]):
        dct = im_jpeg["coef_arrays"][i]
        dct_estim = impred_jpeg["coef_arrays"][i]

        p_list = []
        for k in range(4):
            for l in range(4):
                if (k, l) == (0, 0):
                    continue

                f_exp, f_obs = [], []
                for j in range(5):
                    h = H_i(dct, k, l, j)
                    h_estim = H_i(dct_estim, k, l, j)
                    if h < 5 or h_estim < 5:
                        break
                    f_exp.append(h_estim)
                    f_obs.append(h)
                # click.echo(f_exp, f_obs)

                chi, p = scipy.stats.chisquare(f_obs, f_exp)
                p_list.append(p)

        p = np.mean(p_list)
        if p < 0.05:
            click.echo(f"Hidden data found in channel {i}: {p}")
        else:
            click.echo(f"No hidden data found in channel {i}")


def calibration_f5_octave_jpeg(filename, return_result=False):
    """ It uses JPEG from octave """
    tmpdir = tempfile.mkdtemp()
    predfile = os.path.join(tmpdir, 'img.jpg')
    os.system("convert -chop 4x4 " + filename + " " + predfile)

    im_jpeg = JPEG(filename)
    impred_jpeg = JPEG(predfile)
    beta_avg = 0.0
    for i in range(im_jpeg.components()):
        dct_b = im_jpeg.coeffs(i)
        dct_0 = impred_jpeg.coeffs(i)
        b01 = beta_kl(dct_0, dct_b, 0, 1)
        b10 = beta_kl(dct_0, dct_b, 1, 0)
        b11 = beta_kl(dct_0, dct_b, 1, 1)
        beta = (b01 + b10 + b11) / 3
        if beta > 0.05:
            beta_avg += beta
            if not return_result:
                click.echo(f"Hidden data found in channel {i}: {beta}")
        else:
            if not return_result:
                click.echo(f"No hidden data found in channel {i}")
    beta_avg /= im_jpeg.components()
    if return_result:
        return beta_avg

    shutil.rmtree(tmpdir)


# }}}


# -- NAIVE ATTACKS

# {{{ high_pass_filter()
def high_pass_filter(input_image, output_image):
    I = imread(input_image)
    if len(I.shape) == 3:
        kernel = np.array([[[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]],
                           [[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]],
                           [[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]]])
    else:
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])

    If = ndimage.convolve(I, kernel)
    imsave(output_image, If)


# }}}

# {{{ low_pass_filter()
def low_pass_filter(input_image, output_image):
    I = imread(input_image)
    if len(I.shape) == 3:
        kernel = np.array([[[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]])
    else:
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])

    kernel = kernel.astype('float32') / 9
    If = ndimage.convolve(I, kernel)
    imsave(output_image, If)


# }}}

# {{{ imgdiff()
def imgdiff(image1, image2):
    I1 = imread(image1).astype('int16')
    I2 = imread(image2).astype('int16')
    np.set_printoptions(threshold=sys.maxsize)

    if len(I1.shape) != len(I2.shape):
        click.echo("Error, both images must have the same dimensionality")
        sys.exit(0)

    if len(I1.shape) == 2:
        D = I1 - I2
        click.echo(D)

    elif len(I1.shape) == 3:
        D1 = I1[:, :, 0] - I2[:, :, 0]
        D2 = I1[:, :, 1] - I2[:, :, 1]
        D3 = I1[:, :, 2] - I2[:, :, 2]
        click.echo("Channel 1:")
        click.echo(D1)
        click.echo("Channel 2:")
        click.echo(D2)
        click.echo("Channel 3:")
        click.echo(D3)
    else:
        click.echo(f"Error, too many dimensions: {I1.shape}")


# }}}

# {{{ imgdiff_pixels()
def imgdiff_pixels(image1, image2):
    I1 = imread(image1).astype('int16')
    I2 = imread(image2).astype('int16')
    np.set_printoptions(threshold=sys.maxsize)

    if len(I1.shape) != len(I2.shape):
        click.echo("Error, both images must have the same dimensionality")
        sys.exit(0)

    if len(I1.shape) == 2:
        D = I1 - I2
        pairs = list(zip(I1.ravel(), D.ravel()))
        click.echo(np.array(pairs, dtype=('i4,i4')).reshape(I1.shape))


    elif len(I1.shape) == 3:
        D1 = I1[:, :, 0] - I2[:, :, 0]
        D2 = I1[:, :, 1] - I2[:, :, 1]
        D3 = I1[:, :, 2] - I2[:, :, 2]
        click.echo("Channel 1:")
        pairs = list(zip(I1[:, :, 0].ravel(), D1.ravel()))
        pairs_diff = [p for p in pairs if p[1] != 0]
        # click.echo(np.array(pairs, dtype=('i4,i4')).reshape(I1[:,:,0].shape))
        click.echo(pairs_diff)
        click.echo("Channel 2:")
        pairs = list(zip(I1[:, :, 1].ravel(), D2.ravel()))
        pairs_diff = [p for p in pairs if p[1] != 0]
        # click.echo(np.array(pairs, dtype=('i4,i4')).reshape(I1[:,:,1].shape))
        click.echo(pairs_diff)
        click.echo("Channel 3:")
        pairs = list(zip(I1[:, :, 2].ravel(), D3.ravel()))
        pairs_diff = [p for p in pairs if p[1] != 0]
        # click.echo(np.array(pairs, dtype=('i4,i4')).reshape(I1[:,:,2].shape))
        click.echo(pairs_diff)
    else:
        click.echo(f"Error, too many dimensions: {I1.shape}")


# }}}

# {{{ print_diffs()
def print_diffs(cover, stego):
    def print_list(l, ln):
        for i in range(0, len(l), ln):
            click.echo(l[i:i + ln])

    C = imread(cover, pilmode='RGB').astype('int16')
    S = imread(stego, pilmode='RGB').astype('int16')
    np.set_printoptions(threshold=sys.maxsize)

    if len(C.shape) != len(S.shape):
        click.echo("Error, both images must have the same dimensionality")
        sys.exit(0)

    if len(C.shape) == 2:
        D = S - C
        pairs = list(zip(C.ravel(), S.ravel(), D.ravel()))
        pairs_diff = [p for p in pairs if p[2] != 0]
        print_list(pairs_diff, 5)


    elif len(C.shape) == 3:
        D1 = S[:, :, 0] - C[:, :, 0]
        D2 = S[:, :, 1] - C[:, :, 1]
        D3 = S[:, :, 2] - C[:, :, 2]
        click.echo("\nChannel 1:")
        pairs = list(zip(C[:, :, 0].ravel(), S[:, :, 0].ravel(), D1.ravel()))
        pairs_diff = [p for p in pairs if p[2] != 0]
        print_list(pairs_diff, 5)
        click.echo("\nChannel 2:")
        pairs = list(zip(C[:, :, 1].ravel(), S[:, :, 1].ravel(), D2.ravel()))
        pairs_diff = [p for p in pairs if p[2] != 0]
        print_list(pairs_diff, 5)
        click.echo("\nChannel 3:")
        pairs = list(zip(C[:, :, 2].ravel(), S[:, :, 2].ravel(), D3.ravel()))
        pairs_diff = [p for p in pairs if p[2] != 0]
        print_list(pairs_diff, 5)
    else:
        click.echo(f"Error, too many dimensions: {C.shape}")


# }}}

# {{{ print_dct_diffs()
def print_dct_diffs(cover, stego):
    # import jpeg_toolbox as jt

    def print_list(l, ln):
        mooc = 0
        for i in range(0, len(l), ln):
            click.echo(l[i:i + ln])
            v = l[i:i + ln][0][2]
            if np.abs(v) > 1:
                mooc += 1

    C_jpeg = JPEG(cover)
    # C_jpeg = jt.load(cover)
    S_jpeg = JPEG(stego)
    # S_jpeg = jt.load(stego)
    for i in range(C_jpeg.components()):
        # for i in range(C_jpeg["jpeg_components"]):
        C = C_jpeg.coeffs(i)
        S = S_jpeg.coeffs(i)
        # C = C_jpeg["coef_arrays"][i]
        # S = S_jpeg["coef_arrays"][i]
        if C.shape != S.shape:
            click.echo(f"WARNING! channels with different size. Channel: {i}")
            continue
        D = S - C
        click.echo(f"\nChannel {i}:")
        pairs = list(zip(C.ravel(), S.ravel(), D.ravel()))
        pairs_diff = [p for p in pairs if p[2] != 0]
        print_list(pairs_diff, 5)

    click.echo("\nCommon DCT coefficients frequency variation:")
    for i in range(C_jpeg.components()):
        click.echo(f"\nChannel {i}:")
        nz_coeffs = np.count_nonzero(C_jpeg.coeffs(i))
        changes = np.sum(np.abs(C_jpeg.coeffs(i) - S_jpeg.coeffs(i)))
        rate = round(changes / nz_coeffs, 4)
        click.echo(f"non zero coeffs: {nz_coeffs}, changes: {changes}, rate: {rate}")
        click.echo("H BAR    COVER      STEGO       DIFF")
        click.echo("------------------------------------")
        for v in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            cover = np.sum(C_jpeg.coeffs(i) == v)
            stego = np.sum(S_jpeg.coeffs(i) == v)
            var = stego - cover
            click.echo(f"{v:+}: {cover:10d} {stego:10d} {var:10d}")


# }}}

# {{{ remove_alpha_channel()
def remove_alpha_channel(input_image, output_image):
    I = imread(input_image)
    I[:, :, 3] = 255
    imsave(output_image, I)


# }}}

# {{{ eof_extract()
def eof_extract(input_image, output_data):
    name, ext = os.path.splitext(input_image)

    eof = None
    if ext.lower() in [".jpeg", ".jpg"]:
        eof = b"\xFF\xD9"
    elif ext.lower() in [".gif"]:
        eof = b"\x00\x3B"
    elif ext.lower() in [".png"]:
        eof = b"\x49\x45\x4E\x44\xAE\x42\x60\x82"
    else:
        click.echo("Please provide a JPG, GIF or PNG file")
        sys.exit(0)

    raw = open(input_image, 'rb').read()
    buff = BytesIO()
    buff.write(raw)
    buff.seek(0)
    bytesarray = buff.read()
    data = bytesarray.rsplit(eof, 1)  # last occurrence

    # data[0] contains the original image
    if len(data[1]) == 0:
        click.echo("No data found")
        sys.exit(0)
    with open(output_data, 'wb') as outf:
        outf.write(data[1])

    ft = magic.Magic(mime=True).from_file(output_data)
    click.echo(f"\nData extracted from {input_image} and saved in {output_data}. Filetype: {ft}")


# }}}

# {{{ lsb_extract()

def lsb_extract(input_image, bits=1, channels='RGB', endian='little', direction='row') -> np.ndarray:
    """Extract a message from an image using the LSBs method and returns it.

    :param input_image: the input image
    :param bits: the number of bits to extract
    :param channels: the channels to extract the message from. Options: 'R', 'G', 'B', 'A' or a combination of them. **Default**: 'RGB'
    :param endian: the endianness of the message. Options: 'little' or 'big'. **Default**: 'little'
    :param direction: the direction in which to traverse the image. Options: 'row' or 'col'. **Default**: 'row'
    :return: the extracted message
    """

    def _extract_bits_opt_little(data):
        div = 8 // bits
        message = np.zeros(len(data) // div, dtype=np.uint8)
        mask = (1 << bits) - 1
        for i in range(div):
            shift = bits * i
            message |= (data[i::div] & mask) << shift
        return message

    def _extract_bits_opt_big(data):
        div = 8 // bits
        message = np.zeros(len(data) // div, dtype=np.uint8)
        mask = (1 << bits) - 1
        for i in range(div):
            shift = 8 - bits - (bits * i)
            message |= (data[i::div] & mask) << shift
        return message

    def _extract_bits_little(data):
        msg_byte = 0
        shift = 0
        message = []
        mask = (1 << bits) - 1
        for byte in data:
            msg_byte |= (byte & mask) << shift
            shift += bits
            if shift >= 8:
                tmp = msg_byte >> 8
                message.append(msg_byte & 0xFF)
                msg_byte = tmp
                shift -= 8
        return np.array(message, dtype=np.uint8)

    def _extract_bits_big(data):
        msg_byte = 0
        shift = 8 - bits
        message = []
        mask = (1 << bits) - 1
        for byte in data:
            msg_byte |= (byte & mask) << shift
            shift += bits
            if shift <= 0:
                tmp = msg_byte >> 8
                message.append(msg_byte & 0xFF)
                msg_byte = tmp
                shift += 8
        return np.array(message, dtype=np.uint8)

    _COL_MAP = {'R': 0, 'G': 1, 'B': 2, 'A': 3}

    def _load_image(img_path: Path, convert_mode='RGB', channels=None, direction='row'):
        if 'A' in channels:
            convert_mode = 'RGBA'

        with Image.open(img_path) as img:
            arr = np.array(img.convert(convert_mode))

        if direction == 'col' or direction == 'column':
            arr = arr.transpose(1, 0, 2)

        channels = [*channels] if channels else None
        if (convert_mode == 'RGB' and 0 < len(channels) < 3) or (convert_mode == 'RGBA' and 0 < len(channels) < 4):
            arr = arr[:, :, [_COL_MAP[c] for c in channels]]
        return arr.reshape(-1)

    def _extract_message(img_path: Path, convert_mode='RGB'):
        data = _load_image(img_path, convert_mode, channels, direction)
        if bits == 1 or bits.bit_count() == 1:
            if endian == 'big':
                return _extract_bits_opt_big(data)
            else:
                return _extract_bits_opt_little(data)
        else:
            if endian == 'big':
                return _extract_bits_big(data)
            else:
                return _extract_bits_little(data)

    return _extract_message(input_image)


# }}}

# {{{ size_diff()

def size_diff(image_pairs, *, payload_length=None, verbose=False):
    """Calculate the size difference between cover and stego images.

    The image_pairs is a list of tuples with the cover and stego image paths.

    :param image_pairs: list of tuples with cover and stego image paths
    :param payload_length: payload size in bytes
    :param verbose: whether to print the differences
    :return: the average size difference
    """

    results = []
    for cover, stego in image_pairs:
        cover_size = os.path.getsize(cover)
        stego_size = os.path.getsize(stego)
        diff = stego_size - cover_size
        if payload_length:
            diff -= payload_length
        percent = diff / cover_size * 100
        if verbose:
            click.echo(f"Cover: {cover_size} bytes, Stego: {stego_size} bytes, Diff: {diff} bytes ({percent:.2f}%)")
        results.append((cover, stego, diff, percent))
    return results

# }}}
