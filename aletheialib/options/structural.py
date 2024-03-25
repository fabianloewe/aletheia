import os
import glob
import sys
from pathlib import Path

import click
import numpy as np
import aletheialib.utils
import aletheialib.attacks
from aletheialib.utils import download_octave_code

from imageio import imread
from multiprocessing import Pool as ThreadPool
from multiprocessing import cpu_count

from PIL import Image
import aletheialib.octave_interface as O


@click.group("struct")
def structural():
    """Structural LSB detectors (Statistical attacks to LSB replacement)."""
    pass


@structural.command()
@click.argument('input')
@click.option('--threshold', default=0.05, help='Threshold for detecting steganographic images (default=0.05)')
def spa(input, threshold):
    """Sample Pairs Analysis (SPA) attack to detect LSB replacement steganography.

    INPUT is the image to analyze.
    """

    import aletheialib.utils
    import aletheialib.attacks
    from imageio import imread

    if not aletheialib.utils.is_valid_image(input):
        click.echo("Please, provide a valid image")
        sys.exit(0)

    click.echo("Using threshold:", threshold)

    I = imread(input)
    if len(I.shape) == 2:
        bitrate = aletheialib.attacks.spa_image(I, None)
        if bitrate < threshold:
            click.echo("No hidden data found")
        else:
            click.echo("Hidden data found"), bitrate
    else:
        bitrate_R = aletheialib.attacks.spa_image(I, 0)
        bitrate_G = aletheialib.attacks.spa_image(I, 1)
        bitrate_B = aletheialib.attacks.spa_image(I, 2)

        if bitrate_R < threshold and bitrate_G < threshold and bitrate_B < threshold:
            click.echo("No hidden data found")
            sys.exit(0)

        if bitrate_R >= threshold:
            click.echo("Hidden data found in channel R", bitrate_R)
        if bitrate_G >= threshold:
            click.echo("Hidden data found in channel G", bitrate_G)
        if bitrate_B >= threshold:
            click.echo("Hidden data found in channel B", bitrate_B)
    sys.exit(0)


@structural.command()
@click.argument('input')
@click.option('--threshold', default=0.05, help='Threshold for detecting steganographic images (default=0.05)')
def rs(input, threshold):
    """RS attack to detect LSB replacement steganography.

    INPUT is the image to analyze.
    """

    import numpy as np
    import aletheialib.utils
    import aletheialib.attacks
    from imageio import imread

    if not aletheialib.utils.is_valid_image(input):
        click.echo("Please, provide a valid image")
        sys.exit(0)

    click.echo("Using threshold:", threshold)

    I = np.asarray(imread(input))
    if len(I.shape) == 2:
        bitrate = aletheialib.attacks.rs_image(I)
        if bitrate < threshold:
            click.echo("No hidden data found")
        else:
            click.echo("Hidden data found", bitrate)
    else:
        bitrate_R = aletheialib.attacks.rs_image(I, 0)
        bitrate_G = aletheialib.attacks.rs_image(I, 1)
        bitrate_B = aletheialib.attacks.rs_image(I, 2)

        if bitrate_R < threshold and bitrate_G < threshold and bitrate_B < threshold:
            click.echo("No hidden data found")
            sys.exit(0)

        if bitrate_R >= threshold:
            click.echo("Hidden data found in channel R", bitrate_R)
        if bitrate_G >= threshold:
            click.echo("Hidden data found in channel G", bitrate_G)
        if bitrate_B >= threshold:
            click.echo("Hidden data found in channel B", bitrate_B)
        sys.exit(0)


@structural.command()
@click.argument('input')
@click.option('--threshold', default=0.05, help='Threshold for detecting steganographic images (default=0.05)')
def ws(input, threshold):
    """Weighted Stego Attack (WS) to detect LSB replacement steganography.

    INPUT is the image to analyze.
    """

    download_octave_code("WS")

    if not aletheialib.utils.is_valid_image(input):
        click.echo("Please, provide a valid image")
        sys.exit(0)

    click.echo("Using threshold:", threshold)

    path = aletheialib.utils.absolute_path(sys.argv[2])
    im = Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('WS', path, params={"channel": 1})["data"][0][0]
        alpha_G = O._attack('WS', path, params={"channel": 2})["data"][0][0]
        alpha_B = O._attack('WS', path, params={"channel": 3})["data"][0][0]

        if alpha_R < threshold and alpha_G < threshold and alpha_B < threshold:
            click.echo("No hidden data found")

        if alpha_R >= threshold:
            click.echo("Hidden data found in channel R", alpha_R)
        if alpha_G >= threshold:
            click.echo("Hidden data found in channel G", alpha_G)
        if alpha_B >= threshold:
            click.echo("Hidden data found in channel B", alpha_B)

    else:
        alpha = O._attack('WS', path, params={"channel": 1})["data"][0][0]
        if alpha >= threshold:
            click.echo("Hidden data found", alpha)
        else:
            click.echo("No hidden data found")

    sys.exit(0)


@structural.command()
@click.argument('input')
@click.option('--threshold', default=0.05, help='Threshold for detecting steganographic images (default=0.05)')
def triples(input, threshold):
    """Triples attack to detect LSB replacement steganography.

    INPUT is the image to analyze.
    """

    download_octave_code("TRIPLES")

    if not aletheialib.utils.is_valid_image(input):
        click.echo("Please, provide a valid image")
        sys.exit(0)

    click.echo("Using threshold:", threshold)

    path = aletheialib.utils.absolute_path(input)
    im = Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('TRIPLES', path, params={"channel": 1})["data"][0][0]
        alpha_G = O._attack('TRIPLES', path, params={"channel": 2})["data"][0][0]
        alpha_B = O._attack('TRIPLES', path, params={"channel": 3})["data"][0][0]

        if alpha_R < threshold and alpha_G < threshold and alpha_B < threshold:
            click.echo("No hidden data found")

        if alpha_R >= threshold:
            click.echo("Hidden data found in channel R", alpha_R)
        if alpha_G >= threshold:
            click.echo("Hidden data found in channel G", alpha_G)
        if alpha_B >= threshold:
            click.echo("Hidden data found in channel B", alpha_B)

    else:
        alpha = O._attack('TRIPLES', path, params={"channel": 1})["data"][0][0]
        if alpha >= threshold:
            click.echo("Hidden data found", alpha)
        else:
            click.echo("No hidden data found")

    sys.exit(0)


@structural.command()
@click.argument('input')
@click.option('--threshold', default=0.05, help='Threshold for detecting steganographic images (default=0.05)')
def aump(input, threshold):
    """AUMP attack to detect LSB replacement steganography.

    INPUT is the image to analyze.
    """

    download_octave_code("AUMP")

    if not aletheialib.utils.is_valid_image(input):
        click.echo("Please, provide a valid image")
        sys.exit(0)

    click.echo("Using threshold:", threshold)

    path = aletheialib.utils.absolute_path(input)
    im = Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('AUMP', path, params={"channel": 1})["data"][0][0]
        alpha_G = O._attack('AUMP', path, params={"channel": 2})["data"][0][0]
        alpha_B = O._attack('AUMP', path, params={"channel": 3})["data"][0][0]

        if alpha_R < threshold and alpha_G < threshold and alpha_B < threshold:
            click.echo("No hidden data found")

        if alpha_R >= threshold:
            click.echo("Hidden data found in channel R", alpha_R)
        if alpha_G >= threshold:
            click.echo("Hidden data found in channel G", alpha_G)
        if alpha_B >= threshold:
            click.echo("Hidden data found in channel B", alpha_B)
    else:
        alpha = O._attack('WS', path, params={"channel": 1})["data"][0][0]
        if alpha >= threshold:
            click.echo("Hidden data found", alpha)
        else:
            click.echo("No hidden data found")

    sys.exit(0)


def spa_detect(p):
    f, threshold = p
    I = imread(f)
    if len(I.shape) == 2:
        bitrate = aletheialib.attacks.spa_image(I, None)
    else:
        bitrate_R = aletheialib.attacks.spa_image(I, 0)
        bitrate_G = aletheialib.attacks.spa_image(I, 1)
        bitrate_B = aletheialib.attacks.spa_image(I, 2)
        bitrate = max(bitrate_R, bitrate_G, bitrate_B)

    if bitrate < threshold:
        return False
    return True


@structural.command()
@click.option('--test-cover-dir', required=True, help='Directory containing cover images',
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--test-stego-dir', required=True, help='Directory containing stego images',
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--threshold', default=0.05, help='Threshold for detecting steganographic images (default=0.05)')
def spa_score(test_cover_dir, test_stego_dir, threshold):
    """Score for Sample Pairs Analysis (SPA) attack to detect LSB replacement steganography."""
    cover_files = sorted(test_cover_dir.glob('*'))
    stego_files = sorted(test_stego_dir.glob('*'))

    click.echo("Using threshold", threshold)

    batch = 1000
    n_core = cpu_count()

    # Process thread pool in batches
    for i in range(0, len(cover_files), batch):
        params_batch = zip(cover_files[i:i + batch], [threshold] * batch)
        pool = ThreadPool(n_core)
        pred_cover = pool.map(spa_detect, params_batch)
        pool.close()
        pool.terminate()
        pool.join()

    for i in range(0, len(stego_files), batch):
        params_batch = zip(stego_files[i:i + batch], [threshold] * batch)
        pool = ThreadPool(n_core)
        pred_stego = pool.map(spa_detect, params_batch)
        pool.close()
        pool.terminate()
        pool.join()

    ok = np.sum(np.array(pred_cover) == 0) + np.sum(np.array(pred_stego) == 1)
    score = ok / (len(pred_cover) + len(pred_stego))
    click.echo("score:", score)


# }}}

# {{{ ws_score
def ws_detect(f):
    if not aletheialib.utils.is_valid_image(f):
        click.echo("Please, provide a valid image:", f)
        return False

    threshold = 0.05
    path = aletheialib.utils.absolute_path(f)
    im = Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('WS', path, params={"channel": 1})["data"][0][0]
        alpha_G = O._attack('WS', path, params={"channel": 2})["data"][0][0]
        alpha_B = O._attack('WS', path, params={"channel": 3})["data"][0][0]
        alpha = max(alpha_R, alpha_G, alpha_B)
    else:
        alpha = O._attack('WS', path, params={"channel": 1})["data"][0][0]

    if alpha < threshold:
        return False
    return True


@structural.command()
@click.option('--test-cover-dir', required=True, help='Directory containing cover images',
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--test-stego-dir', required=True, help='Directory containing stego images',
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
def ws_score(test_cover_dir, test_stego_dir):
    """Score for Weighted Stego Attack (WS) to detect LSB replacement steganography."""

    cover_files = sorted(test_cover_dir.glob('*'))
    stego_files = sorted(test_stego_dir.glob('*'))

    batch = 1000
    n_core = cpu_count()

    # Process thread pool in batches
    for i in range(0, len(cover_files), batch):
        params_batch = cover_files[i:i + batch]
        pool = ThreadPool(n_core)
        pred_cover = pool.map(ws_detect, params_batch)
        pool.close()
        pool.terminate()
        pool.join()

    for i in range(0, len(stego_files), batch):
        params_batch = stego_files[i:i + batch]
        pool = ThreadPool(n_core)
        pred_stego = pool.map(ws_detect, params_batch)
        pool.close()
        pool.terminate()
        pool.join()

    ok = np.sum(np.array(pred_cover) == 0) + np.sum(np.array(pred_stego) == 1)
    score = ok / (len(pred_cover) + len(pred_stego))
    click.echo("score:", score)
