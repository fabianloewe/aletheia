import os
import sys
from pathlib import Path

import click


@click.group()
def tools():
    """Tools."""
    pass


@tools.command()
@click.argument('input')
@click.argument('output')
@click.pass_context
def hpf(ctx, input, output):
    """High-pass filter.

    INPUT is the image to filter.\n
    OUTPUT is the output image.

    NOTE: Batch mode is supported and INPUT and OUTPUT must be directories if activated.
    """

    import aletheialib.attacks
    import aletheialib.utils

    if ctx.obj['batch']:
        input_files = Path(input).rglob("*.*")
        output_dir = Path(output)
        assert output_dir.is_dir(), "Output must be a directory in batch mode."
        with click.progressbar(input_files, label="Processing images") as bar:
            for f in bar:
                if not aletheialib.utils.is_valid_image(f):
                    click.echo(f"{f} is not a valid image")
                    continue
                aletheialib.attacks.high_pass_filter(f, output_dir / f.name)
    else:
        aletheialib.attacks.high_pass_filter(input, output)
    sys.exit(0)


@tools.command()
@click.argument('cover')
@click.argument('stego')
def print_diffs(cover, stego):
    """Differences between two images.

    COVER is the cover image.\n
    STEGO is the stego image.
    """

    import aletheialib.utils
    import aletheialib.attacks

    cover = aletheialib.utils.absolute_path(cover)
    stego = aletheialib.utils.absolute_path(stego)
    if not os.path.isfile(cover):
        click.echo(f"Cover file not found: {cover}")
        sys.exit(0)
    if not os.path.isfile(stego):
        click.echo(f"Stego file not found: {stego}")
        sys.exit(0)

    aletheialib.attacks.print_diffs(cover, stego)
    sys.exit(0)


@tools.command()
@click.argument('cover')
@click.argument('stego')
def print_dct_diffs(cover, stego):
    """Differences between the DCT coefficients of two JPEG images.

    COVER is the cover image.\n
    STEGO is the stego image.
    """

    import aletheialib.attacks
    import aletheialib.utils

    cover = aletheialib.utils.absolute_path(cover)
    stego = aletheialib.utils.absolute_path(stego)

    if not os.path.isfile(cover):
        click.echo(f"Cover file not found: {cover}")
        sys.exit(0)
    if not os.path.isfile(stego):
        click.echo(f"Stego file not found: {stego}")
        sys.exit(0)

    name, ext = os.path.splitext(cover)
    if ext.lower() not in [".jpeg", ".jpg"] or not os.path.isfile(cover):
        click.echo("Please, provide a JPEG image!\n")
        sys.exit(0)

    name, ext = os.path.splitext(stego)
    if ext.lower() not in [".jpeg", ".jpg"] or not os.path.isfile(stego):
        click.echo("Please, provide a JPEG image!\n")
        sys.exit(0)

    aletheialib.attacks.print_dct_diffs(cover, stego)
    sys.exit(0)


@tools.command()
@click.argument('input')
@click.argument('output')
@click.pass_context
def rm_alpha(ctx, input, output):
    """Opacity of the alpha channel to 255.

    INPUT is the input image.\n
    OUTPUT is the output image.

    NOTE: Batch mode is supported and INPUT and OUTPUT must be directories if activated.
    """

    import aletheialib.utils
    import aletheialib.attacks

    if ctx.obj['batch']:
        input_files = Path(input).rglob("*.*")
        output_dir = Path(output)
        assert output_dir.is_dir(), "Output must be a directory in batch mode."
        with click.progressbar(input_files, label="Processing images") as bar:
            for f in bar:
                if not aletheialib.utils.is_valid_image(f):
                    click.echo(f"{f} is not a valid image")
                    continue
                aletheialib.attacks.remove_alpha_channel(f, output_dir / f.name)
    else:
        aletheialib.attacks.remove_alpha_channel(input, output)
    sys.exit(0)


@tools.command()
@click.argument('image')
def plot_histogram(image):
    """Plot histogram.

    IMAGE is the image to plot.
    """

    import imageio
    import aletheialib.utils
    from matplotlib import pyplot as plt

    fn = aletheialib.utils.absolute_path(image)
    I = imageio.imread(fn)
    data = []
    if len(I.shape) == 1:
        data.append(I.flatten())
    else:
        for i in range(I.shape[2]):
            data.append(I[:, :, i].flatten())

    plt.hist(data, range(0, 255), color=["r", "g", "b"])
    plt.show()
    sys.exit(0)


@tools.command()
@click.argument('image')
def plot_dct_histogram(image):
    """Plot DCT histogram.

    IMAGE is the JPEG image to plot.
    """

    import aletheialib.utils
    import aletheialib.jpeg
    from matplotlib import pyplot as plt

    fn = aletheialib.utils.absolute_path(image)
    name, ext = os.path.splitext(fn)
    if ext.lower() not in [".jpeg", ".jpg"] or not os.path.isfile(fn):
        click.echo("Please, provide a JPEG image!\n")
        sys.exit(0)
    I = aletheialib.jpeg.JPEG(fn)
    channels = ["r", "g", "b"]
    dct_list = []
    for i in range(I.components()):
        dct = I.coeffs(i).flatten()
        dct_list.append(dct)
        # counts, bins = np.histogram(dct, range(-5, 5))
        # plt.plot(bins[:-1], counts, channels[i])
    plt.hist(dct_list, range(-10, 10), rwidth=1, color=["r", "g", "b"])

    plt.show()
    sys.exit(0)


@tools.command()
@click.argument('image')
@click.option('-w', '--width-range', help="Width range to plot", type=(int, int), required=True)
@click.option('-h', '--height-range', help="Height range to plot", type=(int, int), required=True)
@click.pass_context
def print_pixels(ctx, image, width_range, height_range):
    """Print a range of pixels.

    IMAGE is the image to plot.

    Example:
    \b
    python aletheia.py print-pixels test.png -w 400 410 -h 200 220

    NOTE: Batch mode is supported and IMAGE must be a directory if activated.
    """

    import imageio

    def _print_pixels(image, width_range, height_range):
        ws, we = width_range
        hs, he = height_range

        I = imageio.imread(image)

        if len(I.shape) == 2:
            click.echo(f"Image shape: {I.shape}")
            click.echo(I[hs:he, ws:we])
        else:
            click.echo(f"Image shape: {I.shape[:2]}")
            for ch in range(I.shape[2]):
                click.echo(f"Channel: {ch}")
                click.echo(I[hs:he, ws:we, ch])
                click.echo()

    if ctx.obj['batch']:
        input_files = Path(image).rglob("*.*")
        for f in input_files:
            click.echo(f'Processing {f}...')
            _print_pixels(f, width_range, height_range)
    else:
        _print_pixels(image, width_range, height_range)


@tools.command()
@click.argument('image')
@click.option('-w', '--width-range', help="Width range to plot", type=(int, int), required=True)
@click.option('-h', '--height-range', help="Height range to plot", type=(int, int), required=True)
@click.pass_context
def print_coeffs(ctx, image, width_range, height_range):
    """Print a range of JPEG coefficients.

    IMAGE is the JPEG image to plot.

    Example:
    \b
    python aletheia.py print-coeffs test.jpg -w 400 410 -h 200 220

    NOTE: Batch mode is supported and IMAGE must be a directory if activated.
    """

    import aletheialib.utils
    from aletheialib.jpeg import JPEG

    def _print_coeffs(image, width_range, height_range):
        ws, we = width_range
        hs, he = height_range

        fn, ext = os.path.splitext(image)
        if ext[1:].lower() not in ["jpg", "jpeg"]:
            click.echo("ERROR: Please, provide a JPEG image")
            sys.exit(0)

        img = aletheialib.utils.absolute_path(image)
        im_jpeg = JPEG(img)

        for i in range(im_jpeg.components()):
            coeffs = im_jpeg.coeffs(i)
            click.echo(f"Image shape: {coeffs.shape}")
            click.echo(f"Channel: {i}")
            click.echo(coeffs[hs:he, ws:we])
            click.echo()

    if ctx.obj['batch']:
        input_files = Path(image).rglob("*.*")
        for f in input_files:
            click.echo(f'Processing {f}...')
            _print_coeffs(f, width_range, height_range)
    else:
        _print_coeffs(image, width_range, height_range)


@tools.command()
@click.argument('image')
@click.argument('output')
@click.pass_context
def eof_extract(ctx, image, output):
    """Extract the data after EOF.

    IMAGE is the image to extract.
    OUTPUT is the output file.

    NOTE: Batch mode is supported and INPUT and OUTPUT must be directories if activated.
    """

    import aletheialib.attacks
    import aletheialib.utils

    if ctx.obj['batch']:
        input_files = Path(image).rglob("*.*")
        output_dir = Path(output)
        assert output_dir.is_dir(), "Output must be a directory in batch mode."

        for f in input_files:
            if not aletheialib.utils.is_valid_image(f):
                click.echo(f"{f} is not a valid image")
                continue
            click.echo(f'Processing {f}...')
            aletheialib.attacks.eof_extract(f, output_dir / f.name)
    else:
        if not os.path.isfile(image):
            click.echo("Please, provide a valid image!\n")

        aletheialib.attacks.eof_extract(image, output)
    sys.exit(0)


@tools.command()
@click.argument('input')
@click.pass_context
def print_metadata(ctx, input):
    """Print Exif metadata.

    INPUT is the image to analyze.

    NOTE: Batch mode is supported and INPUT must be a directory if activated.
    """

    import aletheialib.attacks

    if ctx.obj['batch']:
        input_files = Path(input).rglob("*.*")
        for f in input_files:
            click.echo(f'Processing {f}...')
            aletheialib.attacks.exif(f)
    else:
        if not os.path.isfile(input):
            click.echo("Please, provide a valid image!\n")

        aletheialib.attacks.exif(input)
    sys.exit(0)


@tools.command()
@click.argument('input')
@click.option('--num-lsbs', default=1, help="Number of LSBs to extract")
@click.option('--channels', default="RGB", help="Channels to extract")
@click.option('--endian', default="little", help="The endianness to use for recovering the data")
@click.pass_context
def lsb_extract(ctx, input, num_lsbs, channels, endian):
    """Extract data from the LSBs.

    INPUT is the image to analyze.

    NOTE: Batch mode is supported and INPUT must be a directory if activated.
    """

    import aletheialib.attacks

    if ctx.obj['batch']:
        input_files = Path(input).rglob("*.*")
        with click.progressbar(input_files, label="Extracting LSBs") as bar:
            for f in bar:
                aletheialib.attacks.lsb_extract(f, num_lsbs, channels, endian)
    else:
        if not os.path.isfile(input):
            click.echo("Please, provide a valid image!\n")

        aletheialib.attacks.lsb_extract(input, num_lsbs, channels, endian)
    sys.exit(0)
