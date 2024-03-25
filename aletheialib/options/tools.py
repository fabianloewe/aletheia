import os
import sys

import click


@click.group()
def tools():
    """Tools."""
    pass


@tools.command()
@click.argument('input')
@click.argument('output')
def hpf(input, output):
    """High-pass filter.

    INPUT is the image to filter.\n
    OUTPUT is the output image.
    """

    import aletheialib.attacks
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
        click.echo("Cover file not found:", cover)
        sys.exit(0)
    if not os.path.isfile(stego):
        click.echo("Stego file not found:", stego)
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
        click.echo("Cover file not found:", cover)
        sys.exit(0)
    if not os.path.isfile(stego):
        click.echo("Stego file not found:", stego)
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
def rm_alpha(input, output):
    """Opacity of the alpha channel to 255.

    INPUT is the input image.\n
    OUTPUT is the output image.
    """

    import aletheialib.utils
    import aletheialib.attacks

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
def plot_dct_histogram():
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
def print_pixels(image, width_range, height_range):
    """Print a range of pixels.

    IMAGE is the image to plot.

    Example:
    \b
    python aletheia.py click.echo-pixels test.png -w 400 410 -h 200 220
    """

    import imageio
    I = imageio.imread(image)

    ws, we = width_range
    hs, he = height_range

    if len(I.shape) == 2:
        click.echo("Image shape:", I.shape[:2])
        click.echo(I[hs:he, ws:we])
    else:
        click.echo("Image shape:", I.shape[:2])
        for ch in range(I.shape[2]):
            click.echo("Channel:", ch)
            click.echo(I[hs:he, ws:we, ch])
            click.echo()


@tools.command()
@click.argument('image')
@click.option('-w', '--width-range', help="Width range to plot", type=(int, int), required=True)
@click.option('-h', '--height-range', help="Height range to plot", type=(int, int), required=True)
def print_coeffs(image, width_range, height_range):
    """Print a range of JPEG coefficients.

    IMAGE is the JPEG image to plot.

    Example:
    \b
    python aletheia.py click.echo-coeffs test.jpg -w 400 410 -h 200 220
    """

    ws, we = width_range
    hs, he = height_range

    fn, ext = os.path.splitext(sys.argv[2])
    if ext[1:].lower() not in ["jpg", "jpeg"]:
        click.echo("ERROR: Please, provide a JPEG image")
        sys.exit(0)

    import aletheialib.utils
    from aletheialib.jpeg import JPEG

    img = aletheialib.utils.absolute_path(sys.argv[2])
    im_jpeg = JPEG(img)

    for i in range(im_jpeg.components()):
        coeffs = im_jpeg.coeffs(i)
        click.echo("Image shape:", coeffs.shape)
        click.echo("Channel:", i)
        click.echo(coeffs[hs:he, ws:we])
        click.echo()


@tools.command()
@click.argument('image')
@click.argument('output')
def eof_extract(image, output):
    """Extract the data after EOF.

    IMAGE is the image to extract.
    OUTPUT is the output file.
    """

    if not os.path.isfile(image):
        click.echo("Please, provide a valid image!\n")

    import aletheialib.attacks
    aletheialib.attacks.eof_extract(image, output)
    sys.exit(0)


@tools.command()
@click.argument('input')
def print_metadata(input):
    """Print Exif metadata.

    INPUT is the image to analyze.
    """

    if not os.path.isfile(input):
        click.echo("Please, provide a valid image!\n")

    import aletheialib.attacks
    aletheialib.attacks.exif(input)
    sys.exit(0)


@tools.command()
@click.argument('input')
@click.option('--num-lsbs', default=1, help="Number of LSBs to extract")
@click.option('--channels', default="RGB", help="Channels to extract")
@click.option('--endian', default="little", help="The endianness to use for recovering the data")
def lsb_extract(input, num_lsbs, channels, endian):
    """Extract data from the LSBs.

    INPUT is the image to analyze.
    """

    if not os.path.isfile(input):
        click.echo("Please, provide a valid image!\n")

    import aletheialib.attacks
    aletheialib.attacks.lsb_extract(input, num_lsbs, channels, endian)
    sys.exit(0)
