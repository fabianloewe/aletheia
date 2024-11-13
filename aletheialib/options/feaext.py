import sys

import click

from aletheialib.utils import download_octave_code, download_octave_jpeg_toolbox


@click.group()
def feaext():
    """Feature extraction."""
    pass


@feaext.command()
@click.argument('input')
@click.argument('output-file')
def srm(input, output_file):
    """Full Spatial Rich Models.

    INPUT is the image or directory with the images to analyze.\n
    OUTPUT_FILE is the file to store the features in.
    """

    download_octave_code("SRM")

    import aletheialib.feaext

    aletheialib.feaext.extract_features(aletheialib.feaext.SRM_extract, input, output_file)
    sys.exit(0)


@feaext.command()
@click.argument('input')
@click.argument('output-file')
def srmq1(input, output_file):
    """Spatial Rich Models with fixed quantization q=1c.

    INPUT is the image or directory with the images to analyze.\n
    OUTPUT_FILE is the file to store the features in.
    """

    download_octave_code("SRMQ1")

    import aletheialib.feaext

    aletheialib.feaext.extract_features(aletheialib.feaext.SRMQ1_extract, input, output_file)
    sys.exit(0)


@feaext.command()
@click.argument('input')
@click.argument('output-file')
def scrmq1(input, output_file):
    """Spatial Color Rich Models with fixed quantization q=1c.

    INPUT is the image or directory with the images to analyze.\n
    OUTPUT_FILE is the file to store the features in.
    """

    download_octave_code("SCRMQ1")

    import aletheialib.feaext

    aletheialib.feaext.extract_features(aletheialib.feaext.SCRMQ1_extract, input, output_file)
    sys.exit(0)


@feaext.command()
@click.argument('input')
@click.argument('output-file')
@click.option('--quality', default="auto", help="JPEG quality")
@click.option('--rotations', default=32, help="Number of rotations for Gabor kernel")
def gfr(input, output_file, quality, rotations):
    """JPEG steganalysis with 2D Gabor Filters.

    INPUT is the image or directory with the images to analyze.\n
    OUTPUT_FILE is the file to store the features in.
    """

    download_octave_jpeg_toolbox()
    download_octave_code("GFR")

    import aletheialib.feaext

    if quality == "auto":
        click.echo("JPEG quality not provided, using detection via 'identify'")

    if rotations == 32:
        click.echo(f"Number of rotations for Gabor kernel no provided, using: {rotations}")

    params = {
        "quality": quality,
        "rotations": rotations
    }

    aletheialib.feaext.extract_features(aletheialib.feaext.GFR_extract, input, output_file, params)
    sys.exit(0)


@feaext.command()
@click.argument('input')
@click.argument('output-file')
@click.option('--quality', default="auto", help="JPEG quality")
def dctr(input, output_file, quality):
    """JPEG Low complexity features extracted from DCT residuals.

    INPUT is the image or directory with the images to analyze.\n
    OUTPUT_FILE is the file to store the features in.
    """

    download_octave_jpeg_toolbox()
    download_octave_code("DCTR")

    import aletheialib.feaext

    if quality == "auto":
        click.echo("JPEG quality not provided, using detection via 'identify'")

    params = {
        "quality": quality,
    }

    aletheialib.feaext.extract_features(aletheialib.feaext.DCTR_extract, input, output_file, params)
    sys.exit(0)
