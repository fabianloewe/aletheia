import os
import sys
import glob

import click


@click.group()
def calibration():
    """Calibration attacks."""
    pass


@calibration.command("calibration")
@click.argument('input')
def launch():
    """Calibration attacks to JPEG steganography using F5.

    INPUT is the image or directory with the images to analyze.
    """

    # if sys.argv[2] not in ["f5", "chisquare_mode"]:
    #    click.echo("Please, provide a valid method")
    #    sys.exit(0)

    import aletheialib.utils
    import aletheialib.attacks

    if os.path.isdir(sys.argv[2]):
        for f in glob.glob(os.path.join(sys.argv[2], "*")):
            if not aletheialib.utils.is_valid_image(f):
                click.echo(f, " is not a valid image")
                continue

            if not f.lower().endswith(('.jpg', '.jpeg')):
                click.echo(f, "is not a JPEG file")
                continue

            try:
                fn = aletheialib.utils.absolute_path(f)
                beta = aletheialib.attacks.calibration_f5_octave_jpeg(fn, True)
                click.echo(f, ", beta:", beta)
            except:
                click.echo("Error processing", f)
    else:

        if not aletheialib.utils.is_valid_image(sys.argv[2]):
            click.echo("Please, provide a valid image")
            sys.exit(0)

        if not sys.argv[2].lower().endswith(('.jpg', '.jpeg')):
            click.echo("Please, provide a JPEG file")
            sys.exit(0)

        fn = aletheialib.utils.absolute_path(sys.argv[2])
        aletheialib.attacks.calibration_f5_octave_jpeg(fn)
        # click.echo(aletheialib.attacks.calibration_f5_octave_jpeg(fn, True))

        # if "f5" in sys.argv[2]:
        #    aletheialib.attacks.calibration_f5(fn)
        # elif "chisquare_mode" in sys.argv[2]:
        #    aletheialib.attacks.calibration_chisquare_mode(fn)
