import sys

import click

from aletheialib.utils import download_F5


@click.group()
def brute_force():
    """Brute force attacks."""
    pass


@brute_force.command()
@click.argument('unhide_command')
@click.argument('passw_file')
def generic(unhide_command, passw_file):
    """Brute force with a generic command.

    UNHIDE_COMMAND is the command to unhide the data.\n
    PASSW_FILE is the file with the passwords.

    Example:
    \b
    aletheia brute-force-generic 'steghide extract -sf image.jpg -xf output.txt -p <PASSWORD> -f' resources/passwords.txt
    """

    import aletheialib.brute_force
    aletheialib.brute_force.generic(unhide_command, passw_file)


@brute_force.command()
@click.argument('image')
@click.argument('passw_file')
def steghide(image, passw_file):
    """Brute force steghide.

    IMAGE is the image to analyze.\n
    PASSW_FILE is the file with the passwords.

    Example:
    \b
    aletheia brute-force-steghide image.jpg resources/passwords.txt
    """

    import aletheialib.brute_force
    aletheialib.brute_force.steghide(image, passw_file)


@brute_force.command()
@click.argument('image')
@click.argument('passw_file')
def outguess(image, passw_file):
    """Brute force outguess.

    IMAGE is the image to analyze.\n
    PASSW_FILE is the file with the passwords.

    Example:
    \b
    aletheia brute-force-outguess image.jpg resources/passwords.txt
    """

    import aletheialib.brute_force
    aletheialib.brute_force.outguess(image, passw_file)


@brute_force.command()
@click.argument('image')
@click.argument('passw_file')
def openstego(image, passw_file):
    """Brute force openstego.

    IMAGE is the image to analyze.\n
    PASSW_FILE is the file with the passwords.

    Example:
    \b
    aletheia brute-force-openstego image.jpg resources/passwords.txt
    """

    import aletheialib.brute_force
    aletheialib.brute_force.openstego(image, passw_file)


@brute_force.command()
@click.argument('image')
@click.argument('passw_file')
def f5(image, passw_file):
    """Brute force F5.

    IMAGE is the image to analyze.\n
    PASSW_FILE is the file with the passwords.

    Example:
    \b
    aletheia brute-force-f5 image.jpg resources/passwords.txt
    """

    download_F5()

    import aletheialib.brute_force
    aletheialib.brute_force.f5(image, passw_file)


@brute_force.command()
@click.argument('image')
@click.argument('passw_file')
def stegosuite(image, passw_file):
    """Brute force Stegosuite.

    IMAGE is the image to analyze.\n
    PASSW_FILE is the file with the passwords.

    Example:
    \b
    aletheia brute-force-stegosuite image.jpg resources/passwords.txt
    """

    import aletheialib.brute_force
    aletheialib.brute_force.stegosuite(image, passw_file)
