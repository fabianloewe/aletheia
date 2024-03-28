import glob
import os
import sys
from pathlib import Path

import click


@click.group()
def auto_group():
    """Automatic steganalysis."""
    pass

def _format_line(value, length):
    if value > 0.5:
        return ("[" + str(round(value, 1)) + "]").center(length, ' ')

    return str(round(value, 1)).center(length, ' ')


def load_model(nn, model_name):
    # Get the directory where the models are installed
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, os.pardir, os.pardir, 'aletheia-models')

    model_path = os.path.join(dir_path, model_name + ".h5")
    if not os.path.isfile(model_path):
        click.echo(f"ERROR: Model file not found: {model_path}\n")
        sys.exit(-1)
    nn.load_model(model_path, quiet=True)
    return nn


@auto_group.command()
@click.argument('input', type=click.Path(exists=True, resolve_path=True, path_type=Path))
@click.option('--dev', default="CPU", help="The device to use (CPU or GPU)")
def auto(input: Path, dev):
    """Attempts to detect steganographic content in images automatically.

    INPUT is the image or directory with the images to analyze.
    """

    if dev == "CPU":
        click.echo("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import aletheialib.models

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if input.is_dir():
        files = list(input.glob('*.*'))
    else:
        files = [input]

    nn = aletheialib.models.NN("effnetb0")
    files = nn.filter_images(files)
    if len(files) == 0:
        click.echo("ERROR: please provide valid files")
        sys.exit(0)

    jpg_files = []
    bitmap_files = []
    for f in files:
        _, ext = os.path.splitext(f)
        if ext.lower() in ['.jpg', '.jpeg']:
            jpg_files.append(f)
        else:
            bitmap_files.append(f)

    # TODO: Find model paths

    # JPG
    if len(jpg_files) > 0:

        nn = load_model(nn, "effnetb0-A-alaska2-outguess")
        outguess_pred = nn.predict(jpg_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-steghide")
        steghide_pred = nn.predict(jpg_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-nsf5")
        nsf5_pred = nn.predict(jpg_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-juniw")
        juniw_pred = nn.predict(jpg_files, 10)

        mx = 20
        click.echo("")
        click.echo(' ' * mx + " Outguess  Steghide   nsF5  J-UNIWARD *")
        click.echo('-' * mx + "---------------------------------------")
        for i in range(len(jpg_files)):
            name = os.path.basename(jpg_files[i])
            if len(name) > mx:
                name = name[:mx - 3] + "..."
            else:
                name = name.ljust(mx, ' ')

            click.echo(name,
                       _format_line(outguess_pred[i], 9),
                       _format_line(steghide_pred[i], 8),
                       _format_line(nsf5_pred[i], 8),
                       _format_line(juniw_pred[i], 8),
                       )

    # BITMAP
    if len(bitmap_files) > 0:

        nn = load_model(nn, "effnetb0-A-alaska2-lsbr")
        lsbr_pred = nn.predict(bitmap_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-lsbm")
        lsbm_pred = nn.predict(bitmap_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-steganogan")
        steganogan_pred = nn.predict(bitmap_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-hill")
        hill_pred = nn.predict(bitmap_files, 10)

        mx = 20
        click.echo("")
        click.echo(' ' * mx + "   LSBR      LSBM  SteganoGAN  HILL *")
        click.echo('-' * mx + "-------------------------------------")
        for i in range(len(bitmap_files)):
            name = os.path.basename(bitmap_files[i])
            if len(name) > mx:
                name = name[:mx - 3] + "..."
            else:
                name = name.ljust(mx, ' ')

            click.echo(name,
                       _format_line(lsbr_pred[i], 10),
                       _format_line(lsbm_pred[i], 8),
                       _format_line(steganogan_pred[i], 8),
                       _format_line(hill_pred[i], 8),
                       )

    click.echo("")
    click.echo("* Probability of steganographic content using the indicated method.\n")


# }}}

@auto_group.command()
@click.argument('sim')
@click.argument('img_dir', type=click.Path(exists=True, resolve_path=True, path_type=Path))
@click.option('--dev', default="CPU", help="The device to use (CPU or GPU)")
def dci(sim, img_dir: Path, dev):
    """Performs a steganalysis using the DCI method.

    SIM is the simulation method to use.
    IMG_DIR is the directory with the images.
    """

    files = list(img_dir.glob('*.*'))

    if dev == "CPU":
        click.echo("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if not os.path.isdir(img_dir):
        click.echo("ERROR: Please, provide a valid directory\n")
        sys.exit(0)

    if len(files) < 10:
        click.echo("ERROR: We need more images from the same actor\n")
        sys.exit(0)

    ext = os.path.splitext(files[0])[1].lower().replace('.jpeg', '.jpg')
    for f in files:
        curr_ext = os.path.splitext(f)[1].lower().replace('.jpeg', '.jpg')
        if ext != curr_ext:
            click.echo(f"ERROR: All images must be of the same type: {curr_ext}!={ext} \n")
            sys.exit(0)

    embed_fn_saving = False
    if ext == '.jpg':
        embed_fn_saving = True

    import shutil
    import tempfile
    import aletheialib.stegosim
    import aletheialib.models
    import numpy as np

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    A_nn = aletheialib.models.NN("effnetb0")
    A_files = files

    fn_sim = aletheialib.stegosim.embedding_fn(sim)
    method = sim
    method = method.replace("-sim", "")

    B_dir = tempfile.mkdtemp()
    click.echo("Preparind the B set ...")
    aletheialib.stegosim.embed_message(fn_sim, sys.argv[3], "0.40", B_dir,
                                       embed_fn_saving=embed_fn_saving)

    B_nn = aletheialib.models.NN("effnetb0")
    B_files = glob.glob(os.path.join(B_dir, '*'))

    # Make some replacements to adapt the name of the method with the name
    # of the model file
    method = method.replace("-color", "")
    method = method.replace("j-uniward", "juniw")

    A_nn = load_model(A_nn, "effnetb0-A-alaska2-" + method)
    B_nn = load_model(B_nn, "effnetb0-B-alaska2-" + method)

    # Predictions for the DCI method
    p_aa = A_nn.predict(A_files, 10)
    p_ab = A_nn.predict(B_files, 10)
    p_bb = B_nn.predict(B_files, 10)
    p_ba = B_nn.predict(A_files, 10)

    p_aa = np.round(p_aa).astype('uint8')
    p_ab = np.round(p_ab).astype('uint8')
    p_ba = np.round(p_ba).astype('uint8')
    p_bb = np.round(p_bb).astype('uint8')

    # Inconsistencies
    inc = ((p_aa != p_bb) | (p_ba != 0) | (p_ab != 1)).astype('uint8')
    inc1 = (p_aa != p_bb).astype('uint8')
    inc2 = ((p_ba != 0) | (p_ab != 1)).astype('uint8')
    inc2c = (p_ab != 1).astype('uint8')
    inc2s = (p_ba != 0).astype('uint8')

    for i in range(len(p_aa)):
        r = ""
        if inc[i]:
            r = str(round(p_aa[i], 3)) + " (inc)"
        else:
            r = round(p_aa[i], 3)
        click.echo(A_files[i], "\t", r)

    """
    click.echo("#inc:", np.sum(inc==1), "#incF1:", np.sum(inc1==1), "#incF2:", np.sum(inc2==1),
           "#incF2C", np.sum(inc2c), "#incF2S:", np.sum(inc2s))
    click.echo("#no_inc:", len(A_files)-np.sum(inc==1))
    click.echo("--")
    """
    click.echo("DCI prediction score:", round(1 - float(np.sum(inc == 1)) / (2 * len(p_aa)), 3))

    shutil.rmtree(B_dir)
