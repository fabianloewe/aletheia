import sys
import os

import click
import numpy as np
from imageio import imread, imwrite
from aletheialib.utils import download_octave_code
from aletheialib.utils import download_octave_aux_file
from aletheialib.utils import download_octave_jpeg_toolbox
from aletheialib.utils import download_F5

@click.group()
def embsim():
    """Embedding simulators."""
    pass


@embsim.command("lsbr-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def lsbr(input, payload, output_dir):
    """LSB replacement simulator."""

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.lsbr,
                                       input, payload, output_dir)
    sys.exit(0)


@embsim.command("lsbm-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def lsbm(input, payload, output_dir):
    """LSB matching simulator."""

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.lsbm,
                                       input, payload, output_dir)
    sys.exit(0)


@embsim.command("hugo-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def hugo(input, payload, output_dir):
    """HUGO simulator."""

    download_octave_code("HUGO")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.hugo,
                                       input, payload, output_dir)
    sys.exit(0)


@embsim.command("wow-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def wow(input, payload, output_dir):
    """WOW simulator."""

    download_octave_jpeg_toolbox()
    download_octave_code("WOW")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.wow,
                                       input, payload, output_dir)
    sys.exit(0)


@embsim.command("s-uniward-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def s_uniward(input, payload, output_dir):
    """Spatial UNIWARD simulator."""

    download_octave_code("S_UNIWARD")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.s_uniward,
                                       input, payload, output_dir)
    sys.exit(0)


@embsim.command("s-uniward-color-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def s_uniward_color(input, payload, output_dir):
    """Spatial UNIWARD color simulator."""

    download_octave_code("S_UNIWARD_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.s_uniward_color,
                                       input, payload, output_dir)
    sys.exit(0)


@embsim.command("hill-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def hill(input, payload, output_dir):
    """HILL simulator."""

    download_octave_code("HILL")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.hill,
                                       input, payload, output_dir)
    sys.exit(0)


@embsim.command("hill-color-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def hill_color(input, payload, output_dir):
    """HILL color simulator."""

    download_octave_code("HILL_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.hill_color,
                                       input, payload, output_dir)


@embsim.command("j-uniward-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def j_uniward(input, payload, output_dir):
    """JPEG UNIWARD simulator."""

    download_octave_jpeg_toolbox()
    download_octave_code("J_UNIWARD")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.j_uniward,
                                       input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("j-uniward-color-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def j_uniward_color(input, payload, output_dir):
    """JPEG UNIWARD color simulator."""

    download_octave_jpeg_toolbox()
    download_octave_code("J_UNIWARD_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.j_uniward_color,
                                       input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("j-mipod-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def j_mipod(input, payload, output_dir):
    """JPEG MiPOD simulator."""

    download_octave_jpeg_toolbox()
    download_octave_code("J_MIPOD")
    download_octave_aux_file("ixlnx3_5.mat")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.j_mipod,
                                       input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("j-mipod-color-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def j_mipod_color(input, payload, output_dir):
    """JPEG MiPOD color simulator."""

    download_octave_jpeg_toolbox()
    download_octave_code("J_MIPOD_COLOR")
    download_octave_aux_file("ixlnx3_5.mat")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.j_mipod_color,
                                       input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("ebs-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def ebs(input, payload, output_dir):
    """EBS simulator."""

    download_octave_jpeg_toolbox()
    download_octave_code("EBS")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.ebs,
                                       input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("ebs-color-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def ebs_color(input, payload, output_dir):
    """EBS color simulator."""

    download_octave_jpeg_toolbox()
    download_octave_code("EBS_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.ebs_color,
                                       input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("ued-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def ued(input, payload, output_dir):
    """UED simulator."""

    download_octave_jpeg_toolbox()
    download_octave_code("UED")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.ued,
                                       input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("ued-color-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def ued_color(input, payload, output_dir):
    """UED color simulator."""

    download_octave_jpeg_toolbox()
    download_octave_code("UED_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.ued_color,
                                       input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("nsf5-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def nsf5(input, payload, output_dir):
    """nsF5 simulator."""

    download_octave_jpeg_toolbox()
    download_octave_code("NSF5")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.nsf5,
                                       input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("nsf5-color-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def nsf5_color(input, payload, output_dir):
    """nsF5 color simulator."""

    download_octave_jpeg_toolbox()
    download_octave_code("NSF5_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.nsf5_color,
                                       input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("experimental-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def experimental(input, payload, output_dir):
    """Experimental simulator.

    NOTE: Please, put your EXPERIMENTAL.m file into external/octave\
    """

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.experimental, input, payload, output_dir)
    sys.exit(0)


@embsim.command("steghide-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def steghide(input, payload, output_dir):
    """Steghide simulator."""

    import aletheialib.stegosim
    import aletheialib.utils
    aletheialib.utils.check_bin("steghide")

    aletheialib.stegosim.embed_message(aletheialib.stegosim.steghide, input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("outguess-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def outguess(input, payload, output_dir):
    """Outguess simulator."""

    import aletheialib.stegosim
    import aletheialib.utils
    aletheialib.utils.check_bin("outguess")

    aletheialib.stegosim.embed_message(aletheialib.stegosim.outguess, input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("steganogan-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def steganogan(input, payload, output_dir):
    """SteganoGAN simulator."""

    import aletheialib.stegosim
    import aletheialib.utils
    aletheialib.utils.check_bin("steghide")

    aletheialib.stegosim.embed_message(aletheialib.stegosim.steganogan,
                                       input, payload, output_dir, embed_fn_saving=True)
    sys.exit(0)


@embsim.command("f5-sim")
@click.option('-i', '--input', required=True, help="Input image or directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
def f5(input, payload, output_dir):
    """F5 simulator."""

    import aletheialib.stegosim
    import aletheialib.utils

    download_octave_jpeg_toolbox()
    download_F5()

    aletheialib.stegosim.embed_message(aletheialib.stegosim.f5, input, payload, output_dir,
                                       embed_fn_saving=True)
    sys.exit(0)


@embsim.command("adversarial-adaptive-sim")
@click.option('-i', '--input', required=True, help="Images directory")
@click.option('-p', '--payload', required=True, help="Payload file")
@click.option('-o', '--output-dir', required=True, help="Output directory")
@click.option('-m', '--model-file', required=True, help="Model file")
@click.option('--dev', default="CPU", help="The device to use (CPU or GPU)")
def adversarial_adaptive(input, payload, output_dir, model_file, dev):
    """Adversarial adaptive simulator."""

    if dev == "CPU":
        click.echo("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import glob
    import aletheialib.stegosim
    import aletheialib.models

    nn = aletheialib.models.NN("effnetb0")
    nn.load_model(model_file)

    # Generate gradient files
    if not os.path.exists("/tmp/gradients"):
        os.mkdir("/tmp/gradients")

    for f in glob.glob(os.path.join(input, '*')):
        bn = os.path.basename(f)
        gradient = nn.get_gradient(f, 0)
        gradient_path = os.path.join("/tmp/gradients", bn)
        click.echo("Save gradient", gradient_path)
        np.save(gradient_path, gradient)

    aletheialib.stegosim.embed_message(
        aletheialib.stegosim.adversarial_adaptive, input, payload, output_dir)

    sys.exit(0)


@embsim.command("adversarial-fix")
@click.option('-i', '--input', required=True, help="Images directory")
@click.option('-o', '--output-dir', required=True, help="Output directory")
@click.option('-m', '--model-file', required=True, help="Model file")
@click.option('--dev', default="CPU", help="The device to use (CPU or GPU)")
def adversarial_fix(input, output_dir, model_file, dev):
    """Adversarial fix."""

    if dev == "CPU":
        click.echo("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import glob
    import aletheialib.stegosim
    import aletheialib.models

    nn = aletheialib.models.NN("effnetb0")
    nn.load_model(model_file)

    # Generate gradient files
    if not os.path.exists("/tmp/gradients"):
        os.mkdir("/tmp/gradients")

    for f in glob.glob(os.path.join(input, '*')):

        I = imread(f).astype('int16')
        bn = os.path.basename(f)
        dst_path = os.path.join(output_dir, bn)
        imwrite(dst_path, I.astype('uint8'))

        for it in range(100):

            gradient, prediction = nn.get_gradient(dst_path, 0, return_prediction=True)
            gradient = gradient.reshape((gradient.shape[1:]))
            prediction = prediction.reshape((2,))[1]

            gradient = np.round(gradient).astype('int16')
            gradient -= gradient % 4

            # gradient[gradient>16] = 16
            # gradient[gradient<-16] = -16

            click.echo(it, ":", f, "prediction:", prediction, ":", np.sum(gradient >= 4), np.sum(gradient <= -4))
            if prediction < 0.5:
                click.echo("ok")
                break

            if np.sum(gradient >= 4) == 0 and np.sum(gradient <= -4) == 0:
                click.echo("cannot improve")
                break

            I = imread(dst_path).astype('int16')
            # I[gradient>=4] += gradient[gradient>=4]
            # I[gradient<=-4] -= gradient[gradient<=-4]
            I[gradient >= 4] += 4
            I[gradient <= -4] -= 4
            # I[gradient>=1] -= 1
            # I[gradient<=-1] += 1
            I[I < 0] = 0
            I[I > 255] = 255
            imwrite(dst_path, I.astype('uint8'))

    sys.exit(0)
