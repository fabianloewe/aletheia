import glob
import os
import random
import shutil
import sys
from pathlib import Path

import click
import numpy as np

from aletheialib.utils import download_e4s


@click.group()
def ml():
    """Machine learnging based steganalysis."""
    pass


@ml.command()
@click.option('--cover-dir', help="Directory containing cover images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--stego-dir', help="Directory containing stego images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--output-dir', help="Output directory where actors will be created", required=True,
              type=click.Path(file_okay=False, resolve_path=True, path_type=Path))
@click.option('--valid', help="Number of images for the validation set", required=True, type=int)
@click.option('--test', help="Number of images for the testing set", required=True, type=int)
@click.option('--seed', help="Seed for reproducible results", required=True, type=int)
def split_sets(cover_dir, stego_dir, output_dir, valid, test, seed):
    """Prepare sets for training and testing."""

    cover_files = np.array(sorted(cover_dir.glob('*')))
    stego_files = np.array(sorted(stego_dir.glob('*')))

    if len(cover_files) != len(stego_files):
        click.echo("ERROR: we expect the same number of cover and stego files")
        sys.exit(0)

    indices = list(range(len(cover_files)))
    random.seed(seed)
    random.shuffle(indices)

    valid_indices = indices[:valid // 2]
    test_C_indices = indices[valid // 2:valid // 2 + test // 2]
    test_S_indices = indices[valid // 2 + test // 2:valid // 2 + test]
    train_indices = indices[valid // 2 + test:]

    train_C_dir = os.path.join(output_dir, "train", "cover")
    train_S_dir = os.path.join(output_dir, "train", "stego")
    valid_C_dir = os.path.join(output_dir, "valid", "cover")
    valid_S_dir = os.path.join(output_dir, "valid", "stego")
    test_C_dir = os.path.join(output_dir, "test", "cover")
    test_S_dir = os.path.join(output_dir, "test", "stego")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    os.makedirs(train_C_dir, exist_ok=True)
    os.makedirs(train_S_dir, exist_ok=True)
    os.makedirs(valid_C_dir, exist_ok=True)
    os.makedirs(valid_S_dir, exist_ok=True)
    os.makedirs(test_C_dir, exist_ok=True)
    os.makedirs(test_S_dir, exist_ok=True)

    for f in cover_files[train_indices]:
        shutil.copy(f, train_C_dir)

    for f in stego_files[train_indices]:
        shutil.copy(f, train_S_dir)

    for f in cover_files[valid_indices]:
        shutil.copy(f, valid_C_dir)

    for f in stego_files[valid_indices]:
        shutil.copy(f, valid_S_dir)

    for f in cover_files[test_C_indices]:
        shutil.copy(f, test_C_dir)

    for f in stego_files[test_S_indices]:
        shutil.copy(f, test_S_dir)


@ml.command()
@click.option('--cover-dir', help="Directory containing cover images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--stego-dir', help="Directory containing stego images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--double-dir', help="Directory containing double stego images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--output-dir', help="Output directory where actors will be created", required=True,
              type=click.Path(file_okay=False, resolve_path=True, path_type=Path))
@click.option('--valid', help="Number of images for the validation set", required=True, type=int)
@click.option('--test', help="Number of images for the testing set", required=True, type=int)
@click.option('--seed', help="Seed for reproducible results", required=True, type=int)
def split_sets_dci(cover_dir, stego_dir, double_dir, output_dir, valid, test, seed):
    """Prepare sets for training and testing (DCI)."""

    cover_files = np.array(sorted(cover_dir.glob('*')))
    stego_files = np.array(sorted(stego_dir.glob('*')))
    double_files = np.array(sorted(double_dir.glob('*')))

    if len(cover_files) != len(stego_files) or len(stego_files) != len(double_files):
        click.echo("split-sets-dci error: we expect sets with the same number of images")
        sys.exit(0)

    indices = list(range(len(cover_files)))
    random.seed(seed)
    random.shuffle(indices)

    valid_indices = indices[:valid // 2]
    test_C_indices = indices[valid // 2:valid // 2 + test // 2]
    test_S_indices = indices[valid // 2 + test // 2:valid // 2 + test]
    train_indices = indices[valid // 2 + test:]

    A_train_C_dir = os.path.join(output_dir, "A_train", "cover")
    A_train_S_dir = os.path.join(output_dir, "A_train", "stego")
    A_valid_C_dir = os.path.join(output_dir, "A_valid", "cover")
    A_valid_S_dir = os.path.join(output_dir, "A_valid", "stego")
    A_test_C_dir = os.path.join(output_dir, "A_test", "cover")
    A_test_S_dir = os.path.join(output_dir, "A_test", "stego")
    B_train_S_dir = os.path.join(output_dir, "B_train", "stego")
    B_train_D_dir = os.path.join(output_dir, "B_train", "double")
    B_valid_S_dir = os.path.join(output_dir, "B_valid", "stego")
    B_valid_D_dir = os.path.join(output_dir, "B_valid", "double")
    B_test_S_dir = os.path.join(output_dir, "B_test", "stego")
    B_test_D_dir = os.path.join(output_dir, "B_test", "double")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    os.makedirs(A_train_C_dir, exist_ok=True)
    os.makedirs(A_train_S_dir, exist_ok=True)
    os.makedirs(A_valid_C_dir, exist_ok=True)
    os.makedirs(A_valid_S_dir, exist_ok=True)
    os.makedirs(A_test_C_dir, exist_ok=True)
    os.makedirs(A_test_S_dir, exist_ok=True)
    os.makedirs(B_train_S_dir, exist_ok=True)
    os.makedirs(B_train_D_dir, exist_ok=True)
    os.makedirs(B_valid_S_dir, exist_ok=True)
    os.makedirs(B_valid_D_dir, exist_ok=True)
    os.makedirs(B_test_S_dir, exist_ok=True)
    os.makedirs(B_test_D_dir, exist_ok=True)

    for f in cover_files[train_indices]:
        shutil.copy(f, A_train_C_dir)

    for f in stego_files[train_indices]:
        shutil.copy(f, A_train_S_dir)
        shutil.copy(f, B_train_S_dir)

    for f in double_files[train_indices]:
        shutil.copy(f, B_train_D_dir)

    for f in cover_files[valid_indices]:
        shutil.copy(f, A_valid_C_dir)

    for f in stego_files[valid_indices]:
        shutil.copy(f, A_valid_S_dir)
        shutil.copy(f, B_valid_S_dir)

    for f in double_files[valid_indices]:
        shutil.copy(f, B_valid_D_dir)

    for f in cover_files[test_C_indices]:
        shutil.copy(f, A_test_C_dir)

    for f in stego_files[test_S_indices]:
        shutil.copy(f, A_test_S_dir)

    for f in stego_files[test_C_indices]:
        shutil.copy(f, B_test_S_dir)

    for f in double_files[test_S_indices]:
        shutil.copy(f, B_test_D_dir)


@ml.command()
@click.option('--cover-dir', help="Directory containing cover images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--stego-dir', help="Directory containing stego images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--output-dir', help="Output directory where actors will be created", required=True,
              type=click.Path(file_okay=False, resolve_path=True, path_type=Path))
@click.option('--innocent', help="Number of innocent actors", required=True, type=int)
@click.option('--guilty', help="Number of guilty actors", required=True, type=int)
@click.option('--min', help="Minimum number of images for each actor", required=True, type=int)
@click.option('--max', help="Maximum number of images for each actor", required=True, type=int)
@click.option('--seed', help="Seed for reproducible results", required=True, type=int)
def create_actors(cover_dir, stego_dir, output_dir, innocent, guilty, min, max, seed):
    """Prepare actors for training and testing."""

    cover_files = sorted(cover_dir.glob('*'))
    stego_files = sorted(stego_dir.glob('*'))

    if len(cover_files) != len(stego_files):
        click.echo("ERROR: we expect the same number of cover and stego files");
        sys.exit(0)

    innocent_actors = os.path.join(output_dir, "innocent")
    guilty_actors = os.path.join(output_dir, "guilty")

    os.makedirs(innocent_actors, exist_ok=True)
    os.makedirs(guilty_actors, exist_ok=True)

    random.seed(seed)

    # Innocent actors
    for i in range(1, innocent + 1):
        num_actor_images = random.randint(min, max)
        actor_dir = os.path.join(innocent_actors, f"actor{i}")

        if os.path.isdir(actor_dir):
            click.echo(f"Innocent actor already exists actor{i}")
            continue

        if not os.path.isdir(actor_dir):
            os.mkdir(actor_dir)

        for j in range(1, num_actor_images + 1):
            # random image
            k = random.randint(0, len(cover_files) - 1)
            src_file = cover_files[k]
            n, ext = os.path.splitext(src_file)
            dst_file = os.path.join(innocent_actors, f"actor{i}/{j}{ext}")
            if os.path.exists(dst_file):
                click.echo(f"Already exists actor{i}/{j}{ext}")
            else:
                shutil.copy(src_file, dst_file)
                click.echo(f"Copy {src_file} actor{i}/{j}{ext}")

    # Guilty actors
    for i in range(1, guilty + 1):
        num_actor_images = random.randint(min, max)
        actor_dir = os.path.join(guilty_actors, f"actor{i}")

        if os.path.isdir(actor_dir):
            click.echo(f"Guilty actor already exists actor{i}")
            continue

        f = open(os.path.join(guilty_actors, f"actor{i}.txt"), 'w')

        if not os.path.isdir(actor_dir):
            os.mkdir(actor_dir)

        for j in range(1, num_actor_images + 1):

            # random image
            k = random.randint(0, len(cover_files) - 1)

            prob_stego = round(random.uniform(0.1, 1), 2)
            if random.random() < prob_stego:
                src_file = stego_files[k]
                n, ext = os.path.splitext(src_file)
                f.write(f'{j}{ext}, 1\n')
            else:
                src_file = cover_files[k]
                n, ext = os.path.splitext(src_file)
                f.write(f'{j}{ext}, 0\n')

            dst_file = os.path.join(guilty_actors, f"actor{i}/{j}{ext}")

            if os.path.exists(dst_file):
                click.echo(f"Already exists actor{i}/{j}{ext}")
            else:
                shutil.copy(src_file, dst_file)
                click.echo(f"Copy {src_file} actor{i}/{j}{ext}")

        f.close()


@ml.command()
@click.option('--trn-cover-dir', help="Directory containing training cover images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--trn-stego-dir', help="Directory containing training stego images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--val-cover-dir', help="Directory containing validation cover images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--val-stego-dir', help="Directory containing validation stego images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--model-name', help="A name for the model", required=True)
@click.option('--dev', help="Device: GPU Id or 'CPU'", default='CPU')
@click.option('--early-stopping', help="Early stopping iterations x1000", default=10)
@click.option('--batch', help="Batch size", default=16)
def effnetb0(trn_cover_dir, trn_stego_dir, val_cover_dir, val_stego_dir, model_name, dev, early_stopping, batch):
    """Train a model with EfficientNet B0."""

    if dev == "CPU":
        click.echo("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    trn_cover_files = sorted(trn_cover_dir.glob('*'))
    trn_stego_files = sorted(trn_stego_dir.glob('*'))
    val_cover_files = sorted(val_cover_dir.glob('*'))
    val_stego_files = sorted(val_stego_dir.glob('*'))

    click.echo("train:", len(trn_cover_files), "+", len(trn_stego_files))
    click.echo("valid:", len(val_cover_files), "+", len(val_stego_files))

    if (not len(trn_cover_files) or not len(trn_stego_files) or
            not len(val_cover_files) or not len(val_stego_files)):
        click.echo("ERROR: directory without files found")
        sys.exit(0)

    import aletheialib.models
    nn = aletheialib.models.NN("effnetb0", model_name=model_name, shape=(512, 512, 3))
    nn.train(trn_cover_files, trn_stego_files, batch,  # 36|40
             # nn = aletheialib.models.NN("effnetb0", model_name=model_name, shape=(32,32,3))
             # nn.train(trn_cover_files, trn_stego_files, 500, # 36|40
             val_cover_files, val_stego_files, 10,
             1000000, early_stopping)


@ml.command()
@click.option('--test-cover-dir', help="Directory containing test cover images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--test-stego-dir', help="Directory containing test stego images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--model-file', help="Path of the model", required=True)
@click.option('--dev', help="Device: GPU Id or 'CPU'", default='CPU')
def effnetb0_score(test_cover_dir, test_stego_dir, model_file, dev):
    """Score with EfficientNet B0."""

    import aletheialib.models

    if dev == "CPU":
        click.echo("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    cover_files = sorted(test_cover_dir.glob('*'))
    stego_files = sorted(test_stego_dir.glob('*'))

    nn = aletheialib.models.NN("effnetb0")
    nn.load_model(model_file)

    pred_cover = nn.predict(cover_files, 10)
    pred_stego = nn.predict(stego_files, 10)

    ok = np.sum(np.round(pred_cover) == 0) + np.sum(np.round(pred_stego) == 1)
    score = ok / (len(pred_cover) + len(pred_stego))

    click.echo("score:", score)


@ml.command()
@click.option('--test-dir', help="Directory containing test images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--model-file', help="Path of the model", required=True)
@click.option('--dev', help="Device: GPU Id or 'CPU'", default='CPU')
def effnetb0_predict(test_dir, model_file, dev):
    """Predict with EfficientNet B0."""

    import aletheialib.models

    if dev == "CPU":
        click.echo("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    nn = aletheialib.models.NN("effnetb0")
    nn.load_model(model_file)

    if os.path.isdir(test_dir):
        test_files = sorted(glob.glob(os.path.join(test_dir, '*')))
    else:
        test_files = [test_dir]

    test_files = nn.filter_images(test_files)
    if len(test_files) == 0:
        click.echo("ERROR: please provice valid files")
        sys.exit(0)

    pred = nn.predict(test_files, 10)

    for i in range(len(pred)):
        click.echo(test_files[i], round(pred[i], 3))


@ml.command()
@click.option('--A-test-cover-dir', help="Directory containing A-cover images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--A-test-stego-dir', help="Directory containing A-stego images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--B-test-stego-dir', help="Directory containing B-stego images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--B-test-double-dir', help="Directory containing B-double images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--A-model-file', help="Path of the A-model", required=True)
@click.option('--B-model-file', help="Path of the B-model", required=True)
@click.option('--dev', help="Device: GPU Id or 'CPU'", default='CPU')
def effnetb0_dci_score(A_test_cover_dir, A_test_stego_dir, B_test_stego_dir, B_test_double_dir,
                       A_model_file, B_model_file, dev):
    """DCI Score with EfficientNet B0."""

    import aletheialib.models
    from sklearn.metrics import accuracy_score

    if dev == "CPU":
        click.echo("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    A_cover_files = sorted(A_test_cover_dir.glob('*'))
    A_stego_files = sorted(A_test_stego_dir.glob('*'))
    B_stego_files = sorted(B_test_stego_dir.glob('*'))
    B_double_files = sorted(B_test_double_dir.glob('*'))

    A_nn = aletheialib.models.NN("effnetb0")
    A_nn.load_model(A_model_file)
    B_nn = aletheialib.models.NN("effnetb0")
    B_nn.load_model(B_model_file)

    A_files = A_cover_files + A_stego_files
    B_files = B_stego_files + B_double_files

    p_aa_ = A_nn.predict(A_files, 10)
    p_ab_ = A_nn.predict(B_files, 10)
    p_bb_ = B_nn.predict(B_files, 10)
    p_ba_ = B_nn.predict(A_files, 10)

    p_aa = np.round(p_aa_).astype('uint8')
    p_ab = np.round(p_ab_).astype('uint8')
    p_ba = np.round(p_ba_).astype('uint8')
    p_bb = np.round(p_bb_).astype('uint8')

    y_true = np.array([0] * len(A_cover_files) + [1] * len(A_stego_files))
    inc = ((p_aa != p_bb) | (p_ba != 0) | (p_ab != 1)).astype('uint8')
    inc1 = (p_aa != p_bb).astype('uint8')
    inc2 = ((p_ba != 0) | (p_ab != 1)).astype('uint8')
    inc2c = (p_ab != 1).astype('uint8')
    inc2s = (p_ba != 0).astype('uint8')
    C_ok = ((p_aa == 0) & (p_aa == y_true) & (inc == 0)).astype('uint8')
    S_ok = ((p_aa == 1) & (p_aa == y_true) & (inc == 0)).astype('uint8')

    tp = np.sum((p_aa == 1) & (p_aa == y_true))
    tn = np.sum((p_aa == 0) & (p_aa == y_true))
    fp = np.sum((p_aa == 1) & (p_aa != y_true))
    fn = np.sum((p_aa == 0) & (p_aa != y_true))
    click.echo(f"aa-confusion-matrix tp: tp={tp}, tn={tn}, fp={fp}, fn={fn}")

    tp = np.sum((p_aa == 1) & (p_aa == y_true) & (inc == 0))
    tn = np.sum((p_aa == 0) & (p_aa == y_true) & (inc == 0))
    fp = np.sum((p_aa == 1) & (p_aa != y_true) & (inc == 0))
    fn = np.sum((p_aa == 0) & (p_aa != y_true) & (inc == 0))
    click.echo(f"dci-confusion-matrix tp: tp={tp}, tn={tn}, fp={fp}, fn={fn}")

    click.echo("#inc:", np.sum(inc == 1), "#incF1:", np.sum(inc1 == 1), "#incF2:", np.sum(inc2 == 1),
          "#incF2C", np.sum(inc2c), "#incF2S:", np.sum(inc2s))
    click.echo("#no_inc:", len(A_files) - np.sum(inc == 1))
    click.echo("#C-ok:", np.sum(C_ok == 1))
    click.echo("#S-ok:", np.sum(S_ok == 1))
    click.echo("aa-score:", accuracy_score(y_true, p_aa))
    click.echo("bb-score:", accuracy_score(y_true, p_bb))
    click.echo("dci-score:", round(float(np.sum(C_ok == 1) + np.sum(S_ok == 1)) / (len(A_files) - np.sum(inc == 1)), 2))
    click.echo("--")
    click.echo("dci-prediction-score:", round(1 - float(np.sum(inc == 1)) / (2 * len(p_aa)), 3))


@ml.command()
@click.option('--A-test-dir', help="Directory containing A test images", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--B-test-dir', help="Directory containing B test images", required=True,
              type=click.Path(exists=True, resolve_path=True, path_type=Path))
@click.option('--A-model-file', help="Path of the A-model", required=True)
@click.option('--B-model-file', help="Path of the B-model", required=True)
@click.option('--dev', help="Device: GPU Id or 'CPU'", default='CPU')
def effnetb0_dci_predict(A_test_dir, B_test_dir, A_model_file, B_model_file, dev):
    """DCI Prediction with EfficientNet B0."""

    import aletheialib.models

    if len(sys.argv) < 7:
        dev_id = "CPU"
        click.echo("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[6]

    if dev_id == "CPU":
        click.echo("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    A_files = sorted(A_test_dir.glob('*'))
    B_files = sorted(B_test_dir.glob('*'))

    A_nn = aletheialib.models.NN("effnetb0")
    A_nn.load_model(A_model_file)
    B_nn = aletheialib.models.NN("effnetb0")
    B_nn.load_model(B_model_file)

    p_aa = A_nn.predict(A_files, 10)
    p_ab = A_nn.predict(B_files, 10)
    p_bb = B_nn.predict(B_files, 10)
    p_ba = B_nn.predict(A_files, 10)

    p_aa = np.round(p_aa).astype('uint8')
    p_ab = np.round(p_ab).astype('uint8')
    p_ba = np.round(p_ba).astype('uint8')
    p_bb = np.round(p_bb).astype('uint8')

    inc = ((p_aa != p_bb) | (p_ba != 0) | (p_ab != 1)).astype('uint8')
    inc1 = (p_aa != p_bb).astype('uint8')
    inc2 = ((p_ba != 0) | (p_ab != 1)).astype('uint8')
    inc2c = (p_ab != 1).astype('uint8')
    inc2s = (p_ba != 0).astype('uint8')

    for i in range(len(p_aa)):
        r = ""
        if inc[i]:
            r = "INC"
        else:
            r = round(p_aa[i], 3)
        click.echo(A_files[i], r)

    click.echo("#inc:", np.sum(inc == 1), "#incF1:", np.sum(inc1 == 1), "#incF2:", np.sum(inc2 == 1),
          "#incF2C", np.sum(inc2c), "#incF2S:", np.sum(inc2s))
    click.echo("#no_inc:", len(A_files) - np.sum(inc == 1))
    click.echo("--")
    click.echo("dci-prediction-score:", round(1 - float(np.sum(inc == 1)) / (2 * len(p_aa)), 3))


@ml.command()
@click.option('--cover-fea', help="Cover feature file", required=True)
@click.option('--stego-fea', help="Stego feature file", required=True)
@click.option('--model-file', help="Path of the model", required=True,
              type=click.Path(dir_okay=False, resolve_path=True, path_type=Path))
def e4s(cover_fea, stego_fea, model_file):
    """Train Ensemble Classifiers for Steganalysis."""

    download_e4s()

    import aletheialib.models
    from sklearn.model_selection import train_test_split
    import pandas
    import numpy

    X_cover = pandas.read_csv(cover_fea, delimiter=" ").values
    X_stego = pandas.read_csv(stego_fea, delimiter=" ").values
    # X_cover=numpy.loadtxt(cover_fea)
    # X_stego=numpy.loadtxt(stego_fea)

    X = numpy.vstack((X_cover, X_stego))
    y = numpy.hstack(([0] * len(X_cover), [1] * len(X_stego)))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10)

    clf = aletheialib.models.Ensemble4Stego()
    clf.fit(X_train, y_train)
    val_score = clf.score(X_val, y_val)

    clf.save(model_file)
    click.echo("Validation score:", val_score)


@ml.command()
@click.option('--model-file', help="Path of the model", required=True,
              type=click.Path(dir_okay=False, resolve_path=True, path_type=Path))
@click.option('--feature-extractor', help="Feature extractor", required=True)
@click.option('--path', help="Image or directory", required=True,
              type=click.Path(exists=True, resolve_path=True, path_type=Path))
def e4s_predict(model_file, feature_extractor, path):
    """Predict using EC."""

    download_e4s()

    import aletheialib.models
    import aletheialib.utils

    files = []
    if os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                path = os.path.abspath(os.path.join(dirpath, f))
                if not aletheialib.utils.is_valid_image(path):
                    click.echo("Warning, please provide a valid image: ", f)
                else:
                    files.append(path)
    else:
        files = [path]

    clf = aletheialib.models.Ensemble4Stego()
    clf.load(model_file)
    for f in files:
        # TODO: make it multithread
        X = aletheialib.feaext.extractor_fn(feature_extractor)(f)
        X = X.reshape((1, X.shape[0]))
        p = clf.predict(X)
        # click.echo(p)
        if p[0] == 0:
            click.echo(os.path.basename(f), "Cover")
        else:
            click.echo(os.path.basename(f), "Stego")


A_models = {}
B_models = {}


def _load_model(nn, model_name):
    # Get the directory where the models are installed
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, os.pardir, os.pardir, 'aletheia-models')

    model_path = os.path.join(dir_path, model_name + ".h5")
    if not os.path.isfile(model_path):
        click.echo(f"ERROR: Model file not found: {model_path}\n")
        sys.exit(-1)
    nn.load_model(model_path, quiet=False)
    return nn


def _actor(image_dir, output_file, dev_id="cpu"):
    files = glob.glob(os.path.join(image_dir, '*'))

    if not os.path.isdir(image_dir):
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

    if ext == '.jpg':
        simulators = ["outguess-sim", "steghide-sim", "nsf5-color-sim", "j-uniward-color-sim"]
    else:
        simulators = ["steganogan-sim", "lsbm-sim", "hill-color-sim", "s-uniward-color-sim"]

    import gc
    import aletheialib.utils
    import aletheialib.stegosim
    import aletheialib.models
    import numpy as np
    from tensorflow.python.framework import ops

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    A_nn = aletheialib.models.NN("effnetb0")
    A_files = files

    output = ""
    for simulator in simulators:
        if simulator in ["steganogan-sim", "outguess-sim", "steghide-sim", "nsf5-color-sim", "j-uniward-color-sim"]:
            embed_fn_saving = True
        else:
            embed_fn_saving = False

        fn_sim = aletheialib.stegosim.embedding_fn(simulator)

        # Make some replacements to adapt the name of the method with the name
        # of the model file
        method = simulator
        method = method.replace("-sim", "")
        method = method.replace("-color", "")
        method = method.replace("j-uniward", "juniw")
        method = method.replace("s-uniward", "uniw")

        actor_dir = aletheialib.utils.absolute_path(os.path.dirname(A_files[0]))
        B_dir = os.path.join(actor_dir + "-B-cache", method.upper())

        if os.path.exists(B_dir):
            click.echo(f"Cache directory already exists:", B_dir)
        else:
            click.echo(f"Prepare embeddings:", B_dir)
            os.makedirs(B_dir, exist_ok=True)
            aletheialib.stegosim.embed_message(fn_sim, actor_dir, "0.10-0.50", B_dir,
                                               embed_fn_saving=embed_fn_saving, show_debug_info=False)

        B_nn = aletheialib.models.NN("effnetb0")
        B_files = glob.glob(os.path.join(B_dir, '*'))

        if method in A_models:
            ops.reset_default_graph()
            A_nn = A_models[method]
            gc.collect()
        else:
            A_nn = aletheialib.models.NN("effnetb0")
            A_nn = _load_model(A_nn, "effnetb0-A-alaska2-" + method)
            A_models[method] = A_nn

        if method in B_models:
            ops.reset_default_graph()
            B_nn = B_models[method]
            gc.collect()
        else:
            B_nn = aletheialib.models.NN("effnetb0")
            B_nn = _load_model(B_nn, "effnetb0-B-alaska2-" + method)
            B_models[method] = B_nn

        # Predictions for the DCI method
        _p_aa = A_nn.predict(A_files, 50)
        _p_ab = A_nn.predict(B_files, 50)
        _p_bb = B_nn.predict(B_files, 50)
        _p_ba = B_nn.predict(A_files, 50)

        p_aa = np.round(_p_aa).astype('uint8')
        p_ab = np.round(_p_ab).astype('uint8')
        p_bb = np.round(_p_bb).astype('uint8')
        p_ba = np.round(_p_ba).astype('uint8')

        # Inconsistencies
        inc = ((p_aa != p_bb) | (p_ba != 0) | (p_ab != 1)).astype('uint8')

        positives = 0
        for i in range(len(p_aa)):
            r = ""
            if inc[i]:
                r = str(round(_p_aa[i], 3)) + " (inc)"
            else:
                r = round(_p_aa[i], 3)
            # click.echo(A_files[i], "\t", r)
            if p_aa[i] == 1:
                positives += 1

        dci = round(1 - float(np.sum(inc == 1)) / (2 * len(p_aa)), 3)
        positives_perc = round((positives) / len(p_aa), 3)
        # click.echo(f"method={method}, DCI-pred={dci}, positives={positives_perc}")
        output += f"{dci}, {positives_perc}, "

    bn = os.path.basename(image_dir)
    with open(output_file, "a+") as myfile:
        myfile.write(bn + ", " + output[:-2] + "\n")


@ml.command()
@click.option('--actors-dir', help="Directory containing actors", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--output-file', help="Output file", required=True,
              type=click.Path(resolve_path=True, path_type=Path))
@click.option('--dev', help="Device: GPU Id or 'CPU'", default='CPU')
def actor_predict_fea(actors_dir, output_file, dev):
    """Predict with actors.

    Example:
      python3 aletheia.py actor-predict-fea actors/A1 data.csv
    """

    if output_file.exists():
        output_file.unlink()
    _actor(actors_dir, output_file, dev)


@ml.command()
@click.option('--actors-dir', help="Directory containing actors", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option('--output-file', help="Output file", required=True,
              type=click.Path(resolve_path=True, path_type=Path))
@click.option('--dev', help="Device: GPU Id or 'CPU'", default='CPU')
def actors_predict_fea(actors_dir, output_file, dev):
    """Predict with actors.

    Example:
        python3 aletheia.py actors-predict-fea actors/ data.csv
    """

    if output_file.exists():
        output_file.unlink()
    for path in actors_dir.glob('*'):
        if not os.path.isdir(path):
            continue
        _actor(path, output_file, dev)


@ml.command()
@click.argument('input')
@click.option('--payload', help="Payload")
@click.option('--fea-extract', help="Feature extractor", required=True)
@click.option('--images', help="Images directory", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
def ats(input, payload, fea_extract, images):
    """ATS.

    INPUT is either a custom command or a simulator name.

    Example:
        python3 aletheia.py ats hill-sim --payload 0.40 --fea-extract srm --images image_dir/
    """

    import aletheialib.stegosim as stegosim
    import aletheialib.feaext as feaext
    import aletheialib.utils as utils
    import tempfile
    import pandas
    import numpy

    embed_fn_saving = False

    if payload is not None:
        emb_sim = input
        feaextract = fea_extract
        A_dir = images
        fn_sim = stegosim.embedding_fn(emb_sim)
        fn_feaextract = feaext.extractor_fn(feaextract)
        if emb_sim in ["j-uniward-sim", "j-uniward-color-sim",
                       "ued-sim", "ued-color-sim", "ebs-sim", "ebs-color-sim",
                       "nsf5-sim", "nsf5-color-sim"]:
            embed_fn_saving = True
    else:
        click.echo("custom command")
        payload = sys.argv[2]  # uggly hack
        feaextract = fea_extract
        A_dir = images
        fn_sim = stegosim.custom
        embed_fn_saving = True
        fn_feaextract = feaext.extractor_fn(feaextract)

    B_dir = tempfile.mkdtemp()
    C_dir = tempfile.mkdtemp()
    stegosim.embed_message(fn_sim, A_dir, payload, B_dir, embed_fn_saving=embed_fn_saving)
    stegosim.embed_message(fn_sim, B_dir, payload, C_dir, embed_fn_saving=embed_fn_saving)

    fea_dir = tempfile.mkdtemp()
    A_fea = os.path.join(fea_dir, "A.fea")
    C_fea = os.path.join(fea_dir, "C.fea")
    feaext.extract_features(fn_feaextract, A_dir, A_fea)
    feaext.extract_features(fn_feaextract, C_dir, C_fea)

    A = pandas.read_csv(A_fea, delimiter=" ").values
    C = pandas.read_csv(C_fea, delimiter=" ").values

    X = numpy.vstack((A, C))
    y = numpy.hstack(([0] * len(A), [1] * len(C)))

    from aletheialib import models
    clf = models.Ensemble4Stego()
    clf.fit(X, y)

    files = []
    for dirpath, _, filenames in os.walk(B_dir):
        for f in filenames:
            path = os.path.abspath(os.path.join(dirpath, f))
            if not utils.is_valid_image(path):
                click.echo("Warning, this is not a valid image: ", f)
            else:
                files.append(path)

    for f in files:
        B = fn_feaextract(f)
        B = B.reshape((1, B.shape[0]))
        p = clf.predict(B)
        if p[0] == 0:
            click.echo(os.path.basename(f), "Cover")
        else:
            click.echo(os.path.basename(f), "Stego")

    shutil.rmtree(B_dir)
    shutil.rmtree(C_dir)
    shutil.rmtree(fea_dir)

# }}}
