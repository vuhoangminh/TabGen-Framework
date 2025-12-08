import os
import ntpath
from engine.utils.print_utils import print_separator
import glob


def get_project_dir(path, project_name):
    paths = path.split(project_name)
    return paths[0] + project_name


def split_dos_path_into_components(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)

            break

    folders.reverse()
    return folders


def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]


def make_dir(dir):
    if not os.path.exists(dir):
        print_separator()
        print("making dir", dir)
        os.makedirs(dir)


def get_path_yaml(
    dataset,
    max_patients,
    aug,
    generated_patients=0,
    gan_model="",
    model_name="unet",
):
    if gan_model == "":
        return f"{dataset}-num-{max_patients}_train-0.60_val-0.20_test-0.20_{model_name}_{aug}"
    else:
        return f"{dataset}-num-{max_patients}_train-0.60_val-0.20_test-0.20_{model_name}_{aug}_gen-{generated_patients}_gan-{gan_model}"


def get_modality(path, ext=".nii.gz"):
    filename = get_filename(path)
    modality = filename.replace(ext, "")
    return modality


def get_hyperopt_path(project_name, database_path="database", folder="optimization"):
    save_path = database_path + f"/{folder}"
    make_dir(save_path)
    file_name = project_name + ".hyperopt"
    return os.path.join(save_path, file_name)


def get_folder(args):
    loss_version = args.loss_version

    if args.is_test:
        pre = "test-"
    else:
        pre = ""

    if args.is_drop_id:
        suf = ""
    else:
        suf = "-id_1"

    dataset = args.dataset

    if loss_version == 0:
        if args.private:
            return f"{pre}{dataset}-{args.arch}-lv_{args.loss_version}-dp-bs_{args.batch_size}-epochs_{args.epochs}-ed_{args.embedding_dim}-dd_{args.discriminator_dim}-gd_{args.generator_dim}-sigma_{args.dp_sigma}-clip_{args.dp_weight_clip}-losscorr_{args.is_loss_corr:.2e}-lossdwp_{args.is_loss_dwp:.2e}-condvec_{args.is_condvec}{suf}"
        else:
            return f"{pre}{dataset}-{args.arch}-lv_{args.loss_version}-bs_{args.batch_size}-epochs_{args.epochs}-ed_{args.embedding_dim}-dd_{args.discriminator_dim}-gd_{args.generator_dim}-losscorr_{args.is_loss_corr:.2e}-lossdwp_{args.is_loss_dwp:.2e}-condvec_{args.is_condvec}{suf}"
    elif loss_version == 1:
        if args.private:
            return f"{pre}{dataset}-{args.arch}-lv_{args.loss_version}-dp-bs_{args.batch_size}-epochs_{args.epochs}-ed_{args.embedding_dim}-dd_{args.discriminator_dim}-gd_{args.generator_dim}-sigma_{args.dp_sigma}-clip_{args.dp_weight_clip}-losscorr_{args.is_loss_corr:.2e}-lossdwp_{args.is_loss_dwp:.2e}-condvec_{args.is_condvec}{suf}"
        else:
            return f"{pre}{dataset}-{args.arch}-lv_{args.loss_version}-bs_{args.batch_size}-epochs_{args.epochs}-ed_{args.embedding_dim}-dd_{args.discriminator_dim}-gd_{args.generator_dim}-losscorr_{args.is_loss_corr:.2e}-lossdwp_{args.is_loss_dwp:.2e}-condvec_{args.is_condvec}{suf}"
    elif loss_version == 2 or loss_version == 4 or loss_version == 5:
        if args.private:
            return f"{pre}{dataset}-{args.arch}-lv_{args.loss_version}-dp-bs_{args.batch_size}-epochs_{args.epochs}-ed_{args.embedding_dim}-dd_{args.discriminator_dim}-gd_{args.generator_dim}-sigma_{args.dp_sigma}-clip_{args.dp_weight_clip}-moment_{args.n_moment_loss_dwp}-losscorcorr_{args.is_loss_corr:.2e}-lossdis_{args.is_loss_dwp:.2e}-condvec_{args.is_condvec}{suf}"
        else:
            return f"{pre}{dataset}-{args.arch}-lv_{args.loss_version}-bs_{args.batch_size}-epochs_{args.epochs}-ed_{args.embedding_dim}-dd_{args.discriminator_dim}-gd_{args.generator_dim}-moment_{args.n_moment_loss_dwp}-losscorcorr_{args.is_loss_corr:.2e}-lossdis_{args.is_loss_dwp:.2e}-condvec_{args.is_condvec}{suf}"
    elif loss_version == 3:
        if args.private:
            return f"{pre}{dataset}-{args.arch}-lv_{args.loss_version}-dp-bs_{args.batch_size}-epochs_{args.epochs}-ed_{args.embedding_dim}-dd_{args.discriminator_dim}-gd_{args.generator_dim}-sigma_{args.dp_sigma}-clip_{args.dp_weight_clip}-moment_{args.n_moment_loss_dwp}-losscorcorr_{args.is_loss_corr:.2e}-normalizedlossdis_{args.is_loss_dwp:.2e}-condvec_{args.is_condvec}{suf}"
        else:
            return f"{pre}{dataset}-{args.arch}-lv_{args.loss_version}-bs_{args.batch_size}-epochs_{args.epochs}-ed_{args.embedding_dim}-dd_{args.discriminator_dim}-gd_{args.generator_dim}-moment_{args.n_moment_loss_dwp}-losscorcorr_{args.is_loss_corr:.2e}-normalizedlossdis_{args.is_loss_dwp:.2e}-condvec_{args.is_condvec}{suf}"


def get_folder_technical_paper(args):
    loss_version = args.loss_version

    if args.is_test:
        pre = "test-"
    else:
        pre = ""

    """Builds a filename based on given arguments and loss version.

    Args:
        args: Namespace containing arguments.
        pre: Prefix for the filename.
        suf: Suffix for the filename.
        loss_version: Loss version identifier.

    Returns:
        The constructed filename as a string.
    """

    if args.row_number is not None:
        base_filename = f"{pre}{args.dataset}-rownum_{args.row_number}-{args.arch}-lv_{loss_version}-bs_{args.batch_size}-epochs_{args.epochs}"
    else:
        base_filename = f"{pre}{args.dataset}-{args.arch}-lv_{loss_version}-bs_{args.batch_size}-epochs_{args.epochs}"

    if args.arch in ["ctgan", "dpcgans", "copulagan"]:
        if args.private:
            base_filename += f"-dp_1-dps_{args.dp_sigma}-dpwc_{args.dp_weight_clip}"
        base_filename += f"-ed_{args.embedding_dim}-dd_{args.discriminator_dim}-gd_{args.generator_dim}-glr_{args.generator_lr:.2e}"
    elif args.arch in ["tvae"]:
        base_filename += f"-ed_{args.embedding_dim}-cd_{args.compress_dims}-dd_{args.decompress_dims}-l2_{args.l2scale:.2e}"
    elif args.arch in ["ctab"]:
        base_filename += f"-nc_{args.n_class_layer}-cd_{args.class_dim}-rd_{args.random_dim}-nc_{args.num_channels}-tr_{args.test_ratio}"
    elif args.arch in ["tabddpm"]:
        if args.model_type == "mlp":
            base_filename += f"-df_{args.d_first}-dm_{args.d_middle}-dl_{args.d_last}-nl_{args.n_layers}-lr_{args.lr:.2e}-model_{args.model_type}"
        else:
            base_filename += f"-dm_{args.d_main}-dh_{args.d_hidden}-df_{args.dropout_first:.1f}-ds_{args.dropout_second:.1f}-nl_{args.n_blocks}-lr_{args.lr:.2e}-model_{args.model_type}"
    elif args.arch in ["tabsyn"]:
        base_filename += f"-dt_{args.dim_t}-lr_{args.lr:.2e}-fac_{args.factor:.2e}"
    else:
        raise NotImplementedError

    if loss_version in [2, 3, 4, 5]:
        base_filename += f"-moment_{args.n_moment_loss_dwp}"

    if loss_version == 1:
        base_filename += (
            f"-losscorr_{args.is_loss_corr:.2e}-lossdwp_{args.is_loss_dwp:.2e}"
        )
    elif loss_version == 2 or loss_version == 4 or loss_version == 5:
        base_filename += (
            f"-losscorcorr_{args.is_loss_corr:.2e}-lossdis_{args.is_loss_dwp:.2e}"
        )
    elif loss_version == 3:
        base_filename += f"-losscorcorr_{args.is_loss_corr:.2e}-normalizedlossdis_{args.is_loss_dwp:.2e}"

    base_filename += f"-condvec_{args.is_condvec}"

    return base_filename


def find_non_largest_csv_files(folder_path):
    """Finds all CSV files in a folder whose integer part is not the largest.

    Args:
      folder_path: The path to the folder containing the CSV files.

    Returns:
      A tuple of non-largest CSV files and the largest CSV file.
    """

    csv_files = glob.glob(os.path.join(folder_path, "fake_*.csv"))
    csv_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if len(csv_files) >= 1:
        largest_csv_file = csv_files[-1]
        non_largest_csv_files = csv_files[:-1]
        return non_largest_csv_files, largest_csv_file
    else:
        return [], []


def main():
    output_dir = "/mnt/sda2/3DUnetCNN_BRATS/projects/pros/database/prediction/pros_2018_is-256-256-128_crop-0_bias-0_denoise-0_norm-11_hist-0_ps-128-128-128_segnet3d_crf-0_loss-dice_xent_aug-1_model/validation_case_956"


if __name__ == "__main__":
    main()
