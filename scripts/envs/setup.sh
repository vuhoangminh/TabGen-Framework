wget "https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh"
bash Anaconda3-2023.09-0-Linux-x86_64.sh -b

conda create --solver=libmamba -y -n custom -c rapidsai -c conda-forge -c nvidia cudf=23.10 cuml=23.10 python=3.9 cuda-version=11.2 jupyterlab dash
conda activate custom

# Install all the dependencies
pip install ctgan
pip install imgui==1.3.0 glfw==2.2.0 pyopengl==3.1.5 imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 click requests psutil
pip install scipy monai
pip install -U albumentations
pip install nilearn
pip install SimpleITK
pip install comet-ml
pip install opacus
pip install matplotlib
pip install seaborn
pip install xgboost==1.7.6
pip install autoflake
pip install termcolor
pip install hyperopt
pip install sdv==1.8.0
pip install wandb
pip uninstall -y torch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install scikit_posthocs
pip install tabulate
pip install researchpy
