# iNeRF: Neural Radiance Fields With iOS Estimated Camera Properties & Depth

## Environment
We provide a `environment.yml` file to set up a `conda` environment:

```sh
git clone https://github.com/ActiveVisionLab/nerfmm.git
cd nerfmm
conda env create -f environment.yml
```

```sh
pip install open3d
pip install pytorch3d

```

## Get Data

#### Apple iPhone Pro: Pillow Dataset
```shell
wget https://www.robots.ox.ac.uk/~ryan/nerfmm2021/nerfmm_release_data.tar.gz
tar -xzvf path/to/the/tar.gz
```

## Train Model

```shell
python learn_3d_model.py
```
