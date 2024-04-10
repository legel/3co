# iNeRF: Neural Radiance Fields With iOS Estimated Camera Properties & Depth

## Environment

```sh
conda env create -f environment.yml
```

```sh
pip install open3d
pip install pytorch3d

```

## Get Data

#### Apple iPhone Pro: Pillow Dataset
```shell
wget https://x3co.s3.amazonaws.com/pillow_small.tar.gz
wget https://x3co.s3.amazonaws.com/pillow_large.tar.gz

tar -xzvf pillow_small.tar.gz
tar -xzvf pillow_large.tar.gz

mv pillow_small data/pillow_small
mv pillow_large data/pillow_large
```

## Train Model

```shell
python learn.py
```
