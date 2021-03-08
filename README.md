# geoprofile
Interactive tool to extract and plot profile along geospatial raster transect.

### Features
- interactive gui to select points along transect
- samples input raster
- plots transect profile

### Examples
See [notebooks](./examples/) for processing examples.

### Installation
```
$ git clone https://https://github.com/GlacioHack/geoprofile.git
$ cd ./geoprofile
$ conda env create -f environment_ubuntu_18.04_date_2021-03-07.yml
$ conda activate geoprofile
$ pip install -e .
```
To make this kernel selectable  

```
$ conda activate geoprofile
$ python -m ipykernel install --user --name geoprofile
```