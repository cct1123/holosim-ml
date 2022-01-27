[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# holosim-ml 📡
```holosim-ml``` is a library for beam simulation and analysis of radio holography data from complex optical systems. This code uses machine learning to efficiently determine the position of hundreds of mirror adjusters on multiple mirrors with few micron accuracy. We apply this approach to the example of the Simons Observatory 6 m telescope.

For example, ```holosim-ml``` simulates the far-field beam pattern, with mirror surface error root-mean-squares of 0 µm, 20 µm, 35 µm, and 50 µm.  The side-lobes around the central beam increase as RMS of panel errors increases.  This figure was produced with the ```Window Function.ipynb``` notebook, at 150 GHz.

<p align="center">
     <img src="far_fields.png" width="80%" />
</p>

## Dependencies
* Python>=3.6
* numpy, scipy, matplotlib, sklearn

or simply run 
```
$ pip install -r requirements.txt
```
## Installation
```
$ git clone https://github.com/cct1123/holosim-ml/
```

## Contributions
If you have write access to this repository, please:
* create a new branch
* push your changes to that branch
* merge or rebase to get in sync with main
* submit a pull request on github
* If you do not have write access, create a fork of this repository and proceed as described above. For more details, see Contributing.

## 
Email: chesmore@uchicago.edu for .sav files.