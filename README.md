# Shading-aware Multi-view Stereo 
![Travis CI](https://travis-ci.org/flanggut/smvs.svg?branch=master)

*developing...*

This repository contains an implementation of 'Shading-aware Multi-view Stereo' [1] [[pdf]](http://www.gcc.tu-darmstadt.de/media/gcc/papers/Langguth-2016-SMV.pdf). The framework itself is based on 'A New Variational Framework for Multiview Surface Reconstruction' [2].

### How to compile

The code only depends on [MVE](https://github.com/simonfuhrmann/mve), libjpeg, libpng, and libtiff. Compilation is supported on Unix and macOS systems:

	git clone https://github.com/simonfuhrmann/mve.git
	git clone https://github.com/flanggut/smvs.git
	make -C mve
	make -C smvs

### How to use

Before you start please have a look at the general reconstruction pipeline in the [MVE Wiki](https://github.com/simonfuhrmann/mve/wiki/MVE-Users-Guide#the-reconstruction-pipeline). This project is intended as an alternative multi-view stereo step and replaces the `dmrecon` and `scene2pset` applications with the `smvsrecon`. The complete pipeline is therefore:

	makescene -i <image-dir> <scene-dir>
	sfmrecon <scene-dir>
	smvsrecon <scene-dir>
	fssrecon <scene-dir>/smvs-[B,S].ply <scene-dir>/smvs-surface.ply
	meshclean -p10 <scene-dir>/smvs-surface.ply <scene-dir>/smvs-clean.ply

#### Arguments

If you run `smvsrecon` without any arguments it automatically chooses the most robust and reasonably fast setting. The images are rescaled to be on average around 2MP, the shading-based optimization is disabled and the optimization is running up to scale 2; the output will be `smvs-B.ply`. This behavior can be changed with various command line arguments.

* **Shading-based optimization**: The shading-based optimization as described in [1] can be activated with `-S`. This will output the point cloud `smvs-S.ply`. Keep in mind that the lighting model is limited to a low-dimensional global illumination based in spherical harmonics. As noted in the paper this model cannot handle complex scenes. Also try to supply linear images to the reconstruction pipeline that are not tone mapped or altered as this can also have very negative effects on the reconstruction. If you have simple JPGs with SRGB gamma correction you can remove it with the `--gamma-srgb` option.
* **Scale**: `smvs` has two scale options. `-s` affects the size of the input images and will downscale them by the respective power of 2 (this is analog to the same option of `dmrecon`, e.g. `-s2` would downscale to 1/4th of the original size). `-o` affects the scale of the optimization - the finest resolution of the bicubic patches will have the size of the respective power of 2 (e.g. `-o2` will optimize patches covering down to 4x4 pixels).
* **Semi-global Matching**: The basic algorithms described in [1] and [2] rely on a sparse initialization and surface expansion / shrinking to generate a good coverage of the scene. This is not guaranteed to work reliably for every scene. This application also contains an extension that initializes the surface with a coarse SGM to increase coverage. It is enabled by default and can be disabled using `--no-sgm`. Additionally, this allows for the reconstruction of scenes that have not been generated via structure-from-motion and do not contain an initial sparse point cloud. In this case the range for an initial depth sweep has to be supplied via `--sgm-range`.

For more details please also have a look at the usage output of the application. Note that some of the features are still WIP.

###References
[1] **Shading-aware Mult-view Stereo** - *Fabian Langguth, Kalyan Sunkavalli, Sunil Hadap, Michael Goesele* - ECCV 2016

	@inproceedings{langguth-2016-smvs,
	  title = {Shading-aware Multi-view Stereo},
	  author = {F. Langguth and K. Sunkavalli and S. Hadap and M. Goesele},
	  booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
	  year = {2016}
	}

[2] **A New Variational Framework for Multiview Surface Reconstruction** - *Ben Semerjian* - ECCV 2014

	@inproceedings{semerjian-2014-varsurf,
	  title = {A New Variational Framework for Multiview Surface Reconstruction},
	  author = {B. Semerjian},
	  booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
	  year = {2014}
	}
