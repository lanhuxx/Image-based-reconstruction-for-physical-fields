![](./impact_case.jpg)

# Data-set-of-a-3D-impact-case
The warehouse of the training samples of the EReConNN

The 3D Computer Aided Design (CAD) model of the impact problem is presented. The impact body is a cuboid whose material is Al alloy 6061-T6 as shown in Table 1, and defined with an initial velocity v0 along the negative direction of z-axis. Furthermore, a point mass of 300kg is coupled in the center of the other side of the collision surface, which is marked by using a red cross.

![](./The_CAD_model_of_the_impact_case.jpg)

The dataset of the impact process contains 5,633 images. The pixels of each one is `480x960x3`.

## How it is generated
The impact case is simulated by the Abaqus CAE, and the iterations are designed as 5,633. Then run the script from the `code/` folder. `_abaqus8_.guiLog`, the images can be saved iteration by iteration.

You can generate your own dataset just modify the job name in the `code/` folder. `_abaqus8_.guiLog`:

```
o1 = session.openOdb(name='C:/Temp/Job-6.odb')
```
and then run it.

## How to use
The images are read by ***cv2.imread***, and the read function is ***ImageReader*** in the `code/` folder. `image_reader.py`. The input to the function contains file name, input path, label path, picture format and size. As for the ***size***, in order to improve the universality of the algorithm, each image is re-sized to one of ***size*size*3*** by using the Bilinear Interpolation, so the ***size*** can be adjusted according to demand and its default value is 64.

## Citation policy
Please cite our work if you write a scientific paper using this code and/or dataset.

```latex
@article{title={Image-based reconstruction for strong-nonlinear transient problems by using an Enhanced ReConNN},
  author={Yu Lia, Hu Wanga, Wenquan Shuai},
  journal={...},
  year={2019}
}
```
