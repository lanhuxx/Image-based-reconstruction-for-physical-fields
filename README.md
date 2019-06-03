![](./EReConNN.png)
# Image-based-reconstruction-for-physical-fields
With the improvement of the pattern recognition and feature extraction of Deep Neural Networks (DPNNs), image-based design and optimization have been widely used in multidisciplinary researches. Recently, a ***Reconstructive Neural Network (ReConNN)*** has been proposed to obtain an image-based model from an analysis-based model. However, it has difficulties handling nonlinear transient impact problems.
Therefore, an improved ReConNN, ***EReConNN***, method is proposed to address the mentioned weaknesses. Time-dependent ordered images can be generated. Furthermore, the improved method is successfully applied in impact simulation case and engineering experiment. Through the experiments, comparisons and analyses, the improved method is demonstrated to outperform the former one in terms of its accuracy, efficiency and costs.

***Coming soon ......***

![](./impact_case.jpg)
# Data-set-of-a-3D-impact-case
The warehouse of the training samples of the EReConNN

The 3D Computer Aided Design (CAD) model of the impact problem is presented. The impact body is a cuboid whose material is Al alloy 6061-T6 as shown in Table 1, and defined with an initial velocity v0 along the negative direction of z-axis. Furthermore, a point mass of 300kg is coupled in the center of the other side of the collision surface, which is marked by using a red cross.

![](./The_CAD_model_of_the_impact_case.jpg)

The dataset of the impact process contains 563 images. The pixels of each one is `480x960x3`. You can download the ***picture_480x960_563_samples.7z***, and a dataset with 5,633 samples will be upload subsequently.

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
@article{
  title={Image-based reconstruction for the impact problems by using DPNNs},
  author={Yu Lia, Hu Wanga, Wenquan Shuai, et al},
  journal={arXiv:1905.03229},
  year={2019}
}
```
