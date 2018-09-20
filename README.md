# Pedestrian Benchmark Framework (PBF)

This repository contains a software framework designed for experimentation with different classical and state-of-the-art pedestrian detection algorithms and benchamrk datasets.

The code is written in Matlab and contains modified scripts based on the original code published by the authors of the algorithms. There are utility functions based on the [Piotr's toolbox](https://pdollar.github.io/toolbox/) allowing one to extract and manipulate data and evaluate the performance of the algorithms.

### Dependencies
* [Piotr's toolbox](https://pdollar.github.io/toolbox/)
* Caltech [evaluation/labeling](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

Download and copy the above files to `utilities` folder.

### Table of contents
* [Datasets](#datasets)
* [Algorithms](#algorithms)
* [Notes](#notes)
* [Instructions](#instructions)
	* [Extracting data](#ext_data)
	* [Training](#training)
	* [Testing](#testing)
	* [Evaluation](#evaluation)
* [Citation](#citation)
* [Disclaimer](#disclaim)


<a name="datasets"></a>
## Datasets
PBF works with 8 pedestrian detection datasets including [JAAD]( http://data.nvision2.eecs.yorku.ca/JAAD_dataset/), [CityPersons](https://www.cityscapes-dataset.com/), [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/), [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark), [Inria](http://pascal.inrialpes.fr/data/human/), [ETH](https://data.vision.ee.ethz.ch/cvl/aess/dataset/), [TUD-Brussels](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/people-detection-pose-estimation-and-tracking/multi-cue-onboard-pedestrian-detection/), and [Daimler](http://www.gavrila.net/Research/Pedestrian_Detection/Daimler_Pedestrian_Benchmark_D/Daimler_Mono_Ped__Detection_Be/daimler_mono_ped__detection_be.html).

The API uses the standardized versions of ETH, Inria, TUD-Brussels and Daimler in `.seq` and `.vbb` format which can be downloaded from [here](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/). For CityPersons, PBF uses the annotations in `.mat` format from [here](https://bitbucket.org/shanshanzhang/citypersons). For other datasets, download the data from the correspoding links.

The datasets should be placed in a common folder, as follows:

```
<main_folder>
		/<dataset>
			/<annotations_folder>
			/<videos_or_images_folder>
```
For datasets Caltech, ETH, Inria, TUD-Brussels and Daimler copy the content of `annotations.zip` to `annotations` folder and copy `set##.tar` to `videos/set##` folders. In the case of JAAD dataset, copy `vbb_full` and `vbb_part` folders to `annotations` folder and all `.seq` files to `video/set00` folder.

Copy the annotations of CityPersons to `annotations` folder and keep the images folder name as is. For KITTI dataset, just copy and extract the data and annotation folders as they are to the dataset folder.

<a name="algorithms"></a>
## Algorithms

PBF currently implements 10 classical and state-of-the-art pedestrian detection algorithms including [ACF](https://github.com/pdollar/toolbox), [CCF](https://github.com/byangderek/CCF), [Checkerboards](https://bitbucket.org/shanshanzhang/code_filteredchannelfeatures/downloads/), [DeepPed](https://github.com/DenisTome/DeepPed), [LDCF++](https://eshed1.github.io/), [Faster-rcnn](https://github.com/ShaoqingRen/faster_rcnn), [MS-CNN](https://github.com/zhaoweicai/mscnn), [RPN+BF](https://github.com/zhangliliang/RPN_BF), [SDS-RCNN](https://github.com/garrickbrazil/SDS-RCNN), [SPP](https://github.com/chhshen/pedestrian-detection).

For more detail regarding the selection of parameters for algorithms consult with the corresponding publications.

 <a name="notes"></a>
### Notes
To switch between the models with dependency on *Caffe*, Matlab environment has to be restarted each time.

#### ccf

The algorithm is tested using cuda 8.

For compatibality with new changes in caffe do the following changes:

In `cnnPyramid.m:` Comment out `line 13`, `%addpath(['path_to_caffe_codes' '/matlab/caffe']);` and add parameter  `test` to `line 113` as in `caffe('init', cnn.model_def, cnn.model_file, 'test');`.

Comment out `line 114`  `% caffe('set_phase_test');`.

Download the pretrained model *VGG_ILSVRC_16_layers.caffemodel* as instructed by the authors and copy to `data/CaffeNets`.

#### checkerboards

The code is tested with *Checkerboards_CVPR15_codebase* version of the code.

#### deepped

Only the detection code is available for this algorithm.

The code is tested with cuda 7.

#### ldcf++

This algorithm depends on Piotr's toolbox. Make sure the toolbox in `utilities` folder is in Matlab's path.

#### faster rcnn

The code is tested with cuda 7.

For the purpose of pedestrian detection replace the models in `/models` with `/jaad/models_fcnn_ped`.

#### ms-cnn

The code is tested with cuda 8.

Download the pretrained VGG weights as intructed by the authors and place in a folder called. `ms-cnn/pretrained`

#### sds-rcnn

Downlowad the pretrained model as instructed by the authors and place in a folder called `sds-rcnn/pretrained`.

#### spp

Only the detection code is available.

The code relies on optical flow, therefore the performance is optimal with sequential datasets such as JAAD and Caltech (with small skips).

<a name="instructions"></a>
## Instructions
To use the PBF whether for training and testing or data extraction, simply use `experimentScript.m` script.

<a name="ext_data"></a>
### Extracting data
PFB can be used to extract data for training/testing purposes with certain characteristics, e.g. scale of the images, type of annotations, or which portion of the data to use. For example the following example extracts partial occlusion annotations from *mix* subset of JAAD with skip of 10, 30, 30 for training, testinng and validation data in 0.5 scale.

```
dataInfo.dataSet = 'jaad';
dataInfo.anotationType = 'part';  
dataInfo.jaadSubDatasetDir = 'data_indices';
dataInfo.jaadSubDataset = 'mix';
dataInfo.skipTrain = 10;
dataInfo.skipTest = 30;
dataInfo.skipVal = 30;
dataInfo.scale = 0.5;
dataInfo.justDataExtract = 1;

```
Note that setting parameter `dataInfo.justDataExtract` to 1 make the script to only extract the data.

The extracted data is saved under `pedestrian_benchmark/data/<dataset_name>/<data_subset>`. The training data folder is appended with the value of skip, e.g. in the above example the folder will be `jaad_mix_scale_0.5/train10/`. This allows the user to have multiple trining data with different sizes. For testing and validation data, the folder name appears only as `test` and `val`, so if new changes in testing or validation data is needed, the corresponding data folders should be deleted manually.

<a name="training"></a>
### Training
For training set up dataset's parameters as desired (see above) and set `opts.trOrDet` to either 0 (only training) or 1 (training followed by testing). If the dataset with desired parameters exists in `/data` folder, the algorithm starts training, and if not, the data first will be extracted followed by training.

Next, the algorithm parameters for training should be setup accordingly. The following example sets up `ms-cnn` algorithm to be trained using bounding boxes in the range of `[50 inf]`, ignoring the bounding boxes with occlusion tag, and input training images of size `720x1280`.

```
modelSetup = setAlgorithmParams('ms-cnn', 'modelName', 'ms-cnn-example', 'hRng', [50 inf], 'ignOcc', 1, ...
							'scales', 720, 'max_size', 1280);
```
For more detail on how to set algorithms parameters check the comments in `setAlgorithmParams.m`.

<a name="testing"></a>
### Testing

The data set up for testing is similar to training. Simply specify what dataset and with which parameters are desired to run. To perform testing, set `opts.trOrDet` to either 1 or 2 (only testing). The following example runs testing for `ms-cnn` algorithm:

```
dataInfo.testImgsRange = []; % test on all testing data
opts.trOrDet = 2; % only testing
modelSetup = setAlgorithmParams('ms-cnn', 'modelName', 'ms-cnn-example',
							'test_scales', 720, 'test_max_size', 1280);
```
Note that if the `opts.trOrDet = 1` and model name specified under `'modelName'` already exists, the training stage is skipped.

The detection results will be saved under `detection_results/<algorithm_name>/<model_name>/<dataset_name>/<some_name>_dets.txt`. Note that if the detection file already exists, i.e. the chosen model has been tested on a given dataset, testing stage will skip. If another round of testing is needed, set `'reapply'` parameter in `setAlgorithmParams` to 1.

<a name="evaluation"></a>
### Evaluation
To perform evaluation after testing set `opts.doEvaluation` to true. Similar to algorithms, evaluation parameters can be set using `setEvaluationParams` function. The following example performs evalution by fixing the aspect ratio of ground truth boxes to `0.41`, horizontal limit to `[50 inf]`, visibility range to `[0.65 1]` with overlap threshold of `0.5` and saves the miss rate results as a text file.

```
evalSetup = setEvaluationParams('hRng',[50 inf], 'vRng', [.65 1], 'threshold', 0.5, 'saveMR', 1);
```
The result of evaluation is reported in terms of MR<sub>2</sub>, MR<sub>4</sub> and mean IOU.

<a name="citation"></a>
### Citation
If as part of your research, you use any of the algorithms or datasets in PBF, cite the corresponding publications as instructed by the authors. If you use PBF framework for experimentation and benchmarking, please cite our paper:

```
@inproceedings{rasouli2018role,
  title={It’s Not All About Size: On the Role of Data Properties in Pedestrian Detection},
  author={Rasouli, Amir and Kotseruba, Iuliia and Tsotsos, John K},
  booktitle={ECCVW},
  year={2018}
}
```
and [Piotr's toolbox](https://pdollar.github.io/toolbox/) for dependencies.

<a name='disclaim'></a>
### Disclaimer
This is a work in progress with the aim of easing up future experimentation and benchmarking in the field of pedestrian detection. If you have a published work with a  code written in Matlab, or a large scale dataset that want to be included in this framwork, please contact aras@eecs.yorku.ca.
