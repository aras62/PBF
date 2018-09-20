%% Available algorithms
% acf: P. Dollár, et al. "Fast feature pyramids for object detection." PAMI
%      2014.
% ccf: B. Yang, et al. "Convolutional channel features." ICCV 2015.
% checkerboards: S. Zhang, R. Benenson, and B. Schiele. "Filtered channel 
%                features for pedestrian detection." CVPR. 2015.
% deepped: D. Tomè, et al. "Deep convolutional neural networks for
%          pedestrian detection." Signal processing 2016.
% faster-rcnn: S. Ren, et al. "Faster r-cnn: Towards real-time object detection
%              with region proposal networks." NIPS 2015.
% ldcf+: E. Ohn-Bar,and M.Trivedi. "To boost or not to boost? on the limits
%        of boosted trees for object detection." ICPR 2016.
% ms-cnn: Z. Cai, et al. "A unified multi-scale deep convolutional neural
%          network for fast object detection." ECCV 2016.
% rpn_bf: Zhang, Liliang, et al. "Is faster r-cnn doing well for pedestrian 
%         detection?."ECCV 2016.
% sds-rcnn: G. Brazil, X. Yin, and X. Liu. "Illuminating Pedestrians via 
%           Simultaneous Detection & Segmentation." ICCV 2017.
% spp: Paisitkriangkrai, Sakrapee, Chunhua Shen, and Anton Van Den Hengel. 
%      "Strengthening the effectiveness of pedestrian detection with spatially pooled features."
%      ECCV 2014.

% Notes: Checkerboards does not work with jaad original scale
%        spp and deepped only has test code using the original caltech trained model
%        ccf requires a very large memory for training  ~128


%% Available datasets:
% jaad: A. Rasouli, I. Kotseruba, J. K. Tsotsos. "Are They Going to Cross? A Benchmark Dataset and Baseline for Pedestrian Crosswalk Behavior." ICCVW 2017. 
% citypersons: S. Zhang, R. Benenson, and B. Schiele. "Citypersons: A diverse dataset for pedestrian detection." CVPR  2017.
% caltech: P. Dollár, et al. "Pedestrian detection: A benchmark." Computer Vision and Pattern Recognition, CVPR  2009.
% kitti: A. Geiger, P. Lenz, and R. Urtasun. "Are we ready for autonomous driving? the kitti vision benchmark suite." CVPR 2012.
% inria: N. Dalal, and B. Triggs. "Histograms of oriented gradients for human detection."  CVPR 2005. 
% eth: A. Ess, et al. "A mobile vision system for robust multi-person tracking.", CVPR 2008.
% tudbrussels: C. Wojek, S. Walk, B. Schiele, "Multi-Cue Onboard Pedestrian Detection" CVPR 2009
% daimler: M. Enzweiler and D. M. Gavrila. "Monocular Pedestrian Detection: Survey and Experiments" PAMI 2009.

% Notes: Jaad and CityPersons dataset contain train/test/val subsets
%        Caltech and Inria contain only train/test. A val set is generate from a
%        subset of test data
%        Kitti and CityPersons do not contain annotations for test set.
%        Tudbrussels, daimler and eth only have test set

%% Parameters
% dataInfo                      -store options related to dataset
%         .dataset              -the name of the dataset:  jaad, caltech,
%                                tudbrussels, daimler (greyscale), eth, 
%                                inria, citypersons         
%         .anotationType        -only for kitti and jaad
%                                Jaad: part(pedestrians occluded 25% or 
%                                more) or full occlusion( pedestrians 
%                                occluded 75% or more)
%                                Kitti: part:  Partly occluded or full: Difficuly to see
%         .truncation           -only for kitti: defines maximum allowable truncation
%                                0.15 : min
%                                0.30 : med
%                                0.50 : max
%                                1 : all
%         .jaadSubDatasetDir    -only for jaad: the folder containing jaad subsets
%         .jaadSubDataset       -only for jaad: the name of the jaad subset
%                               -cameraGoPro, cameraGarmin, weatherClear, weatherCloudy, weatherClear_Cloudy, mix
%                               
%         .skipTrain            -skip frames for different sets. For defaul values check dbinfo_jaad
%         .skipTest
%         .skipVal
%         .scale                -scale of the data
%         .testImgsRange        -select what subset of test data to be used
%                                for evaluation. set [] to use all data 
%         .resultsPath          -path to save detection results
%         .justDataExtract      -if set to 1, the code only extract annotations and images
%         .debugRescale         -displays file names as being rescaled
%         .dataPath             -path to the folder of datasets
%                                the data format is of form 
%                                <datasets_folder>/<dataset>/
%                                for JAAD -> JAAD/videos/set00/video_####
%                                            JAAD/annotations/part/set00
%                                            JAAD/annotations/full/set00
% ops
%         .doEvaluation         -true/false Performs evaluation
%         .trOrDet              -0 Only train the model, 1 perfrom train and test, 2 Only test using a pretrained model

filesPath = 'utilities/';
p = genpath(filesPath);
addpath(p);

dataInfo.dataSet = 'jaad';
dataInfo.anotationType = 'part';  
dataInfo.truncation = 0.30;
dataInfo.jaadSubDatasetDir = 'data_indices';
dataInfo.jaadSubDataset = 'mix'; 
dataInfo.skipTrain = 100; 
dataInfo.skipTest = 100; 
dataInfo.skipVal = 100; 
dataInfo.scale = 1;
dataInfo.testImgsRange = [10,30]; 
dataInfo.resultsPath = 'detection_results/';
dataInfo.justDataExtract = 0; 
dataInfo.debugRescale = 0; 
dataInfo.dataPath='/media/aras/Storage/datasets/pedestrian_datasets/';

opts.doEvaluation = true; 
opts.trOrDet = 2; 

%% MODEL names according to dataset
algorithm_name = 'acf';
model_name ='acf_jaad_mix_part_scale0.5_skip10';
% Set the parameters for the algorithm (for more details check
% setAlgorithmParams)
modelSetup = setAlgorithmParams(algorithm_name, 'modelName', model_name);
% Set evaluation parameters
evalSetup = setEvaluationParams('ignOcc',1, 'vRng', [.75 1]);
results = runAlgorithmTrainTest(opts, dataInfo, evalSetup, modelSetup);

