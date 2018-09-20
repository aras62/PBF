function results = runAlgorithmTrainTest(opts, dataInfo, evalSetup, modelSetup)

algorithmName = modelSetup.algorithmName; 
doEvaluation = opts.doEvaluation;
trOrDet = opts.trOrDet; 
modelSetup.trOrDet = trOrDet;

% the name of the folders for jaad dataset 
dataSetName = dataInfo.dataSet;
annot_extension = '/annotations';
if strcmpi(dataSetName,'jaad')
    dataSetName = [dataSetName '_' dataInfo.jaadSubDataset];
    annot_extension = [annot_extension '_' dataInfo.anotationType];
end;

%% Generate two folders for spp for optical flow
if strcmpi(algorithmName,'spp')
    if strcmpi(dataInfo.dataSet,'inria')
        fprintf('%s dataset is not applicable to the spp method.\n',dataInfo.dataSet);return;end
    if dataInfo.skipTrain < 2
        fprintf('The minimum skipTrain allowed for spp is %d.\n',2);dataInfo.skipTrain = 2;end
    if dataInfo.skipTest < 2
        fprintf('The minimum skipTest allowed for spp is %d.\n',2);dataInfo.skipTest = 2;end
    dataSetName = [dataSetName '_spp'];
end

%% Change the name of the dataset folder if scale is not 1
if (dataInfo.scale == 1), dataInfo.dataDir = ['data/' dataSetName '/'];
else dataInfo.dataDir = ['data/' dataSetName '_scale_' num2str(dataInfo.scale) '/'];end

%% Setup directories and extract data
dataInfo.testDir = [dataInfo.dataDir 'test/' 'images']; % Path to test images
dataInfo.testDir1 = [dataInfo.dataDir 'test/' 'images1'];% Path to second set of test images (spp only)
dataInfo.valDir = [dataInfo.dataDir 'val/' 'images']; % Path to validation images
dataInfo.testAnnotDir = [dataInfo.dataDir 'test' annot_extension]; % Path to test annotations
dataInfo.valAnnotDir  = [dataInfo.dataDir 'val' annot_extension]; % Path to validation annotations

 
% Extract data from dataset folders
[dataInfo.type, numTestImgs] = generateData(dataInfo);
if dataInfo.justDataExtract, return; end;

% Checks if the range of test images is valid
if modelSetup.trOrDet > 0
    if isempty(dataInfo.testImgsRange) , disp('All test images are used.')
    elseif dataInfo.testImgsRange(1) > dataInfo.testImgsRange(2) || dataInfo.testImgsRange(1) < 1 || dataInfo.testImgsRange(2) <1
        disp('The indicies for test images are invalid. All test images are used.'); dataInfo.testImgsRange = [];
    elseif dataInfo.testImgsRange(2) > numTestImgs
        fprintf('The index exceeds the maximum number of test images.\n There are %d test images available.\n All test images are used.\n',numTestImgs);
        dataInfo.testImgsRange = [];
    else fprintf('%d test images are used with indicies %d - %d.\n',...
            (dataInfo.testImgsRange(2)-dataInfo.testImgsRange(1)+1), dataInfo.testImgsRange(1), dataInfo.testImgsRange(2));end;
end

dataInfo.trainAnnotDir = [dataInfo.dataDir  dataInfo.type annot_extension];% Path to train annotations
dataInfo.trainDir = [dataInfo.dataDir dataInfo.type '/images/'];% Path to train images

algorithmNameLower = lower(algorithmName);
fprintf('\n\nStarting algorithm %s . \n', algorithmName);
switch(algorithmNameLower)
    case 'acf' % HoG, LDCF, ACF
        p = genpath('acf/'); addpath(p);
        bbsNm =   run_acf(dataInfo,modelSetup);
        rmpath(p);
    case 'ccf' % CCF and CCF + CF
        p = genpath('ccf/'); addpath(p);
        bbsNm = run_ccf(dataInfo,modelSetup);
        rmpath(p);
    case 'checkerboards'
        p = genpath('checkerboards/'); addpath(p);
        bbsNm = run_checkerboards(dataInfo,modelSetup);
        rmpath(p);
    case 'deepped'
        p = genpath('deepped/');addpath(p);
        bbsNm = deepped_detect(dataInfo,modelSetup);
        rmpath(p);
    case 'frcnn' % VGG16 ZF
        p = genpath('frcnn/jaad/');addpath(p);
        bbsNm = run_frcnn(dataInfo, modelSetup);
    case 'ldcf+' % LDCF++, ACF++
        p = genpath('ldcf+/'); addpath(p);
        bbsNm =   run_ldcfplus(dataInfo,modelSetup);
        rmpath(p);
    case 'ms-cnn'
        p = genpath('ms-cnn/'); addpath(p);
        bbsNm = run_mscnn(dataInfo, modelSetup);
        rmpath(p);
    case 'rpn-bf'
        p = genpath('rpn-bf/jaad/'); addpath(p);
        bbsNm = run_rpnbf(dataInfo, modelSetup);
        rmpath(p);
    case 'sds-rcnn'
        p = genpath('sds-rcnn/'); addpath(p);
       % modelSetup.modelPath = [filesPath 'models/'];
        bbsNm = run_sds(dataInfo,modelSetup);
        rmpath(p);
    case 'spp'
        p = genpath('spp/'); addpath(p);
        bbsNm = run_spp(dataInfo,modelSetup);
        rmpath(p);
    case 'evalonly'
        fprintf('Only running evaluation!\n');
        bbsNm = modelSetup.modelName;
    otherwise
        error('%s is not a valid method\n',methodName);
end
pause(1)
if (~isempty(bbsNm) && doEvaluation && trOrDet > 0)
    filesPath = 'utilities/'; p = genpath(filesPath); addpath(p);
    results = evaluateResults(bbsNm, dataInfo,evalSetup);
    rmpath(p);
else
   results = []; 
end;
end
