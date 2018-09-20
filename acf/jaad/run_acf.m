function bbsNm = run_acf(dataInfo,modelSetup)
% Demo for aggregate channel features object detector on Caltech dataset.
%
% See also acfReadme.m
%
% Piotr's Computer Vision Matlab Toolbox      Version 3.40
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

%% extract training and testing images and ground truth
% clear all ;

dataSet = dataInfo.dataSet; 
testDir = dataInfo.testDir ;
resultsPath = dataInfo.resultsPath;
modelName = modelSetup.modelName;
modelPath = fullfile('acf', modelSetup.modelPath,'/');
if ~exist(modelPath,'dir'), mkdir(modelPath); end;

trOrDet = modelSetup.trOrDet;
testImgsRange = dataInfo.testImgsRange;
useLDCF = modelSetup.useLDCF;
hRng = modelSetup.hRng;
vRng = modelSetup.vRng;
xRng = modelSetup.xRng;
yRng = modelSetup.yRng;
ignOcc = modelSetup.ignOcc;
squarify = modelSetup.squarify;
modelDs = modelSetup.modelDs;
modelDsPad = modelSetup.modelDsPad;
nWeak = modelSetup.nWeak;
cascCal = modelSetup.cascCal;
reapply = modelSetup.reapply;
calSetup = modelSetup.calSetup; 

%% set up opts for training detector (see acfTrain)
opts=acfTrain_jaad();

if ~modelSetup.channels(1),opts.pPyramid.pChns.pColor.enabled = 0; disp('color features not used');end
if ~modelSetup.channels(2),opts.pPyramid.pChns.pGradMag.enabled = 0;disp('gradMag features not used');end
if ~modelSetup.channels(3),opts.pPyramid.pChns.pGradHist.enabled = 0;disp('HoG features not used');end
    
pLoad = getTags(dataSet);
pLoad = [pLoad 'ignOcc', ignOcc, 'squarify', squarify];
if(~isempty(yRng)),  pLoad = [pLoad 'yRng', yRng]; end
if(~isempty(xRng)),  pLoad = [pLoad 'xRng', xRng]; end
if(~isempty(vRng)),  pLoad = [pLoad 'vRng', vRng]; end
if(~isempty(hRng)),  pLoad = [pLoad 'hRng', hRng]; end


if calSetup
    opts.pBoost.pTree.maxDepth = 5;
    opts.pPyramid.pChns.pGradHist.softBin = 1;
    opts.pBoost.discrete = 0;
    opts.pPyramid.pChns.pColor.smooth = 0;
    opts.pPyramid.pChns.shrink=2;
end
opts.modelDs=modelDs;
opts.modelDsPad= modelDsPad;
opts.nWeak= nWeak;
opts.pLoad = pLoad;
opts.name = [modelPath modelName];
opts.pBoost.pTree.fracFtrs = 1/16;
opts.nNeg = 25000;
opts.nAccNeg = 50000;
opts.pJitter = struct('flip',1);
opts.posGtDir = dataInfo.trainAnnotDir;
opts.posImgDir = dataInfo.trainDir;
pModify=struct('cascThr',-1,'cascCal', cascCal);


%% optionally switch to LDCF version of detector (see acfTrain)
if(useLDCF)
    opts.filters = [5 4];
    modelName = [modelName '_ldcf'];
    opts.name = [modelPath modelName];
    if strcmpi(dataSet,'inria')
        opts.pJitter = struct('flip',1,'nTrn',3,'mTrn',1);
        opts.pBoost.pTree.maxDepth = 3;
        opts.pBoost.discrete = 0;
        opts.seed = 2;
        opts.pPyramid.pChns.shrink = 2;
    end
end



if trOrDet < 2
%% train detector (see acfTrain)
acfTrain_jaad(opts);
bbsNm = [];
end;

if trOrDet > 0
    disp('Begin testing...')
bbsNm =acfTest_jaad('name',modelName, 'modelPath',modelPath, 'imgDir',testDir,...
    'resultsPath', resultsPath,'pModify',pModify,'reapply',reapply, ...
    'dataset',dataSet, 'testImgsRange', testImgsRange, 'scale',dataInfo.scale);
end;

end
