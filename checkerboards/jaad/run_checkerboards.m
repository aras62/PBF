function bbsNm= run_checkerboards(dataInfo,modelSetup)
% Demo for aggregate channel features object detector on Caltech dataset.
%% See also acfReadme.m
%% Piotr's Computer Vision Matlab Toolbox      Version 3.40
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

%% extract training and testing images and ground truth

dataSet = dataInfo.dataSet; 
dataDir = dataInfo.dataDir;
testDir = dataInfo.testDir ;
type = dataInfo.type;
testImgsRange = dataInfo.testImgsRange;
modelName = modelSetup.modelName;
resultsPath = dataInfo.resultsPath;
modelPath = 'checkerboards/output/';
if ~exist(modelPath,'dir'), mkdir(modelPath); end;

trOrDet = modelSetup.trOrDet;
hRng = modelSetup.hRng;
vRng = modelSetup.vRng;
xRng = modelSetup.xRng;
yRng = modelSetup.yRng;
ignOcc = modelSetup.ignOcc;
modelDs = modelSetup.modelDs;
modelDsPad = modelSetup.modelDsPad;
nWeak = modelSetup.nWeak;
reapply = modelSetup.reapply;
squarify = modelSetup.squarify;

pLoad = getTags(dataSet);
pLoad = [pLoad 'ignOcc', ignOcc, 'squarify', squarify];
if(~isempty(yRng)),  pLoad = [pLoad 'yRng', yRng]; end
if(~isempty(xRng)),  pLoad = [pLoad 'xRng', xRng]; end
if(~isempty(vRng)),  pLoad = [pLoad 'vRng', vRng]; end
if(~isempty(hRng)),  pLoad = [pLoad 'hRng', hRng]; end


opts=acfTrain_check();

opts.posGtDir = dataInfo.trainAnnotDir;
opts.posImgDir = dataInfo.trainDir;
opts.name = [modelPath modelName];
opts.modelDs = modelDs; 
opts.modelDsPad = modelDsPad;
opts.pPyramid.smooth = 0;
opts.pPyramid.pChns.pColor.smooth = 0; 
opts.pJitter=struct('flip',1);
opts.pBoost.pTree.fracFtrs = 1;
opts.nWeak = nWeak;
opts.pLoad = pLoad; %[pLoad 'hRng',hRng, 'vRng',vRng];
opts.pPyramid.pChns.shrink = 6; 
opts.stride = 6;
opts.pPyramid.nApprox = 0;
opts.cascThr = -1; 
opts.pPyramid.pChns.cbin = [2,5,5];
opts.pPyramid.pChns.pGradHist.softBin = 1;
opts.pPyramid.pChns.pGradHist.clipHog = Inf;
opts.nNeg = 10000;
opts.nAccNeg = 50000;
opts.nPerNeg = 25;
opts.pPyramid.pChns.pGradHist.binSize = opts.pPyramid.pChns.shrink;
opts.pPyramid.pChns.NNRadius = 1;
opts.pPyramid.nOctUp = 1; 
opts.pBoost.pTree.maxDepth = 4;
opts.pBoost.discrete = 0;


%% train detector (see acfTrain)
if trOrDet < 2
%% train detector (see acfTrain)
acfTrain_check(opts);
bbsNm = [];
end;
if trOrDet > 0

%% test detector and plot roc (see acfTest)
disp('Begin testing...')
   bbsNm=acfTest_check('name',modelName, 'modelPath',modelPath, 'imgDir',testDir,...
    'resultsPath', resultsPath,'reapply',reapply, ...
    'dataset',dataSet, 'testImgsRange', testImgsRange, 'scale', dataInfo.scale);
end
end