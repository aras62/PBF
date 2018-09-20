function bbsNm = run_ldcfplus(dataInfo, modelSetup)

%%%%%%%%%%%%%%%%%%%%%%%%%%% WELCOME %%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% We make minor modifications to Piotr Dollar's Toolbox
%%% (https://github.com/pdollar/toolbox)
%%% Based on the paper "To Boost or Not to Boost? On the Limits of Boosted Trees for Object Detection",
%%% by E. Ohn-Bar and M. Trivedi, International Conference on Pattern Recognition, 2016
%%% The resulting detectors achieve ~18 miss rate (ACF++) and ~15 miss rate
%%% (LDCF++). Ultimately, this was a study of dataset properties and
%%% classifier limitations.

%%% Main experimental options in this script:
% New or old annotations on Caltech Pedestrians
% LDCF or ACF

%%% Final results on Caltech can be found in 'results_Caltech' folder

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%SETUP
dataSet = dataInfo.dataSet;
dataDir = dataInfo.dataDir;
testDir = dataInfo.testDir ;

testImgsRange = dataInfo.testImgsRange;
cascCal = modelSetup.cascCal;
reapply = modelSetup.reapply;
resultsPath = dataInfo.resultsPath;
modelName = modelSetup.modelName;
modelPath = fullfile('ldcf+',modelSetup.modelPath,'/');
if ~exist(modelPath,'dir'), mkdir(modelPath); end;
trOrDet = modelSetup.trOrDet;

%%
octupSET = modelSetup.octupSet;
opts = acfTrain;
opts.modelDs= modelSetup.modelDs.*(2^octupSET);
opts.modelDsPad=modelSetup.modelDsPad.*(2^octupSET);

%DEFAULTS
opts.pPyramid.pChns.pColor.smooth=0;
opts.nWeak=modelSetup.nWeak;
opts.pBoost.discrete=0; opts.pPyramid.pChns.pGradHist.softBin=1;
opts.pPyramid.nOctUp=octupSET;
opts.pPyramid.minDs = opts.modelDs;
opts.pPyramid.pad = ceil((opts.modelDsPad-opts.modelDs)/opts.pPyramid.pChns.shrink/2)*opts.pPyramid.pChns.shrink;
opts.seed = 0;

pLoad = getTags(dataSet);
%If doing flip augmentation, should use %pLoad={'lbls',{'person'},'ilbls',{'people'},'squarify',[]};
hRng = [opts.modelDs(1)./2^opts.pPyramid.nOctUp inf];
vRng = modelSetup.vRng;
xRng = modelSetup.xRng;
yRng = modelSetup.yRng;
ignOcc = modelSetup.ignOcc;
squarify = modelSetup.squarify;

pLoad = [pLoad 'ignOcc', ignOcc, 'squarify', squarify];
if(~isempty(yRng)),  pLoad = [pLoad 'yRng', yRng]; end
if(~isempty(xRng)),  pLoad = [pLoad 'xRng', xRng]; end
if(~isempty(vRng)),  pLoad = [pLoad 'vRng', vRng]; end
if(~isempty(hRng)),  pLoad = [pLoad 'hRng', hRng]; end
    
opts.pLoad = pLoad; %[pLoad 'hRng',[opts.modelDs(1)./2^opts.pPyramid.nOctUp inf], 'vRng',[1 1] ];
opts.winsSave = 0;
opts.pPyramid.pChns.shrink= 4; %2^(octupSET+1);

%SET NAME
opts.name=  [modelPath modelName];

%OUR CHANGES
opts.pBoost.pTree.maxDepth=6;
opts.pBoost.pTree.fracFtrs=1; %1/8; %1/16; %Depending on how much time you have for training
opts.nNeg=100000;
opts.nAccNeg=2*opts.nNeg;


%JITTER POSITIVE SAMPLES.
%May benefit from more, up to you. sclsRange = [1.11:0.01:1.15] or sclsRange = [1.05:0.01:1.15]
%helps ACF a bit
sclsRange = 1.1;
sclsArray = [1 1];
for j=sclsRange, sclsArray = [sclsArray; j 1; 1 j; j j]; end

opts.pJitter= struct('flip',0,'scls',sclsArray);
opts.posGtDir = dataInfo.trainAnnotDir ;
opts.posImgDir = dataInfo.trainDir;

%OPTION 2: LDCF84 OR ACF
bldcf = modelSetup.useLDCF;
if(bldcf),opts.filters = [2 4];
 modelName = [modelName '_ldcf'];
 opts.name = [modelPath modelName];
end

if trOrDet < 2
acfTrain_modified_jaad(opts);
bbsNm = [];
end;
%%
%Testing
if trOrDet > 0
    disp('Begin testing...')
pModify=struct('cascThr',-1,'cascCal',cascCal);

bbsNm = acfTest_modified_jaad('name',modelName, 'modelPath',modelPath, 'imgDir',testDir,...
    'resultsPath', resultsPath,'pModify',pModify,'reapply',reapply, ...
    'dataset',dataSet, 'testImgsRange', testImgsRange, 'scale', dataInfo.scale);
end
end
%%
