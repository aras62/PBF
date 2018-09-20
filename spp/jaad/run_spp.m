function bbsNm = run_spp(dataInfo,modelSetup)

dataset = dataInfo.dataSet;
testDir = dataInfo.testDir ;
testDir1 = dataInfo.testDir1 ;
testImgsRange = dataInfo.testImgsRange;
reapply = modelSetup.reapply;

resultsPath = dataInfo.resultsPath;
modelName = modelSetup.modelName;

scalestr = num2str(dataInfo.scale);
dotIdx = strfind(scalestr,'.');
if ~isempty(dotIdx), scalestr(dotIdx)=''; end;

if ~isempty(testImgsRange)
    bbsNm = [resultsPath 'spp/' modelName '/'  dataset '/' num2str(testImgsRange(1)) '-' num2str(testImgsRange(2)) '_s' scalestr '_Dets.txt'];
else bbsNm = [resultsPath 'spp/' modelName '/'  dataset '/all' '_s' scalestr '_Dets.txt']; end

if(reapply && exist(bbsNm,'file')), delete(bbsNm); end
if(~reapply && exist(bbsNm,'file')), fprintf('the detection file %s already exists. \n',bbsNm); return;end;

files0 = bbGt('getFiles', {testDir}, 1);
files1 = bbGt('getFiles', {testDir1}, 1);

if ~isempty(testImgsRange)
    files0 = files0(testImgsRange(1):testImgsRange(2));
    files1 = files1(testImgsRange(1):testImgsRange(2));
end
res = cell(length(files0),1);
parfor i=1:length(files0)
    fprintf('Processing %s\n', files0{i});
    dat =  runDetection(files0{i},files1{i}, modelSetup);
    IfileID = i;
    dat = [repmat(IfileID,size(dat,1),1) dat]; %IfileID+1
    res{i} = dat;
end

res = cat(1,res{:});
if isempty(res), return; end
d=fileparts(bbsNm); if(~isempty(d)&&~exist(d,'dir')), mkdir(d);end;
dlmwrite(bbsNm,res);

function res = runDetection(TEST_IMAGE_0, TEST_IMAGE_1, setup)

MODEL_SIZE     = setup.modelDs;
IMAGE_SIZE     = setup.imageSize;
NUM_OCTAVE     = setup.octave ;
MODEL_FILENAME = 'model.mat';
STRIDE         = setup.stride;
BING_THRESH    = setup.bingThresh;
PED_THRESH     = setup.pedTresh;

% Load BING and pedestrain detector models
if ~exist(MODEL_FILENAME, 'file'), disp('model is not found'); res = 0; return;
else  model = load(MODEL_FILENAME); ped_model=model.ped_model; bing_model=model.bing_model; end


% Compute optical flow between two consecutive images
alpha = 0.01;
ratio = 0.8;
minWidth = 20;
nOuterFPIterations = 3;
nInnerFPIterations = 1;
nSORIterations = 20;
flowThreshold = 20;
para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

im0 = imread(TEST_IMAGE_0); im1 = imread(TEST_IMAGE_1);
if ~isequal([size(im0,1),size(im0,2)], IMAGE_SIZE),
    im0 = imresize(im0, IMAGE_SIZE);
    im1 = imresize(im1, IMAGE_SIZE);
end

[vx,vy,~] = Coarse2FineTwoFrames(im0, im1, para);
flow = cat(3,vx,vy);
flow=min(flowThreshold, flow); flow=max(-flowThreshold,flow);
flow=single(flow./flowThreshold); % flow image (single)
I   = im2single(im0);  % test image (single)
Iuint8 = im0;          % test image (uint8)


% Pre-compute BING masks at different scales
bing = evalBINGMex(Iuint8,bing_model); bing = bing(24:-1:1);

% get scales at which to compute features and list of real/approx scales
[scales,scaleshw]=getScales(NUM_OCTAVE,0,MODEL_SIZE,4,IMAGE_SIZE);
nScales=length(scales);

bbs = cell(nScales,1);

for i=1:nScales
    if i>length(bing), continue; end
    sc=scales(i); sc1=round(IMAGE_SIZE*sc/4); sc2=sc1*4;
    if size(I,1) ~= sc2(1) && size(I,2) ~= sc2(2)
        I1=imResampleMex(I,sc2(1),sc2(2),1); flow1=imResampleMex(flow,sc2(1),sc2(2),1);
    else
        I1=I; flow1=flow;
    end
    mask = zeros(sc1,'single');
    [h1,w1]=size(bing{i});
    if h1>sc1(1),h1=sc1(1); end, if w1>sc1(2),w1=sc1(2); end
    mask(1:h1,1:w1)=bing{i}(1:h1,1:w1);
    bb=detectPedMex(I1,flow1,mask,ped_model,STRIDE,PED_THRESH,BING_THRESH);
    if ~isempty(bb),
        bb(:,1) = (bb(:,1))./scaleshw(i,2);
        bb(:,2) = (bb(:,2))./scaleshw(i,1);
        bb(:,3:4) = bb(:,3:4)./sc;
        bbs{i}=bb;
    end
end

bbs=cat(1,bbs{:});
bbs=bbNms(bbs,'type','maxg','overlap',0.65,'ovrDnm','min');
res = bbs;
