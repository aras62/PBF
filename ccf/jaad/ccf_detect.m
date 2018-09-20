function bbsNm = ccf_detect(dataInfo,modelSetup)

% setup dataset info
dataset = dataInfo.dataSet; testDir = dataInfo.testDir ;
testImgsRange = dataInfo.testImgsRange; reapply = modelSetup.reapply;
resultsPath = dataInfo.resultsPath; name = modelSetup.modelName;
useCF = modelSetup.useCF;
% load detector
modelPath = fullfile('ccf', modelSetup.modelPath, '/');
ds = cell(1,1);
fileExt = '';

if useCF, fileExt = '_cf'; end;

scalestr = num2str(dataInfo.scale);
dotIdx = strfind(scalestr,'.');
if ~isempty(dotIdx), scalestr(dotIdx)=''; end;

if ~isempty(testImgsRange)
    bbsNm=[resultsPath 'ccf/' name fileExt '/'  dataset '/' num2str(testImgsRange(1)) '-' num2str(testImgsRange(2)) '_s' scalestr '_Dets.txt'];
else bbsNm=[resultsPath 'ccf/' name fileExt '/'  dataset '/all' '_s' scalestr '_Dets.txt']; end
if(reapply && exist(bbsNm,'file')), delete(bbsNm); end
if(~reapply && exist(bbsNm,'file')), fprintf('the detection file %s already exists. \n',bbsNm); return;end;

nm=[modelPath name fileExt '.mat']; 

t=exist(nm,'file');
if(~t),fprintf('Detector named %s does not exists.\n', nm)
bbsNm = []; return; end

% use the CCF detector (set to 1 to use the CCF+CF detector)
d = load(nm); ds{1} = d.detector;

% initialize caffe parameters

model_def = modelSetup.modelDef;
model_file = modelSetup.modelFile;
cnn = struct('model_def',model_def,...
             'model_file',model_file,...
             'device',modelSetup.gpuId,...
             'meanPix', modelSetup.meanPix);

opts = struct('input_size',modelSetup.input_size,'stride',modelSetup.stride,'pad',modelSetup.pad,...
    'minDs', modelSetup.minDs,'nPerOct',modelSetup.nPerOct,'nOctUp',...
    modelSetup.nOctUp, 'nApprox', modelSetup.nApprox,...
    'lambda',0.2666,'imresize', modelSetup.imresize ,'imflip',modelSetup.imflip,...
    'addCf',useCF,'savePyrd', modelSetup.savePyr);

% set imageset
imgNms = bbGt('getFiles',{testDir});
if ~isempty(testImgsRange), imgNms = imgNms(testImgsRange(1):testImgsRange(2));end

% run detection
allBBs = cnnDetect(imgNms, ds, opts, cnn);

results = [];
for i= 1: length(allBBs)
    idx = cat(2,repmat(i,[size(allBBs{i},1) 1]),allBBs{i});
    results = cat(1,results,idx);
end

d=fileparts(bbsNm); if(~isempty(d)&&~exist(d,'dir')), mkdir(d);end;
dlmwrite(bbsNm,results);


end