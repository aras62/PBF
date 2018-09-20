function bbsNm = deepped_detect(dataInfo,modelSetup)

% setup dataset info

% type = dataInfo.type;
testImgsRange = dataInfo.testImgsRange;
reapply = modelSetup.reapply;
dataset = dataInfo.dataSet;
resultsPath = dataInfo.resultsPath;
modelName = modelSetup.modelName;

% modelPath = modelSetup.modelPath;

if ~isempty(testImgsRange)
    bbsNm=[resultsPath 'deepped/' modelName '/'  dataset '/' num2str(testImgsRange(1)) '-' num2str(testImgsRange(2)) '_Dets.txt'];
else bbsNm=[resultsPath 'deepped/' modelName '/'  dataset '/all' '_Dets.txt']; end


if(reapply && exist(bbsNm,'file')), delete(bbsNm); end
if(~reapply && exist(bbsNm,'file')), fprintf('the detection file %s already exists. \n',bbsNm); return;end;

testDir = dataInfo.testDir;
listImages = bbGt('getFiles',{testDir});
if ~isempty(testImgsRange), listImages = listImages(testImgsRange(1):testImgsRange(2));end

% load and adjust the LDCF detector
load(fullfile(modelSetup.ldcfPath, modelName));
pModify = struct('cascThr',-1,'cascCal',.025);
detector = acfModify(detector,pModify);

% load the trained SVM
SVM = load(modelSetup.svmPath);
PersonW = SVM.W;
PersonB = SVM.b;
% load the trained svm of level 2
cl2 = load(modelSetup.svmlvl2Path);

%load the finetuned AlexNet
use_gpu = modelSetup.useGpu ;    %to change to zero if caffe compiled without CUDA support

assert(exist(modelSetup.alexnetPath, 'file') ~= 0);
ld = load(modelSetup.alexnetPath);
rcnn_model = ld.rcnn_model; clear ld;
%assign the new path to the model

rcnn_model.cnn.binary_file = fullfile('deepped/rcnn',rcnn_model.cnn.binary_file);
rcnn_model.cnn.definition_file = fullfile('deepped/rcnn',rcnn_model.cnn.definition_file);


rcnn_model = rcnn_load_model(rcnn_model, use_gpu);

results = [];

for i = 1 : length(listImages)
    img = imread(listImages{i});
    disp(listImages{i})
    % detect possible pedestrians with LDCF
    bbs = acfDetect(img,detector);
    dt_ldcf = bbs;
    
    % evaluate BBs retrieved by LDCF with our finetuned AlexNet
    bbs(:,3) = bbs(:,1) + bbs(:,3);
    bbs(:,4) = bbs(:,2) + bbs(:,4);
    bbs(:,5) = [];

    feat = rcnn_features(img, bbs, rcnn_model);
    if size(feat) == [0,0]
        disp('no detections')
        continue;
    end;

    scores_cnn = feat*PersonW + PersonB;

    % use second level SVM
    scores = [dt_ldcf(:,5) scores_cnn]*cl2.W+cl2.b;
    
    % discard BBs with too low score and apply NMS
    I = find(scores(:) > modelSetup.bbSupThreshold);
    scored_boxes = cat(2, bbs(I, :), scores(I));
    keep = nms(scored_boxes, 0.3);
    dets = scored_boxes(keep, :);
    dets(:,3) = dets(:,3) - dets(:,1);
    dets(:,4) = dets(:,4) - dets(:,2);
    
    idx = [repmat(i,[size(dets,1),1]),dets];
    results = cat(1,results,idx);

end

d=fileparts(bbsNm); if(~isempty(d)&&~exist(d,'dir')), mkdir(d);end;
dlmwrite(bbsNm,results);

end