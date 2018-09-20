function bbsNm = sds_detect(dataInfo, modelSetup)

gpu_id = modelSetup.gpuId;
testImgsRange = dataInfo.testImgsRange;
reapply = modelSetup.reapply;
dataset = dataInfo.dataSet;

resultsPath = dataInfo.resultsPath;
modelName = modelSetup.modelName;

scalestr = num2str(dataInfo.scale);
dotIdx = strfind(scalestr,'.');
if ~isempty(dotIdx), scalestr(dotIdx)=''; end;
modelFolder = [modelName '_' modelSetup.networkType];
if ~isempty(testImgsRange)
bbsNm=[resultsPath 'sds-rcnn/' modelFolder '/'  dataset '/' num2str(testImgsRange(1)) '-' num2str(testImgsRange(2)) '_s' scalestr '_Dets.txt'];
else bbsNm=[resultsPath 'sds-rcnn/' modelFolder '/'  dataset '/all' '_s' scalestr '_Dets.txt']; end


if(reapply && exist(bbsNm,'file')), delete(bbsNm); end
if(~reapply && exist(bbsNm,'file')), fprintf('the detection file %s already exists. \n',bbsNm); return;end;


load([modelSetup.modelRootDir modelName '/' modelSetup.rpnConf]);
load([modelSetup.modelRootDir modelName '/' modelSetup.rcnnConf]);
load([modelSetup.modelRootDir modelName '/' modelSetup.anchors]);
load([modelSetup.modelRootDir modelName '/' modelSetup.bbMeans]);
load([modelSetup.modelRootDir modelName '/' modelSetup.bbStds]);


% test RPN only
rpn_conf.gpu_id     = gpu_id;
rpn_conf.anchors    = anchors;
rpn_conf.bbox_means = bbox_means;
rpn_conf.bbox_stds  = bbox_stds;

reset_caffe(rpn_conf);

rpn_conf.test_dir  = dataInfo.testDir;
imgNms =  bbGt('getFiles',{rpn_conf.test_dir});
if ~isempty(testImgsRange), imgNms = imgNms(testImgsRange(1):testImgsRange(2));end

if strcmpi(modelSetup.networkType,'rpn')
    
% test net
rpn_prototxt = modelSetup.rpnProtoPath;
rpn_weights = [modelSetup.modelRootDir modelName '/' modelSetup.rpnOnlyWeightsPath];

evaluate_results_rpn_jaad(rpn_prototxt, rpn_weights,rpn_conf, bbsNm, imgNms);

elseif strcmpi(modelSetup.networkType,'rpn-bcn')

 rpn_prototxt = modelSetup.rpnProtoPath;
 rpn_weights = [modelSetup.modelRootDir modelName '/' modelSetup.rpnWeightsPath];
 rcnn_prototxt =  modelSetup.rcnnProtoPath;
 rcnn_weights = [modelSetup.modelRootDir modelName '/' modelSetup.rcnnWeightsPath];
 
rcnn_conf.cls_score_sm = modelSetup.cls_score_sm;
rcnn_conf.cls_score2_sm = modelSetup.cls_score2_sm;

evaluate_results_bsn_jaad(rpn_prototxt, rpn_weights, rpn_conf,...
    rcnn_prototxt, rcnn_weights, rcnn_conf, bbsNm, imgNms)
   
else fprintf('Model %s does not exist, select either rpn or rpn-bcn.\n',modelSetup.networkType); bbsNm = []; return; end;

reset_caffe(rpn_conf);

end

