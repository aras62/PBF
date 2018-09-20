function bbsNm = run_rpnbf(dataInfo,modelSetup)
cd('rpn-bf')
trOrDet= modelSetup.trOrDet ;
bbsNm = [];
p = genpath('jaad/'); addpath(p);

startup;
opts.caffe_version          = 'caffe_faster_rcnn';%t
opts.gpu_id                 = modelSetup.gpuId;
% Adjustment for pathes
dataInfo.resultsPath = ['../' dataInfo.resultsPath];
dataInfo.testDir = ['../' dataInfo.testDir];
dataInfo.testAnnotDir = ['../' dataInfo.testAnnotDir];
dataInfo.trainDir = ['../' dataInfo.trainDir];
dataInfo.trainAnnotDir = ['../' dataInfo.trainAnnotDir];

if ~ parallel.internal.gpu.isAnyDeviceSelected
    gpuDevice(opts.gpu_id);
end

evalc('caffe.reset_all();');
caffe.set_device(opts.gpu_id-1);
caffe.set_mode_gpu();

if modelSetup.regenImdbFile, modelSetup.regenRoiFile = 1; end;

if trOrDet < 2
    if ~modelSetup.bfOnly
        rpn_train(dataInfo,modelSetup)
    end
    if ~modelSetup.rpnOnly
        rpnbf_train(dataInfo,modelSetup)
    end
end

if trOrDet > 0
    bbsNm = rpnbf_detect(dataInfo,modelSetup);
end
bbsNm = bbsNm(4:end);
cd('..')
end

