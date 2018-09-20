function bbsNm = run_frcnn(dataInfo,modelSetup)
cd('frcnn')
addpath(genpath('jaad/'));

trOrDet= modelSetup.trOrDet ;
bbsNm = [];
startup;
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = modelSetup.gpuId;

% Adjustment for pathes 
dataInfo.resultsPath = ['../' dataInfo.resultsPath];
dataInfo.testDir = ['../' dataInfo.testDir];
dataInfo.testAnnotDir = ['../' dataInfo.testAnnotDir];
dataInfo.trainDir = ['../' dataInfo.trainDir];
dataInfo.trainAnnotDir = ['../' dataInfo.trainAnnotDir];
dataInfo.valDir = ['../' dataInfo.valDir];
dataInfo.valAnnotDir = ['../' dataInfo.valAnnotDir];

if ~ parallel.internal.gpu.isAnyDeviceSelected
    gpuDevice(opts.gpu_id);
end

evalc('caffe.reset_all();');
caffe.set_device(opts.gpu_id-1);
caffe.set_mode_gpu();

if modelSetup.regenImdbFile, modelSetup.regenRoiFile = 1; end;

if trOrDet < 2
    frcnn_train(dataInfo,modelSetup)
end

if trOrDet > 0
    bbsNm = frcnn_detect(dataInfo,modelSetup);
end
bbsNm = bbsNm(4:end);
cd('..')

end

