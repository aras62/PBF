
function bbsNm = run_mscnn(dataInfo, modelSetup)
trOrDet = modelSetup.trOrDet;
bbsNm = [];

% Initialize a network
if modelSetup.useGpu
    caffe.set_mode_gpu();
    caffe.set_device(modelSetup.gpuId -1);
else
    caffe.set_mode_cpu();
end
if trOrDet < 2
    mscnn_train(dataInfo,modelSetup);
end

if trOrDet > 0
    bbsNm = mscnn_detect(dataInfo, modelSetup);
end
end
