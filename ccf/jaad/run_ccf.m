function bbsNm = run_ccf(dataInfo,modelSetup)
pwd
trOrDet= modelSetup.trOrDet ;
bbsNm = [];

if trOrDet < 2
    ccf_train(modelSetup,dataInfo)
end

if trOrDet > 0
    bbsNm = ccf_detect(dataInfo,modelSetup);
end
end


