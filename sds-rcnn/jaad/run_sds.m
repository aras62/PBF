function bbsNm = run_sds(dataInfo,modelSetup)

trOrDet= modelSetup.trOrDet ;
bbsNm = [];

 if trOrDet < 2
     sds_train(dataInfo,modelSetup)
 end

if trOrDet > 0
    bbsNm = sds_detect(dataInfo, modelSetup);
end
end

