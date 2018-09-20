function [type,numTests] = generateData(dataInfo)

type = 'test';
if (any(strcmpi(dataInfo.dataSet,{'tudbrussels','eth','daimler'})))
    dbInfo_jaad(dataInfo.dataSet,dataInfo.dataPath, dataInfo.jaadSubDatasetDir, dataInfo.jaadSubDataset);
    if(~exist([dataInfo.dataDir type '/annotations'],'dir'))
        dbExtract_jaad([dataInfo.dataDir type],1,dataInfo.skipTest,dataInfo,dataInfo.scale);
    end
else
    annotation_ext = '/annotations';
    if strcmpi(dataInfo.dataSet,'jaad'), annotation_ext  = [annotation_ext '_'  dataInfo.anotationType]; end;
    for s=1:3
        if(s==1),type='test'; skip=dataInfo.skipTest;
        elseif(s==2)
            type='val';skip=dataInfo.skipVal;
        elseif (s==3)
            type='train'; skip=dataInfo.skipTrain;
        end
        dbInfo_jaad([dataInfo.dataSet type],dataInfo.dataPath,dataInfo.jaadSubDatasetDir, dataInfo.jaadSubDataset);
        if(s==3),type=['train' int2str2(skip,2)]; end
        if(exist([dataInfo.dataDir type annotation_ext],'dir')), continue; end
        fprintf('Extracting %s with skip of %d \n',[dataInfo.dataDir type],skip);
        dbExtract_jaad([dataInfo.dataDir type],1,skip,dataInfo, dataInfo.scale);
    end
end
numTests = length(bbGt('getFiles',{dataInfo.testDir}));
end
