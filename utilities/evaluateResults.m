function  results = evaluateResults(bbsNm, dataInfo, evalSetup)
% fprintf('Evaluation begins...\n');
testImgsRange = dataInfo.testImgsRange;
gtDir = dataInfo.testAnnotDir;
thr = evalSetup.threshold;
minthr = evalSetup.thr_min_error;
mul = evalSetup.mul;
ref_mr2 = evalSetup.ref_mr2;
ref_mr4 = evalSetup.ref_mr4;
lims = evalSetup.rocLimits;
hRng = evalSetup.hRng;
vRng = evalSetup.vRng;
xRng = evalSetup.xRng;
yRng = evalSetup.yRng;
ignOcc = evalSetup.ignOcc;
squarify = evalSetup.squarify;
show = evalSetup.show;

pLoad = getTags(dataInfo.dataSet);
pLoad = [pLoad 'ignOcc', ignOcc, 'squarify',squarify];
if(~isempty(yRng)),  pLoad = [pLoad 'yRng', yRng]; end
if(~isempty(xRng)),  pLoad = [pLoad 'xRng', xRng]; end
if(~isempty(vRng)),  pLoad = [pLoad 'vRng', vRng]; end
if(~isempty(hRng)),  pLoad = [pLoad 'hRng', hRng]; end

% run evaluation using bbGt
[objs,gt,dt] = bbGt_jaad('loadAll',gtDir,bbsNm,pLoad);
[~,imgNms] = bbGt('getFiles',{gtDir});

if ~isempty(testImgsRange)
    gt = gt(testImgsRange(1):testImgsRange(2));
    objs = objs(testImgsRange(1):testImgsRange(2));
    dt = dt(1:testImgsRange(2)-testImgsRange(1)+1);
    imgNms = imgNms(testImgsRange(1):testImgsRange(2));
end

% Each gt/dt output row has a flag match that is either -1/0/1:
%  for gt: -1=ignore,  0=fn [unmatched],  1=tp [matched]
%  for dt: -1=ignore,  0=fp [unmatched],  1=tp [matched]
dt0 = dt;
gt0 = gt;

[gt,dt,avg_IOU] = bbGt_jaad('evalRes',gt,dt,thr,mul);

% saves detection results for each image with associated ground truth match
% save eval file
if evalSetup.genIndivResults || evalSetup.saveMR || evalSetup.saveResults || evalSetup.saveImgList ||evalSetup.show
    [nmFold,fileName] = fileparts(bbsNm);
    idx = strfind(nmFold,'/');
    evalPath = ['evaluation_results',nmFold(idx:end)];
    if ~exist(evalPath,'dir'),mkdir(evalPath);end
end


if evalSetup.genIndivResults
    [~,dtb,~] = bbGt_jaad('evalRes_new',gt0,dt0,thr,mul);
    gtMatchesFile = fullfile(evalPath, [fileName '_gtMatches' evalSetup.posFix '_' num2str(evalSetup.threshold) '.txt']);
    fid = fopen(gtMatchesFile ,'w');
    for i = 1: length(dtb)
        for j = 1:size(dtb{i},1)
            if dtb{i}(j,7) > 0
                gtName = objs{i}(dtb{i}(j,7)).lbl;
            else
                gtName = 'na';
            end
            fprintf(fid,'%s\t%03.2f\t%03.2f\t%03.2f\t%03.2f\t%.04f\t%01.f\t%s\t%01.2f\n'...
                ,imgNms{i}, dtb{i}(j,1:6),gtName,dtb{i}(j,8));
        end
    end
    fclose(fid);
end;

[fp,tp,score,miss_mr2] = bbGt_jaad('compRoc',gt,dt,1,ref_mr2);
roc=[score fp tp];
miss_mr2 = exp(mean(log(max(1e-10,1-miss_mr2))));
[~,~,~,miss_mr4] = bbGt_jaad('compRoc',gt,dt,1,ref_mr4);
miss_mr4 = exp(mean(log(max(1e-10,1-miss_mr4))));
average_IOU = sum(avg_IOU)/length(avg_IOU);


results = generateResultStruct(miss_mr2 ,miss_mr4,roc,average_IOU);
results.dt = dt;
results.gt = gt;


%% Save MR files
if evalSetup.saveMR
    fid = fopen(fullfile(evalPath, [fileName '_missRate' evalSetup.posFix '.txt']),'w');
    fprintf(fid,'avg_IOU: %.2f\nMR2: %.2f%%\nMR4: %.2f%%',average_IOU,miss_mr2*100, miss_mr4*100); fclose(fid);
end
if evalSetup.saveResults
    save(fullfile(evalPath, [fileName  '_roc' evalSetup.posFix '_' num2str(evalSetup.threshold) '.mat']), 'results');
end

%% Save list of test images and copy detection file
if evalSetup.saveImgList
    imListName =['testImageList_' dataInfo.dataSet '_' dataInfo.jaadSubDataset ...
        '_skip' num2str(dataInfo.skipTest) '.txt'] ;
    if ~exist(fullfile(evalPath,imListName ),'file')
        fid = fopen(fullfile(evalPath, imListName),'w');
        fprintf(fid,'%s\n',imgNms{:}); fclose(fid);
    end
    copyfile(bbsNm, evalPath);
end


%% Display MR and ROC curves
if miss_mr2 == 1,disp('No detection results'); return; end;

if evalSetup.showResults
    fprintf('*******************  evaluation results*******************\n');
    fprintf('log-average miss rates (MR2, MR4) = %.2f%%, %.2f%%\n',miss_mr2*100, miss_mr4*100);
    fprintf('Mean IOU: %.2f\n\n',average_IOU);
end

%% optionally plot roc
if( show )
    % idx = strfind(bbsNm, '.txt');
    figId=figure(show); plotRoc([fp tp],'logx',1,'logy',1,'xLbl','fppi',...
        'lims',lims,'color','g','smooth',1,'fpTarget',[ref_mr4]);
    title(sprintf('log-average miss rate (MR2,MR4) = %.2f, %.2f%%',miss_mr2*100, miss_mr4*100));
    savefig(fullfile(evalPath, [fileName  '_Roc' evalSetup.posFix]),show,'png');
    pause(2);
    close(figId);
end
end

function resultStruct = generateResultStruct(mr2,mr4,roc, iou)

resultStruct.mr2 = mr2;
resultStruct.mr4 = mr4;
resultStruct.roc = roc;
if nargin > 3
    resultStruct.iou = iou;
end
end
