function ccf_train(modelSetup,dataInfo)

% Check if the detector exist
name = modelSetup.modelName;
modelPath = fullfile('ccf', modelSetup.modelPath, '/');
dataSet = dataInfo.dataSet; 

useCF = modelSetup.useCF;
fileExt = ''; if useCF, fileExt = '_cf'; end;
nm = [modelPath name fileExt  '.mat'];
t=exist(nm,'file');
if(t), fprintf('Detector named %s already exists. Skip training. \n', nm); return; end

% initialize opts struct
opts = initializeOpts();

%% modify opts
opts.name = [modelPath name];
opts.modelDs = modelSetup.modelDs;
opts.modelDsPad =  modelSetup.modelDsPad;
opts.pPyramid.pChns.pColor.smooth=0;
opts.pBoost.discrete=0;
opts.nWeak= modelSetup.nWeak;
opts.pBoost.pTree.maxDepth = 3;%5
opts.pBoost.pTree.fracFtrs = 1/16;
opts.nNeg = 25000;
opts.nAccNeg = 50000;
opts.pPyramid.pChns.pGradHist.softBin=1; 
opts.pJitter=struct('flip',1);
opts.posGtDir = dataInfo.trainAnnotDir;
opts.posImgDir = dataInfo.trainDir;
opts.pPyramid.pChns.shrink = 4;
opts.pPyramid.nOctUp=1;
opts.pBoost.pTree.nThreads = 16;

% additional parameters for detection
hRng = modelSetup.hRng; vRng = modelSetup.vRng;
xRng = modelSetup.xRng; yRng = modelSetup.yRng;
ignOcc = modelSetup.ignOcc;
useLDCF = modelSetup.useLDCF;
useCF =  modelSetup.useCF;
squarify = modelSetup.squarify;
cascCal = modelSetup.cascCal;

pLoad = getTags(dataSet);
pLoad = [pLoad 'ignOcc', ignOcc, 'squarify', squarify];
if(~isempty(yRng)),  pLoad = [pLoad 'yRng', yRng]; end
if(~isempty(xRng)),  pLoad = [pLoad 'xRng', xRng]; end
if(~isempty(vRng)),  pLoad = [pLoad 'vRng', vRng]; end
if(~isempty(hRng)),  pLoad = [pLoad 'hRng', hRng]; end

opts.pLoad = pLoad;

%% check

%% optionally switch to LDCF version of detector (see acfTrain)


if(useLDCF), opts.filters = [5 4];
    opts.name = [modelSetup.modelName '_LDCF']; end



%creat  detector files
t=fileparts(nm); if(~isempty(t) && ~exist(t,'dir')), mkdir(t); end
detector = struct( 'opts',opts, 'clf',[], 'info',[] );
startTrain=clock; nm=[opts.name 'Log.txt'];
if(exist(nm,'file')), diary(nm); diary('off'); delete(nm); end; diary(nm);
RandStream.setGlobalStream(RandStream('mrg32k3a','Seed',opts.seed));


%initialize caffe

Is1Samples = [];
Is0Samples = [];

X1CNN = [];
X0CNN = [];

% iterate bootstraping and training

folderSamples = ['ccf/models/' dataSet '/'];

if ~exist(folderSamples,'dir'), mkdir(folderSamples); end;


if useCF
for stage = 0:numel(opts.nWeak)-1
  diary('on'); fprintf([repmat('-',[1 75]) '\n']);
  fprintf('Collecting Samples %i\n',stage); startStage=clock;
% sample positives and compute info about channels
  if( stage==0 )
    [Is1,IsOrig1] = sampleWins( detector, stage, 1 );
    t=ndims(Is1); if(t==3), t=Is1(:,:,1); else t=Is1(:,:,:,1); end
    t=chnsCompute(t,opts.pPyramid.pChns); detector.info=t.info;
    Is1Samples = Is1;
  end
  
  % compute local decorrelation filters fix for LDCF
  if( stage==0 && length(opts.filters)==2 )
    fs = opts.filters; opts.filters = [];
    X1 = chnsCompute1( IsOrig1, opts );
    fs = chnsCorrelation( X1, fs(1), fs(2) );
    opts.filters = fs; detector.opts.filters = fs;
  end
 
  % compute lambdas
  if( stage==0 && isempty(opts.pPyramid.lambdas) )
    fprintf('Computing lambdas... '); start=clock;
    ds=size(IsOrig1); ds(1:end-1)=1; IsOrig1=mat2cell2(IsOrig1,ds);
    ls=chnsScaling_new(opts.pPyramid.pChns,IsOrig1,0);
    ls=round(ls*10^5)/10^5; detector.opts.pPyramid.lambdas=ls;
    fprintf('done (time=%.0fs).\n',etime(clock,start));
  end
  
  % compute features for positives
  if( stage==0 )
    X1 = chnsCompute1( Is1, opts );
    X1 = reshape(X1,[],size(X1,4))';
    clear Is1 IsOrig1 ls fs ds t;
  end
  
  % sample negatives and compute features
  Is0 = sampleWins( detector, stage, 0 );
  Is0Samples = cat(4,Is0Samples,Is0);
  
  
  X0 = chnsCompute1( Is0, opts ); 
  clear Is0;
  X0 = reshape(X0,[],size(X0,4))';
  
  % accumulate negatives from previous stages
  if( stage>0 )
    n0=size(X0p,1); n1=max(opts.nNeg,opts.nAccNeg)-size(X0,1);
    if(n0>n1 && n1>0), X0p=X0p(randSample(n0,n1),:); end
    if(n0>0 && n1>0), X0=[X0p; X0]; end %#ok<AGROW>
  end; X0p=X0;
  
  % train boosted clf
  detector.opts.pBoost.nWeak = opts.nWeak(stage+1);
  detector.clf = adaBoostTrain(X0,X1,detector.opts.pBoost);
  detector.clf.hs = detector.clf.hs + opts.cascCal;
  
  % update log
  fprintf('Done collecting stage %i (time=%.0fs).\n',...
    stage,etime(clock,startStage)); diary('off');
end
else
    posSamples = [folderSamples 'posSamples.mat'];
    if ~exist( posSamples , 'file')
      % sample positives and compute info about channels
      Is1Samples = sampleWins( detector, 0, 1 );
      save(posSamples,'Is1Samples','-v7.3');
    else
        load(posSamples);
    end
       negSamples = [folderSamples 'negSamples.mat'];
    if ~exist( negSamples , 'file')
      Is0Samples = sampleWins( detector, 0, 0 );
      save(negSamples,'Is0Samples','-v7.3');
    else
               load(negSamples);
    end
end


% Generate CNN features

X1CNN = getCnnFeatures(Is1Samples,modelSetup);
X0CNN = getCnnFeatures(Is0Samples,modelSetup);

if (useCF)
    opts.name = [modelName '_CF'];
    X1CNN = reshape(X1CNN, [], size(X1CNN,4))';
    X1 = cat(1,X1CNN,X1);
    
    X0CNN = reshape(X0CNN, [], size(X0CNN,4))';
    X0 = cat(1,X0CNN,X0);
    
else
    X1 = reshape(X1CNN, [], size(X1CNN,4))';
    X0 = reshape(X0CNN, [], size(X0CNN,4))';
end
    
% save detector
% save([opts.name 'Detector.mat'],'detector');
% Train trees using CNN features

detector = struct( 'opts',[], 'clf',[], 'info',[] );
detector.opts = acfTrain();
opts.modelDs = modelSetup.modelDs;
opts.modelDsPad =  modelSetup.modelDsPad;
detector.opts.pPyramid.pChns.shrink = 4;
detector.opts.pBoost.nWeak = 4096;
detector.opts.pBoost.discrete = 0;
detector.opts.pBoost.pTree.maxDepth = 5;
detector.opts.pBoost.pTree.nThreads = 16;
detector.opts.pBoost.pTree.fracFtrs = 1/16;
detector.clf = adaBoostTrain(X0,X1,detector.opts.pBoost);
detector.clf.hs = detector.clf.hs + 0.025;
detector.info = opts.name;
save([opts.name '.mat'],'_detector');

end

function opts = initializeOpts( varargin )
% Initialize opts struct.
dfs= { 'pPyramid',{}, 'filters',[], ...
  'modelDs',[100 41], 'modelDsPad',[128 64], ...
  'pNms',struct(), 'stride',4, 'cascThr',-1, 'cascCal',.005, ...
  'nWeak',128, 'pBoost', {}, 'seed',0, 'name','', 'posGtDir','', ...
  'posImgDir','', 'negImgDir','', 'posWinDir','', 'negWinDir','', ...
  'imreadf',@imread, 'imreadp',{}, 'pLoad',{}, 'nPos',inf, 'nNeg',5000, ...
  'nPerNeg',25, 'nAccNeg',10000, 'pJitter',{}, 'winsSave',0 };
opts = getPrmDflt(varargin,dfs,1);
% fill in remaining parameters
p=chnsPyramid([],opts.pPyramid); p=p.pPyramid;
p.minDs=opts.modelDs; shrink=p.pChns.shrink;
opts.modelDsPad=ceil(opts.modelDsPad/shrink)*shrink;
p.pad=ceil((opts.modelDsPad-opts.modelDs)/shrink/2)*shrink;
p=chnsPyramid([],p); p=p.pPyramid; p.complete=1;
p.pChns.complete=1; opts.pPyramid=p;
% initialize pNms, pBoost, pBoost.pTree, and pLoad
dfs={ 'type','maxg', 'overlap',.65, 'ovrDnm','min' };
opts.pNms=getPrmDflt(opts.pNms,dfs,-1);
dfs={ 'pTree',{}, 'nWeak',0, 'discrete',1, 'verbose',16 };
opts.pBoost=getPrmDflt(opts.pBoost,dfs,1);
dfs={'nBins',256,'maxDepth',2,'minWeight',.01,'fracFtrs',1,'nThreads',16};
opts.pBoost.pTree=getPrmDflt(opts.pBoost.pTree,dfs,1);
opts.pLoad=getPrmDflt(opts.pLoad,{'squarify',{0,1}},-1);
opts.pLoad.squarify{2}=opts.modelDs(2)/opts.modelDs(1);
end

function [Is,IsOrig] = sampleWins( detector, stage, positive )
% Load or sample windows for training detector.
opts=detector.opts; start=clock;
if( positive ), n=opts.nPos; else n=opts.nNeg; end
if( positive ), crDir=opts.posWinDir; else crDir=opts.negWinDir; end
if( exist(crDir,'dir') && stage==0 )
  % if window directory is specified simply load windows
  fs=bbGt('getFiles',{crDir}); nImg=length(fs); assert(nImg>0);
  if(nImg>n), fs=fs(:,randSample(nImg,n)); else n=nImg; end
  for i=1:n, fs{i}=[{opts.imreadf},fs(i),opts.imreadp]; end
  Is=cell(1,n); parfor i=1:n, Is{i}=feval(fs{i}{:}); end
else
  % sample windows from full images using sampleWins1()
  hasGt=positive||isempty(opts.negImgDir); fs={opts.negImgDir};
  if(hasGt), fs={opts.posImgDir,opts.posGtDir}; end
  fs=bbGt('getFiles',fs); nImg=size(fs,2); assert(nImg>0);
  if(~isinf(n)), fs=fs(:,randperm(nImg)); end; Is=cell(nImg*1000,1);
  diary('off'); tid=ticStatus('Sampling windows',1,30); k=0; i=0; batch=64;
  while( i<nImg && k<n )
    batch=min(batch,nImg-i); Is1=cell(1,batch);
    parfor j=1:batch, ij=i+j; %par
      I = feval(opts.imreadf,fs{1,ij},opts.imreadp{:}); %#ok<PFBNS>
      gt=[]; if(hasGt), [~,gt]=bbGt_jaad('bbLoad',fs{2,ij},opts.pLoad); end
      Is1{j} = sampleWins1( I, gt, detector, stage, positive );
    end
    Is1=[Is1{:}]; k1=length(Is1); Is(k+1:k+k1)=Is1; k=k+k1;
    if(k>n), Is=Is(randSample(k,n)); k=n; end
    i=i+batch; tocStatus(tid,max(i/nImg,k/n));
  end
  Is=Is(1:k); diary('on');
  fprintf('Sampled %i windows from %i images.\n',k,i);
end
% optionally jitter positive windows
if(length(Is)<2), Is={}; return; end
nd=ndims(Is{1})+1; Is=cat(nd,Is{:}); IsOrig=Is;
if( positive && isstruct(opts.pJitter) )
  opts.pJitter.hasChn=(nd==4); Is=jitterImage(Is,opts.pJitter);
  ds=size(Is); ds(nd)=ds(nd)*ds(nd+1); Is=reshape(Is,ds(1:nd));
end
% make sure dims are divisible by shrink and not smaller than modelDsPad
ds=size(Is); cr=rem(ds(1:2),opts.pPyramid.pChns.shrink); s=floor(cr/2)+1;
e=ceil(cr/2); Is=Is(s(1):end-e(1),s(2):end-e(2),:,:); ds=size(Is);
if(any(ds(1:2)<opts.modelDsPad)), error('Windows too small.'); end
% optionally save windows to disk and update log
nm=[opts.name 'Is' int2str(positive) 'Stage' int2str(stage)];
if( opts.winsSave ), save(nm,'Is','-v7.3'); end
fprintf('Done sampling windows (time=%.0fs).\n',etime(clock,start));
diary('off'); diary('on');
end

function Is = sampleWins1( I, gt, detector, stage, positive )
% Sample windows from I given its ground truth gt.
opts=detector.opts; shrink=opts.pPyramid.pChns.shrink;
modelDs=opts.modelDs; modelDsPad=opts.modelDsPad;
if( positive ), bbs=gt; bbs=bbs(bbs(:,5)==0,:); else
  if( stage==0 )
    % generate candidate bounding boxes in a grid
    [h,w,~]=size(I); h1=modelDs(1); w1=modelDs(2);
    n=opts.nPerNeg; ny=sqrt(n*h/w); nx=n/ny; ny=ceil(ny); nx=ceil(nx);
    [xs,ys]=meshgrid(linspace(1,w-w1,nx),linspace(1,h-h1,ny));
    bbs=[xs(:) ys(:)]; bbs(:,3)=w1; bbs(:,4)=h1; bbs=bbs(1:n,:);
  else
    % run detector to generate candidate bounding boxes
    bbs=acfDetect(I,detector); [~,ord]=sort(bbs(:,5),'descend');
    bbs=bbs(ord(1:min(end,opts.nPerNeg)),1:4);
  end
  if( ~isempty(gt) )
    % discard any candidate negative bb that matches the gt
    n=size(bbs,1); keep=false(1,n);
    for i=1:n, keep(i)=all(bbGt('compOas',bbs(i,:),gt,gt(:,5))<.1); end
    bbs=bbs(keep,:);
  end
end
% grow bbs to a large padded size and finally crop windows
modelDsBig=max(8*shrink,modelDsPad)+max(2,ceil(64/shrink))*shrink;
r=modelDs(2)/modelDs(1); assert(all(abs(bbs(:,3)./bbs(:,4)-r)<1e-5));
r=modelDsBig./modelDs; bbs=bbApply('resize',bbs,r(1),r(2));
Is=bbApply('crop',I,bbs,'replicate',modelDsBig([2 1]));
end

function chns = chnsCompute1( Is, opts )
% Compute single scale channels of dimensions modelDsPad.
if(isempty(Is)), chns=[]; return; end
fprintf('Extracting features... '); start=clock; fs=opts.filters;
pChns=opts.pPyramid.pChns; smooth=opts.pPyramid.smooth;
dsTar=opts.modelDsPad/pChns.shrink; ds=size(Is); ds(1:end-1)=1;
Is=squeeze(mat2cell2(Is,ds)); n=length(Is); chns=cell(1,n);
parfor i=1:n
  C=chnsCompute(Is{i},pChns); C=convTri(cat(3,C.data{:}),smooth);
  if(~isempty(fs)), C=repmat(C,[1 1 size(fs,4)]);
    for j=1:size(C,3), C(:,:,j)=conv2(C(:,:,j),fs(:,:,j),'same'); end; end
  if(~isempty(fs)), C=imResample(C,.5); shr=2; else shr=1; end
  ds=size(C); cr=ds(1:2)-dsTar/shr; s=floor(cr/2)+1; e=ceil(cr/2);
  C=C(s(1):end-e(1),s(2):end-e(2),:); chns{i}=C;
end; chns=cat(4,chns{:});
fprintf('done (time=%.0fs).\n',etime(clock,start));
end

function filters = chnsCorrelation( chns, wFilter, nFilter )
% Compute filters capturing local correlations for each channel.
fprintf('Computing correlations... '); start=clock;
[~,~,m,n]=size(chns); w=wFilter; wp=w*2-1;
filters=zeros(w,w,m,nFilter,'single');
for i=1:m
  % compute local auto-scorrelation using Wiener-Khinchin theorem
  mus=squeeze(mean(mean(chns(:,:,i,:)))); sig=cell(1,n);
  parfor j=1:n
    T=fftshift(ifft2(abs(fft2(chns(:,:,i,j)-mean(mus))).^2));
    sig{j}=T(floor(end/2)+1-w+(1:wp),floor(end/2)+1-w+(1:wp));
  end
  sig=double(mean(cat(4,sig{mus>1/50}),4));
  sig=reshape(full(convmtx2(sig,w,w)),wp+w-1,wp+w-1,[]);
  sig=reshape(sig(w:wp,w:wp,:),w^2,w^2); sig=(sig+sig')/2;
  % compute filters for each channel from sig (sorted by eigenvalue)
  [fs,D]=eig(sig); fs=reshape(fs,w,w,[]);
  [~,ord]=sort(diag(D),'descend');
  fs=flipdim(flipdim(fs,1),2); %#ok<DFLIPDIM>
  filters(:,:,i,:)=fs(:,:,ord(1:nFilter));
end
fprintf('done (time=%.0fs).\n',etime(clock,start));
end


