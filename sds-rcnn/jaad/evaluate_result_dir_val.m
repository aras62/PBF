function [scores, thres, recall, dts, gts] = evaluate_result_dir_val(aDirs, minh, gt_path,...
    dataSet)
% Evaluate and plot all pedestrian detection results.
%% Set parameters by altering this function directly.
%
% USAGE
%  dbEval
%
% INPUTS
%
% OUTPUTS
%
% EXAMPLE
%  dbEval
%
% See also bbGt, dbInfo
%
% Caltech Pedestrian Dataset     Version 3.2.1
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]
% List of experiment settings: { name, hr, vr, ar, overlap, filter }
%  name     - experiment name
%  hr       - height range to test
%  vr       - visibility range to test
%  ar       - aspect ratio range to test
%  overlap  - overlap threshold for evaluation
%  filter   - expanded filtering (see 3.3 in PAMI11)

scores = 0;
thres = 0;
if nargin < 3
    minh = 50;
end

exps = {
    'Reasonable',     [minh inf],  [.65 inf], 0,   .5,  1.25
    'All',            [20 inf],  [.2 inf],  0,   .5,  1.25
    'Scale=large',    [100 inf], [inf inf], 0,   .5,  1.25
    'Scale=near',     [80 inf],  [inf inf], 0,   .5,  1.25
    'Scale=medium',   [30 80],   [inf inf], 0,   .5,  1.25
    'Scale=far',      [20 30],   [inf inf], 0,   .5,  1.25
    'Occ=none',       [50 inf],  [inf inf], 0,   .5,  1.25
    'Occ=partial',    [50 inf],  [.65 1],   0,   .5,  1.25
    'Occ=heavy',      [50 inf],  [.2 .65],  0,   .5,  1.25
    'Ar=all',         [50 inf],  [inf inf], 0,   .5,  1.25
    'Ar=typical',     [50 inf],  [inf inf],  .1, .5,  1.25
    'Ar=atypical',    [50 inf],  [inf inf], -.1, .5,  1.25
    'Overlap=25',     [50 inf],  [.65 inf], 0,   .25, 1.25
    'Overlap=50',     [50 inf],  [.65 inf], 0,   .50, 1.25
    'Overlap=75',     [50 inf],  [.65 inf], 0,   .75, 1.25
    'Expand=100',     [50 inf],  [.65 inf], 0,   .5,  1.00
    'Expand=125',     [50 inf],  [.65 inf], 0,   .5,  1.25
    'Expand=150',     [50 inf],  [.65 inf], 0,   .5,  1.50 };
exps=cell2struct(exps',{'name','hr','vr','ar','overlap','filter'});

exps = exps(1);

% List of algorithms: { name, resize, color, style }
%  name     - algorithm name (defines data location)
%  resize   - if true rescale height of each box by 100/128
%  color    - algorithm plot color
%  style    - algorithm plot linestyle

n=1000; clrs=zeros(n,3);
for i=1:n, clrs(i,:)=max(.3,mod([78 121 42]*(i+1),255)/255); end

% remaining parameters and constants
aspectRatio = .41;        % default aspect ratio for all bbs
bnds = [5 5 635 475];     % discard bbs outside this pixel range
samples = 10.^(-2:.25:0); % samples for computing area under the curve

bnds0 = bnds;
% handle special database specific cases
if(any(strcmpi(dataSet,{'inria','tudbrussels','eth', 'jaad', 'kitti'})))
    bnds=[-inf -inf inf inf]; else bnds=bnds0; end

pLoad = getTags(dataSet);

pLoad = [pLoad 'yRng', bnds([2 4])];
pLoad = [pLoad 'xRng', bnds([1 3])];
pLoad = [pLoad 'vRng', exps.vr];
pLoad = [pLoad 'hRng', exps.hr];



% load detections and ground truth and evaluate
[dts, gts] = loadDt_Gt(aspectRatio, aDirs, gt_path, pLoad);
[gts,dts] = bbGt('evalRes',gts,dts,exps.overlap,0);
[~,~,~,miss] = bbGt('compRoc',gts,dts,1,samples );
scores=exp(mean(log(max(1e-10,1-miss))));

    tp=0; missed=0; 
    for i=1:length(gts)
        for j = 1:size(gts{i},1) 
            if (gts{i}(j,5) == 1)
                tp=tp+1;
            elseif (gts{i}(j,5) == 0)
                missed=missed+1;
            end
        
        end
    end
    recall = tp/(tp+missed);


end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function res = evalAlgs( plotName, algs, exps, gts, dts )
% Evaluate every algorithm on each experiment
%
% OUTPUTS
%  res    - nGt x nDt cell of all evaluations, each with fields
%   .stra   - string identifying algorithm
%   .stre   - string identifying experiment
%   .gtr    - [n x 1] gt result bbs for each frame [x y w h match]
%   .dtr    - [n x 1] dt result bbs for each frame [x y w h score match]
%fprintf('Evaluating: %s\n',plotName);
nGt=length(gts); nDt=length(dts);
res=repmat(struct('stra',[],'stre',[],'gtr',[],'dtr',[]),nGt,nDt);
for g=1:nGt
    for d=1:nDt
        gt=gts{g}; dt=dts{d}; n=length(gt); %assert(length(dt)==n);
        stra=algs{d}; stre=exps(g).name;
        fName = [plotName '/ev-' [stre '-' stra] '.mat'];
        if(exist(fName,'file')), R=load(fName); res(g,d)=R.R; continue; end
        %fprintf('\tExp %i/%i, Alg %i/%i: %s/%s\n',g,nGt,d,nDt,stre,stra);
        hr = exps(g).hr.*[1/exps(g).filter exps(g).filter];
        for f=1:n
            bb=dt{f};
            dt{f}=bb(bb(:,4)>=hr(1) & bb(:,4)<hr(2),:); end
        [gtr,dtr] = bbGt('evalRes',gt,dt,exps(g).overlap);
        R=struct('stra',stra,'stre',stre,'gtr',{gtr},'dtr',{dtr});
        res(g,d)=R; %save(fName,'R');
    end
end
end


function [dts,gts] = loadDt_Gt(aspectRatio, aDirs, gt_path, pLoad)
resize = 1;
[resNms,resIds] = bbGt('getFiles',aDirs);
dts = {};
gts = {};
for i=1:length(resNms)
    bbs=load([resNms{i}],'-ascii');
   if(numel(bbs)==0), bbs=zeros(0,6);  end
    dtIds = unique(bbs(:,1), 'stable');
    
    for j = 1:length(dtIds)
        gtName = sprintf('%s/%s_I%05.f.txt',gt_path,resIds{i},dtIds(j)-1);
        [~,gt] = bbGt('bbLoad',gtName, pLoad);
        gts{1,end+1} = gt;
        
        bb=bbs(bbs(:,1)==dtIds(j),2:6);
        bb=bbApply('resize',bb,resize,0,aspectRatio);
        dts{1,end+1} = bb;
    end
end

end


