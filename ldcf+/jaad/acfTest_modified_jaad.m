function bbsNm = acfTest_modified_jaad( varargin )
% Test aggregate channel features object detector given ground truth.
%
% USAGE
%  [miss,roc,gt,dt] = acfTest( pTest )
%
% INPUTS
%  pTest    - parameters (struct or name/value pairs)
%   .name     - ['REQ'] detector name
%   .imgDir   - ['REQ'] dir containing test images
%   .gtDir    - ['REQ'] dir containing test ground truth
%   .pLoad    - [] params for bbGt>bbLoad for test data (see bbGt>bbLoad)
%   .pModify  - [] params for acfModify for modifying detector
%   .thr      - [.5] threshold on overlap area for comparing two bbs
%   .mul      - [0] if true allow multiple matches to each gt
%   .reapply  - [0] if true re-apply detector even if bbs already computed
%   .ref      - [10.^(-2:.25:0)] reference points (see bbGt>compRoc)
%   .lims     - [3.1e-3 1e1 .05 1] plot axis limits
%   .show     - [0] optional figure number for display
%
% OUTPUTS
%  miss     - log-average miss rate computed at reference points
%  roc      - [nx3] n data points along roc of form [score fp tp]
%  gt       - [mx5] ground truth results [x y w h match] (see bbGt>evalRes)
%  dt       - [nx6] detect results [x y w h score match] (see bbGt>evalRes)
%
% EXAMPLE
%
% See also acfTrain, acfDetect, acfModify, acfDemoInria, bbGt
%
% Piotr's Computer Vision Matlab Toolbox      Version 3.40
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

% get parameters
dfs={ 'name','REQ', 'modelPath','', 'imgDir','REQ','resultsPath','', ...
    'pModify',[], 'reapply',0, 'dataset', 'misc', 'testImgsRange', [], 'scale',1 };
[name,modelPath, imgDir,resultsPath,pModify,reapply, dataset, testImgsRange,scale] = ...
    getPrmDflt(varargin,dfs,1);

% run detector on directory of images

scalestr = num2str(scale);
dotIdx = strfind(scalestr,'.');
if ~isempty(dotIdx), scalestr(dotIdx)=''; end;

if ~isempty(testImgsRange)
bbsNm=[resultsPath 'ldcf+/' name '/'  dataset '/' num2str(testImgsRange(1)) '-' num2str(testImgsRange(2)) '_s' scalestr '_Dets.txt'];
else bbsNm=[resultsPath 'ldcf+/' name '/'  dataset '/all' '_s' scalestr '_Dets.txt']; end


if(reapply && exist(bbsNm,'file')), delete(bbsNm); end
if(reapply || ~exist(bbsNm,'file'))
    detector = load([modelPath name '.mat']);
    detector = detector.detector;
    if(~isempty(pModify)), detector=acfModify(detector,pModify); end
    detector.opts.pPyramid.nApprox = 0;
    imgNms = bbGt('getFiles',{imgDir});
    if ~isempty(testImgsRange)
        imgNms = imgNms(testImgsRange(1):testImgsRange(2));
    end
    acfDetect_modified(imgNms,detector,bbsNm);
else fprintf('the detection file %s already exists. \n',bbsNm); end;

end
