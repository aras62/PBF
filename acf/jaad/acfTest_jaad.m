% function [miss,roc,gt,dt] = acfTest_jaad( varargin )
function [bbsNm] = acfTest_jaad( varargin )

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
%  fName    - name of text file
%  pLoad    - parameters (struct or name/value pairs)
%   .format   - [0] gt format 0:default, 1:PASCAL, 2:ImageNet
%   .ellipse  - [1] controls how oriented bb is converted to regular bb
%   .squarify - [] controls optional reshaping of bbs to fixed aspect ratio
%   .lbls     - [] return objs with these labels (or [] to return all)
%   .ilbls    - [] return objs with these labels but set to ignore
%   .hRng     - [] range of acceptable obj heights
%   .wRng     - [] range of acceptable obj widths
%   .aRng     - [] range of acceptable obj areas
%   .arRng    - [] range of acceptable obj aspect ratios
%   .oRng     - [] range of acceptable obj orientations (angles)
%   .xRng     - [] range of x coordinates of bb extent
%   .yRng     - [] range of y coordinates of bb extent
%   .vRng     - [] range of acceptable obj occlusion levels

% get parameters
dfs={ 'name','REQ', 'modelPath','', 'imgDir','REQ','resultsPath','', ...
    'pModify',[], 'reapply',0, 'dataset', 'misc', 'testImgsRange', [], 'scale',1 };
[name,modelPath, imgDir,resultsPath,pModify,reapply, dataset, testImgsRange,scale] = ...
    getPrmDflt(varargin,dfs,1);

scalestr = num2str(scale);
dotIdx = strfind(scalestr,'.');

if ~isempty(dotIdx), scalestr(dotIdx)=''; end;

if ~isempty(testImgsRange)
    bbsNm=[resultsPath 'acf/' name '/'  dataset '/' num2str(testImgsRange(1)) '-' num2str(testImgsRange(2)) '_s' scalestr '_Dets.txt'];
else bbsNm=[resultsPath 'acf/' name '/'  dataset '/all' '_s' scalestr '_Dets.txt']; end

if(reapply && exist(bbsNm,'file')), delete(bbsNm); end
if(~reapply && exist(bbsNm,'file')), fprintf('the detection file %s already exists. \n',bbsNm); return;end;

nm=[modelPath name '.mat'];
t=exist(nm,'file');
if(~t),fprintf('Detector named %s does not exists.\n', nm)
    bbsNm = []; return; end


detector = load(nm);
detector = detector.detector;
if(~isempty(pModify)), detector=acfModify(detector,pModify); end
imgNms = bbGt('getFiles',{imgDir});
if ~isempty(testImgsRange)
    imgNms = imgNms(testImgsRange(1):testImgsRange(2));
end
acfDetect(imgNms, detector, bbsNm );

end
