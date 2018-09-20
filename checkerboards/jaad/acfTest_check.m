function bbsNm = acfTest_check(varargin )
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
% See also acfTrain, acfDetect, acfDemoInria, bbGt
%
% Piotr's Image&Video Toolbox      Version 3.22
% Copyright 2013 Piotr Dollar & Ron Appel.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see external/bsd.txt]

% get parameters
dfs={ 'name','REQ', 'modelPath','', 'imgDir','REQ','resultsPath','', ...
    'reapply',0, 'dataset', 'misc', 'testImgsRange', [], 'scale', 1 };
[name,modelPath, imgDir,resultsPath,reapply, dataset, testImgsRange, scale] = ...
    getPrmDflt(varargin,dfs,1);

%% run detector on directory of images

scalestr = num2str(scale);
dotIdx = strfind(scalestr,'.');
if ~isempty(dotIdx), scalestr(dotIdx)=''; end;
if ~isempty(testImgsRange)
    bbsNm=[resultsPath 'checkerboards/' name '/'  dataset '/' num2str(testImgsRange(1)) '-' num2str(testImgsRange(2)) '_s' scalestr '_Dets.txt'];
else bbsNm=[resultsPath 'checkerboards/' name '/'  dataset '/all' '_s' scalestr '_Dets.txt']; end

if(reapply && exist(bbsNm,'file')), delete(bbsNm); end
if(~reapply && exist(bbsNm,'file')), fprintf('the detection file %s already exists. \n',bbsNm); return;end;


nm=[modelPath name '.mat'];
t=exist(nm,'file');
if(~t),fprintf('Detector named %s does not exists.\n', nm)
    bbsNm = []; return; end


detector = load(nm);
detector = detector.detector;
imgNms = bbGt('getFiles',{imgDir});
if ~isempty(testImgsRange)
    imgNms = imgNms(testImgsRange(1):testImgsRange(2));
end
acfDetect_check(imgNms, detector, bbsNm );

end