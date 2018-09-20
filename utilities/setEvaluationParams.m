function evalSetup = setEvaluationParams(varargin)
ip = inputParser;

%% Opt parameters for evaluation
ip.addParameter('squarify',      {{3,.41}},             @iscell); %set the ratio of bounding boxes 
ip.addParameter('hRng',    [50 inf],          @ismatrix);% Horizontal range of detected pedestrians
ip.addParameter('vRng',    [.65 1],          @ismatrix); % Vertical ratio
ip.addParameter('xRng',    [],          @ismatrix);% Width range to be searched within the image. [5 635] caltech
ip.addParameter('yRng',    [],          @ismatrix);% Height range to be searched within the image. [5 475] caltech
ip.addParameter('ignOcc',   1,          @isscalar);% to ignore gts with occlusion
ip.addParameter('show',    0,          @isscalar); % Show the roc curve
ip.addParameter('showResults',    1,          @isscalar); % Show the miss rate and iou
ip.addParameter('threshold',    0.5,          @isscalar);% The overlap threhsold for detection to be considered as hit
ip.addParameter('thr_min_error',    0.2,          @isscalar);% The overlap threhsold for detection to be considered as hit
ip.addParameter('mul',    0,          @isscalar);% Set 1 to allow detecting multiple bounding boxes
ip.addParameter('ref_mr2',    10.^(-2:.25:0),          @isscalar);% samples for computing area under the curve 
ip.addParameter('ref_mr4',    10.^(-4:.25:0),          @isscalar);% samples for computing area under the curve
ip.addParameter('rocLimits',   [2e-4 50 .035 1],          @ismatrix); % Axis limits for ROC plots
ip.addParameter('genIndivResults',    0,          @isscalar); % 1: generates detection results for each frame with ground truth id it matches
ip.addParameter('posFix',    '',           @ischar);
ip.addParameter('saveMR', 0, @isscalar);% save miss rate results
ip.addParameter('saveResults', 0, @isscalar);% save all results
ip.addParameter('saveImgList', 0, @isscalar);% save a list of image names that were evaluated

ip.parse(varargin{:});
evalSetup = ip.Results;
end
