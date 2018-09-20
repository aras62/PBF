function modelSetup = setAlgorithmParams(algName, varargin)
ip = inputParser;

%% training
algName = lower(algName);
switch(algName)
    case 'acf' % HoG, LDCF, ACF
        ip.addParameter('modelName',    'AcfCaltech+',        @ischar);
        ip.addParameter('squarify',      {{3,.41}},             @iscell); %set the ratio of bounding boxes 
       
        % Parameters for ground truth selection
        ip.addParameter('hRng',    [50 inf],          @ismatrix); % height range of bounding boxes
        ip.addParameter('vRng',    [.65 1],          @ismatrix); % visibily range of pedestrians (only for Caltech)
        ip.addParameter('xRng',    [],          @ismatrix);%[5 635] caltech
        ip.addParameter('yRng',    [],          @ismatrix);%[5 475] caltech
       
        % This tag is to ignore the bounding boxes with occlusion tag. This is particularly useful
        % for JAAD which has two sets of bounding boxes part and full. 
        % For datasets with visibility region (Caltech and Cityperson) set
        % this paramter to 0
        ip.addParameter('ignOcc',    1,          @isscalar); % . Set to 0 for datasets with visibility (e.g. Caltech and cityperson)
        
        % model serch window size
        ip.addParameter('modelDs',    [50 20.5],          @ismatrix); % [50 20.5] caltech [100 41] inria
        ip.addParameter('modelDsPad',    [64 32],          @ismatrix);% [64 32] caltech [128 64]  %inria
        
        % Training paramters
        ip.addParameter('nWeak',    [64 256 1024 4096],          @ismatrix);% [64 256 1024 4096] caltech [32 128 512 2048] inria
        ip.addParameter('cascCal',    0.025,              @isscalar);% 0.025 caltech 0.01 inria
        ip.addParameter('calSetup',    1,              @isscalar);% 0 for inria and jaad
        ip.addParameter('useLDCF',    0,              @isscalar);% Models: 0 = ACF, 1= LDCF
        ip.addParameter('reapply',    1,              @isscalar);% if set to 1, it will rerun the test
        
        % Choose which feature channels to be used for training
        ip.addParameter('channels',    [1 1 1],          @ismatrix);  % color(3), gradMag(1), HOG(6)        
        ip.addParameter('modelPath', 'output', @ischar); % path to read or write models
        
    case 'ccf' % CCF and CCF + CF
        ip.addParameter('modelName',    'Detector_caltech_depth5',        @ischar);
        
        % Parameters for ground truth selection (see acf)
        ip.addParameter('squarify',      {{3,.41}},             @iscell); %set the ratio of bounding boxes 
        ip.addParameter('hRng',    [50 inf],          @ismatrix);
        ip.addParameter('vRng',    [1 1],          @ismatrix);
        ip.addParameter('xRng',    [],          @ismatrix);
        ip.addParameter('yRng',    [],          @ismatrix);
      
        % This tag is to ignore the bounding boxes with occlusion tag. This is particularly useful
        % for JAAD which has two sets of bounding boxes part and full. 
        % For datasets with visibility region (Caltech and Cityperson) set
        % this paramter to 0
        ip.addParameter('ignOcc',    1,          @isscalar);
        
        % model serch window size
        ip.addParameter('modelDs',    [50 20.5].*2,          @ismatrix);
        ip.addParameter('modelDsPad',    [64 32].*2,          @ismatrix);
        
        % Training paramters
        ip.addParameter('nWeak',    [64 256 1024 4096],          @ismatrix);% [64 256 1024 4096] caltech [32 128 512 2048] inria
        ip.addParameter('cascCal',    0.025,              @isscalar);% 0.025 caltech 0.01 inria
        ip.addParameter('input_size',    900,              @isscalar);
        ip.addParameter('stride',    4,              @isscalar);
        ip.addParameter('pad',    16,              @isscalar);
        ip.addParameter('minDs',    72,              @isscalar);
        ip.addParameter('nPerOct',    6,              @isscalar);
        ip.addParameter('nOctUp',    1,              @isscalar);
        ip.addParameter('imresize',    1,              @isscalar);
        ip.addParameter('imflip',    0,              @isscalar);
        ip.addParameter('useLDCF',    1,              @ischar);
        
        % model parameters
        ip.addParameter('id',    6,              @isscalar); % id = 0: 1 thread running, 1~8: 8 threads running for processing
        ip.addParameter('useCF',    0,              @isscalar); % Models: 0 = CCF, 1 = CCF+CF (use CF features from acf)
        ip.addParameter('savePyr',    0,              @isscalar);% saves pyramids. requires a lot of disk space
        
        %Path to pretrained vgg models and protxt files
        ip.addParameter('modelDef',    'ccf/data/CaffeNets/VGG_ILSVRC_16_layers_conv3.prototxt',              @ischar);
        ip.addParameter('modelDefTrain',   'ccf/data/CaffeNets/VGG_ILSVRC_16_layers_conv3_144.prototxt',              @ischar);
        ip.addParameter('modelFile',    'ccf/data/CaffeNets/VGG_ILSVRC_16_layers.caffemodel',              @ischar);
        ip.addParameter('meanPix',    [103.939 116.779 123.68],          @ismatrix);% mean value for network training
        ip.addParameter('nApprox',    0,              @isscalar);%use Power law. doesn't work on caltech
        ip.addParameter('gpuId',    1,              @isscalar);% Which GPU to use
        ip.addParameter('reapply',    0,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('modelPath', 'output', @ischar); % path to read or write models
        
    case 'checkerboards'
        ip.addParameter('modelName',    'Checkerboards',        @ischar);
        
        % Parameters for ground truth selection (see acf)
        ip.addParameter('squarify',      {{3,.41}},             @iscell); %set the ratio of bounding boxes 
        ip.addParameter('hRng',    [50 inf],          @ismatrix);
        ip.addParameter('vRng',    [.65 1],          @ismatrix);
        ip.addParameter('xRng',    [],          @ismatrix);
        ip.addParameter('yRng',    [],          @ismatrix);
   
        % This tag is to ignore the bounding boxes with occlusion tag. This is particularly useful
        % for JAAD which has two sets of bounding boxes part and full. 
        % For datasets with visibility region (Caltech and Cityperson) set
        % this paramter to 0
        ip.addParameter('ignOcc',    1,          @isscalar);
      
        % model serch window size
        ip.addParameter('modelDs',    [96 36],          @ismatrix); % [96 36]caltech [288 108] inria
        ip.addParameter('modelDsPad',    [120 60],          @ismatrix);% [120 60] caltech [480 240]  %inria
        ip.addParameter('nWeak',    [32 512 1024 2048 4096],          @ismatrix);
        ip.addParameter('reapply',    0,              @isscalar);% if set to 1, it will rerun the test
        
    case 'deepped' % HoG, LDCF, ACF
        ip.addParameter('modelName',    'LdcfCaltechDetector',        @ischar);
        ip.addParameter('ldcfPath', 'deepped/toolbox/detector/models/', @ischar);
        ip.addParameter('svmPath', 'deepped/rcnn/data/rcnn_models/DeepPed/SVM_finetuned_alexnet.mat', @ischar);
        ip.addParameter('alexnetPath', 'deepped/rcnn/data/rcnn_models/DeepPed/finetuned_alexNet.mat', @ischar);
        ip.addParameter('svmlvl2Path', 'deepped/rcnn/data/rcnn_models/DeepPed/SVM_level2.mat', @ischar);
        ip.addParameter('useGpu',    true,              @islogical);% for gpu image resize in matlab
        ip.addParameter('reapply',    0,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('bbSupThreshold',    2,              @isscalar);% if set to 1, it will rerun the test

    case 'frcnn' % VGG16 ZF
        
        ip.addParameter('modelName',    'VOC0712',        @ischar);
        ip.addParameter('networkType',    'zf',        @ischar);%Models: vgg and zf(shallower network)
        ip.addParameter('useGpu',   true,              @islogical);% 0 = cpu, 1 = gpu
       
        % This tag is to ignore the bounding boxes with occlusion tag. This is particularly useful
        % for JAAD which has two sets of bounding boxes part and full. 
        % For datasets with visibility region (Caltech and Cityperson) set
        % this paramter to 0
        ip.addParameter('gpuId',    1,              @isscalar);%
        ip.addParameter('detVOC', true,             @islogical); % set true voc model with 20 classes is used
        
        % Parameters for ground truth selection
        ip.addParameter('squarify',      {{3,.41}},             @iscell); %set the ratio of bounding boxes 
        ip.addParameter('hRng',    [50 inf],          @ismatrix);
        ip.addParameter('vRng',    [.65 1],          @ismatrix);
        ip.addParameter('ignOcc',    1,          @isscalar);
        ip.addParameter('nms_overlap_thres',    0.5,              @isscalar);
        ip.addParameter('regenRoiFile',    0,              @isscalar);% Regenerate ROI files
        ip.addParameter('regenImdbFile',    0,              @isscalar);% Regenerate imdb files
        ip.addParameter('anchor_scales',     2.^[3:5],      @ismatrix);
        ip.addParameter('anchor_ratios',      [0.5, 1, 2],      @ismatrix);
        ip.addParameter('detectionThresh',    0.001,              @isscalar);% 0.6 for visualization
        ip.addParameter('bbSupThreshold',     2,              @isscalar);
        
        %new parameters
        ip.addParameter('scales',   600,              @isscalar);        % Image scales -- the short edge of input image
        ip.addParameter('max_size',   1000,              @isscalar);    % Max pixel size of a scaled input image
        ip.addParameter('batch_size',       256,            @isscalar);%120 %0.5
        ip.addParameter('fg_fraction',   0.5,              @isscalar); %1/6   % Max pixel size of a scaled input image
        ip.addParameter('fg_thresh',       0.7,            @isscalar); %0.5
        ip.addParameter('bg_thresh_hi',    0.3,            @isscalar);%0.5
        ip.addParameter('bg_thresh_lo',    0,              @isscalar);
        ip.addParameter('image_means',     128,      @ismatrix);% 256
        ip.addParameter('use_flipped',     false,           @islogical);%false
        ip.addParameter('feat_stride',    16,              @isscalar);
        ip.addParameter('val_interval',    2000,              @isscalar);
        ip.addParameter('doVal',    false,              @islogical);
        ip.addParameter('reapply',    0,              @isscalar);% if set to 1, it will rerun the test
        
        %rpn test setting
        ip.addParameter('per_nms_topN',    -1,              @isscalar);
        ip.addParameter('after_nms_topN',    2000,              @isscalar);
        ip.addParameter('nmsThresh',    0.7,              @isscalar);
        ip.addParameter('rpn_solverstate',    '',        @ischar);%Models: vgg and zf(shallower network)
        ip.addParameter('rcnn_solverstate',    '',        @ischar);%Models: vgg and zf(shallower network)
        
        %Test parameters
        ip.addParameter('test_scales',   600,              @isscalar);    % Max pixel size of a scaled input image
        ip.addParameter('test_max_size',   1000,              @isscalar);    % Max pixel size of a scaled input image
        ip.addParameter('test_nms',        0.3,            @isscalar);%0.5
        ip.addParameter('debug',               false,        @islogical);% Show images after detection
        
        %layer Names
        ip.addParameter('fcnn_bbox_pred',    'bbox_pred',        @ischar);%Models: vgg and zf(shallower network)
        ip.addParameter('fcnn_cls_score',    'cls_score',        @ischar);%Models: vgg and zf(shallower network)
        ip.addParameter('rpn_bbox_pred',    'proposal_bbox_pred',        @ischar);%Models: vgg and zf(shallower network)
        ip.addParameter('rpn_cls_score',    'proposal_cls_score',        @ischar);%Models: vgg and zf(shallower network)
        %others
        ip.addParameter('rpnOnly', false,             @islogical); % only train the first stage rpn
         
    case 'ldcf+' % LDCF++, ACF++
        ip.addParameter('modelName',    'LDCF++_ldcf',        @ischar);       % Available models:  LDCF++, ACF++
        
        % Parameters for ground truth selection
        ip.addParameter('squarify',      {{3,.41}},             @iscell); %set the ratio of bounding boxes 
        ip.addParameter('hRng',    [50 inf],          @ismatrix);
        ip.addParameter('vRng',    [1 1],          @ismatrix);
        ip.addParameter('xRng',    [],          @ismatrix);
        ip.addParameter('yRng',    [],          @ismatrix);
        % This tag is to ignore the bounding boxes with occlusion tag. This is particularly useful
        % for JAAD which has two sets of bounding boxes part and full. 
        % For datasets with visibility region (Caltech and Cityperson) set
        % this paramter to 0
        ip.addParameter('ignOcc',    1,          @isscalar);
        
        % model serch window size
        ip.addParameter('modelDs',    [50 20.5],          @ismatrix); % [50 20.5] caltech [100 41] inria
        ip.addParameter('modelDsPad',    [64 32],          @ismatrix);% [64 32] caltech [128 64]  %inria
        
        % Training parameters
        ip.addParameter('nWeak',    [64 256 1024 4096],          @ismatrix);% [64 256 1024 4096] caltech [32 128 512 2048] inria
        ip.addParameter('cascCal',    0.025,              @isscalar);% 0.025 caltech 0.01 inria
        ip.addParameter('calSetup',    1,              @isscalar);% 0 for inria and jaad
        ip.addParameter('octupSet',    1,              @isscalar);
        ip.addParameter('useLDCF',    0,              @isscalar);%Models:  0 = ACF++ 1 = LDCF++
        ip.addParameter('modelPath', 'output', @ischar); % path to read or write models
        ip.addParameter('reapply',    0,              @isscalar);% if set to 1, it will rerun the test
  
    case 'ms-cnn'
        ip.addParameter('modelName',    'mscnn',        @ischar);
        
        % Parameters for ground truth selection
        ip.addParameter('squarify',      {},             @iscell); %set the ratio of bounding boxes 
        ip.addParameter('hRng',    [30 inf],          @ismatrix);
        ip.addParameter('vRng',    [0.5 1],          @ismatrix);
        ip.addParameter('xRng',    [],          @ismatrix);
        ip.addParameter('yRng',    [],          @ismatrix);
        ip.addParameter('ignOcc',    1,          @isscalar);
        ip.addParameter('reapply',    0,              @isscalar);% if set to 1, it will rerun the test
        
        % This tag is to ignore the bounding boxes with occlusion tag. This is particularly useful
        % for JAAD which has two sets of bounding boxes part and full. 
        % For datasets with visibility region (Caltech and Cityperson) set
        % this paramter to 0
        ip.addParameter('gpuId',    1,              @isscalar);
        ip.addParameter('useGpu',    true,              @islogical);
        
        %training image sizes
        ip.addParameter('scales',    720,          @isscalar);
        ip.addParameter('max_size',    960,          @isscalar);
        ip.addParameter('test_scales',    720,          @isscalar);
        ip.addParameter('test_max_size',    960,          @isscalar);
        ip.addParameter('do_bb_norm',    1,          @isscalar);% set 0 only for the pretrained model. default is 0
        ip.addParameter('showResults',    0,          @isscalar);% show detection results
        ip.addParameter('showWindowData',    0,          @isscalar);% show detection results
        ip.addParameter('regenWindowFile',   1,              @isscalar);
        ip.addParameter('trainProto',    'ms-cnn/jaad/models/',        @ischar);
        ip.addParameter('phase1SolverState',    '',        @ischar);
        ip.addParameter('phase2SolverState',    '',        @ischar);
        ip.addParameter('phase1Only',    false,        @islogical);
        ip.addParameter('phase2Only',    false,        @islogical);
        ip.addParameter('trainRootDir',    'ms-cnn/jaad/models/',        @ischar);
  
        ip.addParameter('modelPath', 'output', @ischar); % path to read or write models

    case 'rpn-bf'
        ip.addParameter('modelName',    'VGG16_jaad_final',        @ischar);
        ip.addParameter('trainModelName',    'VGG16_jaad',        @ischar); %Don't change this
        ip.addParameter('detectionThresh',    0,              @isscalar);% 0.6 for visualization
        ip.addParameter('bbSupThreshold',     2,              @isscalar);
        ip.addParameter('nmsThresh',    0.7,              @isscalar);
        % This tag is to ignore the bounding boxes with occlusion tag. This is particularly useful
        % for JAAD which has two sets of bounding boxes part and full. 
        % For datasets with visibility region (Caltech and Cityperson) set
        % this paramter to 0
        ip.addParameter('gpuId',    1,              @isscalar);
        
        ip.addParameter('useGpu',    false,              @islogical);% for gpu image resize in matlab
        ip.addParameter('useCaffeGpu',    true,              @islogical);% for conv model
        ip.addParameter('boxGpu',    true,              @islogical);% for box filer
        ip.addParameter('batch_size',      120,            @isscalar);
        
        % Parameters for ground truth selection
        ip.addParameter('squarify',      {{3,.41}},             @iscell); %set the ratio of bounding boxes 
        ip.addParameter('hRng',    [50 inf],          @ismatrix);
        ip.addParameter('vRng',    [.65 1],          @ismatrix);
        ip.addParameter('xRng',    [],          @ismatrix);
        ip.addParameter('yRng',    [],          @ismatrix);
        ip.addParameter('ignOcc',    1,          @isscalar);
        ip.addParameter('vRngBBT',    [1 1],          @ismatrix);
        
        ip.addParameter('regenRoiFile',    0,              @isscalar);% Regenerate ROI files
        ip.addParameter('regenImdbFile',    0,              @isscalar); % Regenerate imdb files
        ip.addParameter('flip',    false,              @islogical);
        ip.addParameter('thresh',    0.5,              @isscalar);
        ip.addParameter('doVal',    false,              @islogical);
        ip.addParameter('reapply',    0,              @isscalar);% if set to 1, it will rerun the test
        
        %train rpn params
        ip.addParameter('scales',    720,          @isscalar);
        ip.addParameter('max_size',   960,          @isscalar);
        ip.addParameter('anchor_scales',     2.6*(1.3.^(0:8)),      @ismatrix);
        %test rpn params
        ip.addParameter('test_scales',    720,          @isscalar);
        ip.addParameter('test_max_size',    960,          @isscalar);
        %others
        ip.addParameter('bfOnly', false, @islogical);
        ip.addParameter('rpnOnly', false, @islogical);
  
    case 'sds-rcnn'
        
        %testing parameters
        ip.addParameter('modelName',    'rpn',        @ischar);
        ip.addParameter('rpnConf',    'rpn_conf.mat',        @ischar);
        ip.addParameter('rcnnConf',    'rcnn_conf.mat',        @ischar);
        ip.addParameter('anchors',    'anchors.mat',        @ischar);
        ip.addParameter('bbMeans',    'bbox_means.mat',        @ischar);
        ip.addParameter('bbStds',    'bbox_stds.mat',        @ischar);
        ip.addParameter('rpnOnlyWeightsPath',    'final_RPN_cost_off',        @ischar);
        ip.addParameter('rpnWeightsPath',    'final_RPN',        @ischar);
        ip.addParameter('rcnnWeightsPath',    'final_BCN',        @ischar);
        ip.addParameter('rpnProtoPath',    'sds-rcnn/models/rpn/VGG16_weak_seg/test.prototxt',        @ischar);
        ip.addParameter('rcnnProtoPath',    'sds-rcnn/models/rcnn/VGG16_weak_seg/test.prototxt',        @ischar);
        
        ip.addParameter('modelRootDir',    'sds-rcnn/output/',        @ischar);%'SDS-RCNN/detector/models/'
        ip.addParameter('useGpu',    true,              @islogical);% for gpu image resize in matlab
        
        ip.addParameter('networkType',    'rpn-bcn',        @ischar);% network types: rpn (only), rpn-bcn
        
        ip.addParameter('squarify',      {{3,.41}},             @iscell); %set the ratio of bounding boxes 
        ip.addParameter('hRng',    [50 inf],          @ismatrix);
        ip.addParameter('vRng',    [1 1],          @ismatrix);
        ip.addParameter('minGt',    30,              @isscalar);%minimum gt height to train on
        ip.addParameter('regenRoiFile',    0,              @isscalar);% Regenerate ROI files
        ip.addParameter('regenImdbFile',    0,              @isscalar);% Regenerate imdb files
        ip.addParameter('reapply',    0,              @isscalar);% if set to 1, it will rerun the test
        % This tag is to ignore the bounding boxes with occlusion tag. This is particularly useful
        % for JAAD which has two sets of bounding boxes part and full. 
        % For datasets with visibility region (Caltech and Cityperson) set
        % this paramter to 0
        ip.addParameter('gpuId',    1,              @isscalar);
        ip.addParameter('showPlot',    0,              @isscalar);
        ip.addParameter('rpnSolverState',    '',        @ischar);
        ip.addParameter('rcnnSolverState',    '',        @ischar);
        ip.addParameter('ignOcc',    1,          @isscalar);
                
        % rcnn setup
        ip.addParameter('lr',    0.001,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('step_size',    60000,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('max_iter',    120000,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('snapshot_iter',    10000,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('batch_size',    120,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('crop_size',    [112 112],              @ismatrix);% if set to 1, it will rerun the test
        ip.addParameter('train_batch_size',    20,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('test_batch_size',    15,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('rcnn_feat_stride',    16,          @isscalar);
        ip.addParameter('regenCacheRcnn',    0,              @isscalar);% if set to 1, it will rerun the test
        
        % rpn solver
        ip.addParameter('lr_rpn',    0.001,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('step_size_rpn',    60000,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('max_iter_rpn',    120000,              @isscalar);% if set to 1, it will rerun the test
        ip.addParameter('snapshot_iter_rpn',    10000,              @isscalar);% if set to 1, it will rerun the test
        
        %rpn training
        ip.addParameter('scales',    720,          @isscalar);
        ip.addParameter('max_size',    1800,          @isscalar);
        ip.addParameter('anchor_scales',    1.6*(1.385.^(0:8)),      @ismatrix);
        ip.addParameter('rpn_model',    'VGG16_weak_seg',        @ischar);
        ip.addParameter('rpn_feat_stride',    16,          @isscalar);
        ip.addParameter('base_anchor_size',    16,          @isscalar);
        
        
        %finetuning params
        ip.addParameter('cls_score_sm',    'cls_score_sm',              @ischar);% fused parameter
        ip.addParameter('cls_score2_sm',    'cls_score2_sm',              @ischar);%rcnn parameter
        
        %network Setting
        ip.addParameter('batch_size_rpn',    120,              @isscalar);%
   
    case 'spp'
        ip.addParameter('modelName',    'spp',        @ischar);        ip.addParameter('modelDs',    [50 20.5],          @ismatrix); % [50 20.5] caltech [100 41] inria
        ip.addParameter('imageSize',    [480 640],          @ismatrix); % [50 20.5] caltech [100 41] inria
        ip.addParameter('octave',    8,              @isscalar);
        ip.addParameter('stride',    4,              @isscalar);
        ip.addParameter('bingThresh',    -0.032,              @isscalar);
        ip.addParameter('pedTresh',    -0.5,              @isscalar);
        ip.addParameter('reapply',    0,              @isscalar);
  
    case 'evalonly'
        ip.addParameter('modelName',    'eval',        @ischar);
    otherwise
        error('%s is not a valid algorithm',algName);
end

ip.parse(varargin{:});
modelSetup = ip.Results;
modelSetup.algorithmName = algName;


