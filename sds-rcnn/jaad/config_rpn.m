function conf = config_rpn(dataInfo,modelSetup)

%model name and dataset
conf.model                  =  modelSetup.rpn_model;%'VGG16_weak_seg';
conf.dataset_train          =  dataInfo.trainDir;
conf.dataset_test           =  dataInfo.testDir;
conf.dataset_val            =  dataInfo.valDir;

% solver
conf.solver_type            =  'SGD';
conf.lr                     =  modelSetup.lr_rpn;
conf.step_size              =  modelSetup.step_size_rpn;
conf.max_iter               =  modelSetup.max_iter_rpn;%120000
conf.snapshot_iter          =  modelSetup.snapshot_iter_rpn; %10000

%Extras
conf.dataInfo               =  dataInfo;
conf.dataInfo.regenImdbFile =  modelSetup.regenImdbFile;
conf.dataInfo.regenRoiFile  =  modelSetup.regenRoiFile;
conf.dataInfo.regenCache    =  modelSetup.regenRoiFile || modelSetup.regenImdbFile;
conf.config_name            =  modelSetup.modelName; % [ modelSetup.modelName '_VGG16_weak_seg']
conf.vRng                   =  modelSetup.vRng;
conf.show_plot              =  modelSetup.showPlot;
conf.gpu_id                 =  modelSetup.gpuId;
conf.solverstate            =  modelSetup.rpnSolverState;
conf.ignOcc                 =  modelSetup.ignOcc;
conf.squarify               =  modelSetup.squarify;
% general
conf.display_iter           =  1000;
conf.rng_seed               =  3;
conf.mat_rng_seed           =  3;
conf.scales                 =  modelSetup.scales;
conf.max_size               =  modelSetup.max_size;
conf.bg_thresh_hi           =  0.5;
conf.bg_thresh_lo           =  0;
conf.fg_thresh              =  0.5;
conf.pretrained             =  'vgg16.caffemodel';
conf.image_means            = [123.6800, 116.7790, 103.9390];

% network settings
conf.has_weak               =  true; % has weak segmentation?
conf.feat_stride            =  modelSetup.rpn_feat_stride;%16;   % network stride
conf.cost_sensitive         =  true; % use cost sensitive
conf.cost_mean_height       =  50;   % cost sensitive mean
conf.fg_image_ratio         =  0.5;  % percent fg images
conf.batch_size             =  modelSetup.batch_size_rpn;  % number fg boxes
conf.fg_fraction            =  1/5;  % percent fg boxes

% anchors
conf.anchor_scales          =  modelSetup.anchor_scales;%1.6*(1.385.^(0:8));
conf.anchor_ratios          =  0.41;
conf.base_anchor_size       =  modelSetup.base_anchor_size ;

%% testing
conf.test_min_box_height    =  modelSetup.hRng(1,1);           % min box height to keep
conf.test_min_box_size      =  16;           % min box size to keep (w || h)
conf.nms_per_nms_topN       =  10000;        % boxes before nms
conf.nms_overlap_thres      =  0.5;          % nms threshold IoU
conf.nms_after_nms_topN     =  40;           % boxes after nms
%   conf.test_db                = 'UsaTest';     % dataset to test with
%   conf.val_db                 = 'UsaTrainVal'; % dataset to test with
conf.min_gt_height          =  modelSetup.minGt;           % smallest gt to train on
conf.test_min_h             =  modelSetup.hRng(1,1);           % database setting for min gt



conf.image_means = reshape(conf.image_means, [1 1 3]);

end
