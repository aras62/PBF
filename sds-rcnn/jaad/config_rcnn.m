function conf = config_rcnn(dataInfo, modelSetup)

conf.model                  =  'VGG16_weak_seg';
conf.config_name            =  modelSetup.modelName;%[modelSetup.modelName '_VGG16_weak_seg'];
conf.dataInfo               =  dataInfo;
conf.dataInfo.regenImdbFile =  modelSetup.regenImdbFile;
conf.dataInfo.regenRoiFile  =  modelSetup.regenRoiFile;
if modelSetup.regenImdbFile || modelSetup.regenRoiFile
    conf.dataInfo.regenCache = 1;
else
    conf.dataInfo.regenCache    =  modelSetup.regenCacheRcnn;
end
conf.vRng                   =  modelSetup.vRng;
conf.ignOcc                 =  modelSetup.ignOcc;
conf.squarify               =  modelSetup.squarify;
conf.dataset_train          =  dataInfo.trainDir ;
conf.dataset_test           =  dataInfo.testDir ;
conf.dataset_val            =  dataInfo.valDir;
conf.show_plot              =  modelSetup.showPlot;
conf.gpu_id                 =  modelSetup.gpuId;
conf.solverstate            =  modelSetup.rcnnSolverState;

%   conf.ext                    =  '.jpg';

% solver
conf.solver_type            =  'SGD';
conf.lr                     =  modelSetup.lr;
conf.step_size              =  modelSetup.step_size;
conf.max_iter               =  modelSetup.max_iter;%120000
conf.snapshot_iter          =  modelSetup.snapshot_iter; %10000

% general
conf.display_iter           =  1000;
conf.rng_seed               =  3;
conf.mat_rng_seed           =  3;
conf.fg_thresh              =  0.7;
conf.image_means            = [123.6800, 116.7790, 103.9390];
% network settings
conf.train_batch_size       =  modelSetup.train_batch_size;        % number of proposals train
conf.test_batch_size        =  modelSetup.test_batch_size;        % number of proposals test
conf.crop_size              =  modelSetup.crop_size; % size of images
conf.has_weak               =  true;      % has weak segmentation?
conf.weak_seg_crop          =  [7 7];     % weak segmentation size
conf.feat_stride            =  modelSetup.rcnn_feat_stride;        % network stride
conf.cost_sensitive         =  true;      % use cost sensitive
conf.cost_mean_height       =  50;        % cost sensitive mean
conf.fg_image_ratio         =  0.5;       % percent fg images
conf.batch_size             =  modelSetup.batch_size; % number fg boxes
conf.natural_fg_weight      =  true;      % ignore fg_fraction!
conf.fg_fraction            =  1/5;       % percent fg boxes
conf.feat_scores            =  true;      % fuse feature scores of rpn
conf.padfactor              =  0.2;       % percent padding

conf.cls_score_sm = modelSetup.cls_score_sm;
conf.cls_score2_sm = modelSetup.cls_score2_sm;


%% testing
%   conf.test_db              = 'UsaTest';     % dataset to test with
%   conf.val_db               = 'UsaTrainVal'; % dataset to test with
conf.min_gt_height          =  modelSetup.minGt;           % smallest gt to train on
conf.test_min_h             =  modelSetup.hRng(1,1);           % database setting for min gt

conf.image_means = reshape(conf.image_means, [1 1 3]);

end
