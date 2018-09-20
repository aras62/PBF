function model = ZF_for_Faster_RCNN_Config(model)

root = pwd; %'IVA/jaad';
mSetup = model;
model = [];
model.mean_image                                = fullfile(root, 'models', 'pre_trained_models', 'ZF', 'mean_image');
model.pre_trained_net_file                      = fullfile(root, 'models', 'pre_trained_models', 'ZF', 'ZF.caffemodel');
% Stride in input image pixels at the last conv layer
model.feat_stride                               = 16;

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(root, '/jaad/models', 'rpn_prototxts', 'ZF', 'solver_60k80k.prototxt');
model.stage1_rpn.test_net_def_file              = fullfile(root, '/jaad/models', 'rpn_prototxts', 'ZF', 'test.prototxt');

if isempty(mSetup.rpn_solverstate)
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;
else
model.stage1_rpn.init_net_file                  = mSetup.rpn_solverstate;
end

% rpn test setting
model.stage1_rpn.nms.per_nms_topN             	= mSetup.per_nms_topN;
model.stage1_rpn.nms.nms_overlap_thres       	= mSetup.nmsThresh;
model.stage1_rpn.nms.after_nms_topN          	= mSetup.after_nms_topN;

%% stage 1 fast rcnn, inited from pre-trained network
model.stage1_fast_rcnn.solver_def_file          = fullfile(root, '/jaad/models', 'fast_rcnn_prototxts', 'ZF', 'solver_30k60k.prototxt');
model.stage1_fast_rcnn.test_net_def_file        = fullfile(root, '/jaad/models', 'fast_rcnn_prototxts', 'ZF', 'test.prototxt');


if isempty(mSetup.rcnn_solverstate)
model.stage1_fast_rcnn.init_net_file            = model.pre_trained_net_file;
else
model.stage1_fast_rcnn.init_net_file            = mSetup.rcnn_solverstate;
end

%% stage 2 rpn, only finetune fc layers
model.stage2_rpn.solver_def_file                = fullfile(root, '/jaad/models', 'rpn_prototxts', 'ZF_fc6', 'solver_60k80k.prototxt');
model.stage2_rpn.test_net_def_file              = fullfile(root, '/jaad/models', 'rpn_prototxts', 'ZF_fc6', 'test.prototxt');

% rpn test setting
model.stage2_rpn.nms.per_nms_topN             	= mSetup.per_nms_topN;
model.stage2_rpn.nms.nms_overlap_thres       	= mSetup.nmsThresh;
model.stage2_rpn.nms.after_nms_topN           	= mSetup.after_nms_topN;

%% stage 2 fast rcnn, only finetune fc layers
model.stage2_fast_rcnn.solver_def_file          = fullfile(root, '/jaad/models', 'fast_rcnn_prototxts', 'ZF_fc6', 'solver_30k60k.prototxt');
model.stage2_fast_rcnn.test_net_def_file        = fullfile(root, '/jaad/models', 'fast_rcnn_prototxts', 'ZF_fc6', 'test.prototxt');

%% final test
model.final_test.nms.per_nms_topN              	= mSetup.per_nms_topN; % to speed up nms
model.final_test.nms.nms_overlap_thres       	= mSetup.nms_overlap_thres;
model.final_test.nms.after_nms_topN           	= mSetup.after_nms_topN;
end