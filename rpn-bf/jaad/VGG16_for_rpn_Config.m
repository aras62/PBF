function model = VGG16_for_rpn_Config(model)

model.mean_image                                = fullfile(pwd, 'models', 'VGG16_caltech', 'pre_trained_models', 'vgg_16layers', 'mean_image');
model.pre_trained_net_file                      = fullfile(pwd, 'models', 'VGG16_caltech', 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel');
 
% Stride in input image pixels at the last conv layer
model.feat_stride                               = 16;

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', 'VGG16_caltech', 'rpn_prototxts', 'vgg_16layers_conv3_1', 'solver_60k80k.prototxt');
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', 'VGG16_caltech', 'rpn_prototxts', 'vgg_16layers_conv3_1', 'test.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;
model.stage1_rpn.output_model_file_default = fullfile(pwd, 'output', model.modelName, 'rpn_cachedir', 'snapshots', 'final');
% fullfile(root, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'final');
% rpn test setting
model.stage1_rpn.nms.per_nms_topN               = 10000;
model.stage1_rpn.nms.nms_overlap_thres       	= model.nmsThresh;
model.stage1_rpn.nms.after_nms_topN         	= 40;
end