function frcnn_train(dataInfo, modelSetup)
% script_faster_rcnn_VOC0712plus_VGG16()
% Faster rcnn training and testing with VGG16 model
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% -------------------------------------------------------
clc;
%% -------------------- CONFIG --------------------
networkType = modelSetup.networkType;
% do validation, or not
opts.do_val                 =  modelSetup.doVal ;
% model
if (strcmpi(networkType,'zf'))
    model                       = ZF_for_Faster_RCNN_Config(modelSetup);
    cache_base_proposal         = [modelSetup.modelName '_zf'];%'VOC0712_ZF';
elseif (strcmpi(networkType,'vgg'))
    model                       =  VGG16_for_Faster_RCNN_Config(modelSetup);
    cache_base_proposal         = [modelSetup.modelName '_vgg'];%'VOC0712_vgg';
else
    fprintf('%s is not a valid model.', modelSetup.networkType);return
end

cache_base_fast_rcnn        = '';

% train/test data
dataset                     = [];
dataset                     = jaad_traintest(dataset, dataInfo, 'train', modelSetup);
dataset                     = jaad_traintest(dataset, dataInfo,'test', modelSetup);
dataset                     = jaad_traintest(dataset, dataInfo,'val', modelSetup);


%% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config_jaad('image_means', model.mean_image,...
    'scales', modelSetup.scales,...
    'max_size', modelSetup.max_size,...
    'batch_size', modelSetup.batch_size,...
    'fg_fraction', modelSetup.fg_fraction,...
    'fg_thresh', modelSetup.fg_thresh,...
    'bg_thresh_hi',modelSetup.bg_thresh_hi, ...
    'bg_thresh_lo',modelSetup.bg_thresh_lo, ...
    'image_means', modelSetup.image_means, ...
    'use_flipped', modelSetup.use_flipped, ...
    'feat_stride', model.feat_stride,...
    'test_scales', modelSetup.test_scales,...
    'test_max_size',modelSetup.test_max_size,...
    'test_nms', modelSetup.test_nms,...
    'test_min_box_height' ,modelSetup.hRng(1,1),...
    'val_interval', modelSetup.val_interval,...
    'rpn_bbox_pred', modelSetup.rpn_bbox_pred, ...
    'rpn_cls_score', modelSetup.rpn_cls_score);

conf_fast_rcnn              = fast_rcnn_config('image_means', model.mean_image);
conf_fast_rcnn.fcnn_bbox_pred = modelSetup.fcnn_bbox_pred;
conf_fast_rcnn.fcnn_cls_score = modelSetup.fcnn_cls_score;
% set cache folder for each stage

model                       = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, cache_base_fast_rcnn, model);

% generate anchors and pre-calculate output size of rpn network
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
    = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file, modelSetup.anchor_scales, modelSetup.anchor_ratios);

%%  stage one proposal
fprintf('\n***************\nstage one proposal \n***************\n');
% train
model.stage1_rpn            = do_proposal_train_jaad(conf_proposal, dataset, model.stage1_rpn, opts.do_val);
% test
dataset.roidb_train         = cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test       	= Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);

if modelSetup.rpnOnly
    % save final models, for outside tester
    gather_rpn_only_model(conf_proposal, conf_fast_rcnn, model, dataset);
    return;
end;

%%  stage one fast rcnn
fprintf('\n***************\nstage one fast rcnn\n***************\n');
% train
model.stage1_fast_rcnn      = do_fast_rcnn_train_jaad(conf_fast_rcnn, dataset, model.stage1_fast_rcnn, opts.do_val);

%%  stage two proposal
% net proposal
fprintf('\n***************\nstage two proposal\n***************\n');
% train
model.stage2_rpn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_rpn            = do_proposal_train_jaad(conf_proposal, dataset, model.stage2_rpn, opts.do_val);
% test
dataset.roidb_train        	= cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test         	= Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);

%%  stage two fast rcnn
fprintf('\n***************\nstage two fast rcnn\n***************\n');
% train
model.stage2_fast_rcnn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_fast_rcnn      = do_fast_rcnn_train_jaad(conf_fast_rcnn, dataset, model.stage2_fast_rcnn, opts.do_val);

%% final test
fprintf('\n***************\nfinal test\n***************\n');

model.stage2_rpn.nms        = model.final_test.nms;
dataset.roidb_test       	= Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);

% save final models, for outside tester
Faster_RCNN_Train.gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset);
end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file,scales,anch_ratio)
[output_width_map, output_height_map] ...
    = proposal_calc_output_size_jaad(conf, test_net_def_file);
anchors                = proposal_generate_anchors(cache_name, ...
    'scales', scales, 'ratios',anch_ratio);
end
