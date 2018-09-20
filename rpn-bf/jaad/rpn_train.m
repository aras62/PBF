function RPN_train(dataInfo, modelSetup)
% script_rpn_pedestrian_VGG16_caltech()
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under TByrhe MIT License [see LICENSE for details]
% --------------------------------------------------------

clc;

%% -------------------- CONFIG --------------------

exp_name = modelSetup.modelName;

% do validation, or not 
opts.do_val                 = modelSetup.doVal; 
% model
model                       = VGG16_for_rpn_Config(modelSetup);
% cache base
cache_base_proposal         = ['rpn_' modelSetup.modelName];
% train/test data
dataset                     = [];
dataset                     = jaad_traintest(dataset, dataInfo, 'train', modelSetup);
dataset                     = jaad_traintest(dataset, dataInfo,'test', modelSetup);

% %% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config_caltech('image_means', model.mean_image,...
    'test_min_box_height' ,modelSetup.hRng(1,1),...
    'test_nms', modelSetup.nmsThresh, ...
    'batch_size', modelSetup.batch_size,...
    'max_size', modelSetup.max_size,...
    'scales', modelSetup.scales,...
    'test_scales', modelSetup.test_scales,...
    'test_max_size',modelSetup.test_max_size);

% set cache folder for each stage. Just sets up the names
model.stage1_rpn.cache_name = [cache_base_proposal, '/stage1_rpn'];         
% generate anchors and pre-calculate output size of rpn network 
conf_proposal.exp_name = exp_name;
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file, modelSetup.anchor_scales);

                        
%%  train
fprintf('\n***************\nstage one RPN \n***************\n');
model.stage1_rpn            = do_proposal_train_jaad(conf_proposal, dataset, model.stage1_rpn, opts.do_val);

gather_rpn_model(conf_proposal, model, dataset);

end
function gather_rpn_model(conf_proposal, model, dataset)
imdbs_name = cell2mat(cellfun(@(x) x.name, dataset.imdb_train, 'UniformOutput', false));
cache_dir =  fullfile(pwd, 'output', conf_proposal.exp_name, 'rpn_cachedir', model.stage1_rpn.cache_name, imdbs_name);
anchors_path = fullfile(pwd, 'output', conf_proposal.exp_name, 'rpn_cachedir', model.stage1_rpn.cache_name,'anchors.mat');
dest_folder = fullfile(pwd, 'output',model.modelName);
rpn_test_path = fullfile(pwd, 'models/VGG16_caltech/rpn_prototxts/vgg_16layers_conv3_1/rpn_test.prototxt');
copyfile([model.mean_image, '.mat'], dest_folder);
copyfile([cache_dir '/final'],dest_folder);
copyfile(rpn_test_path,dest_folder);
copyfile(anchors_path,dest_folder);

end
function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file, scales)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size(conf, test_net_def_file); %proposal_calc_output_size_caltech
    anchors                = proposal_generate_anchors_caltech(cache_name, ...
                                    'scales', scales,... % 2.6*(1.3.^(0:8)), ...
                                    'ratios', [1 / 0.41], ...
                                    'exp_name', conf.exp_name);
end