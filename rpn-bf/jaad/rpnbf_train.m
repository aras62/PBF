function RPNBF_train(dataInfo, modelSetup)

clc;
exp_name = modelSetup.modelName;
BF_cachedir = fullfile(pwd, 'output', exp_name, 'bf_cachedir');

%check if the model exists and return
nm  =[BF_cachedir '/' exp_name 'Detector.mat'];
% if exist(nm,'file')
%     fprintf('bf model already exist at %s\n',nm); 
%     return; 
% end

%% -------------------- CONFIG --------------------
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
    'test_max_size',modelSetup.test_max_size,...
    'batch_size',120);

% set cache folder for each stage
model.stage1_rpn.cache_name = [cache_base_proposal, '/stage1_rpn']; 
% generate anchors and pre-calculate output size of rpn network 
conf_proposal.exp_name = exp_name;
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file, modelSetup.anchor_scales);

%% read the RPN model
imdbs_name = cell2mat(cellfun(@(x) x.name, dataset.imdb_train, 'UniformOutput', false));
log_dir = fullfile(pwd, 'output', exp_name, 'rpn_cachedir', model.stage1_rpn.cache_name, imdbs_name);
final_model_path = fullfile(log_dir, 'final');

if exist(final_model_path, 'file')
    model.stage1_rpn.output_model_file = final_model_path;
else
%     model.stage1_rpn.output_model_file = model.stage1_rpn.output_model_file_default;
    error('RPN model does not exist.');
end
            
%% generate proposal for training the BF
model.stage1_rpn.nms.per_nms_topN = -1;
model.stage1_rpn.nms.nms_overlap_thres = 1;
model.stage1_rpn.nms.after_nms_topN = 40;

roidb_test_BF = do_generate_bf_proposal_jaad(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);

model.stage1_rpn.nms.nms_overlap_thres = 0.7;
model.stage1_rpn.nms.after_nms_topN = 1000;
roidb_train_BF = do_generate_bf_proposal_jaad(conf_proposal, model.stage1_rpn, dataset.imdb_train{1}, dataset.roidb_train{1});


%% train the BF
mkdir_if_missing(BF_cachedir);
posGtDir= dataInfo.trainAnnotDir; 
addpath('external/code3.2.1');
addpath(genpath('external/toolbox'));
BF_prototxt_path = fullfile('models', 'VGG16_caltech', 'bf_prototxts', 'test_feat_conv34atrous_v2.prototxt');
conf.image_means = model.mean_image;
conf.test_scales = conf_proposal.test_scales;
conf.test_max_size = conf_proposal.max_size;
if ischar(conf.image_means)
    s = load(conf.image_means);
    s_fieldnames = fieldnames(s);
    assert(length(s_fieldnames) == 1);
    conf.image_means = s.(s_fieldnames{1});
end
log_dir = fullfile(BF_cachedir, 'log');
mkdir_if_missing(log_dir);
caffe_log_file_base = fullfile(log_dir, 'caffe_log');
caffe.init_log(caffe_log_file_base);
caffe_net = caffe.Net(BF_prototxt_path, 'test');
caffe_net.copy_from(final_model_path);
caffe.set_mode_gpu();

% set up opts for training detector (see acfTrain)
opts=DeepTrain_otf_trans_ratio_jaad(); 
opts.cache_dir = BF_cachedir;
opts.name=fullfile(opts.cache_dir, exp_name); %fullfile(opts.cache_dir, 'DeepCaltech_otf');
opts.nWeak=[64 128 256 512 1024 1536 2048];
opts.bg_hard_min_ratio = [1 1 1 1 1 1 1];
opts.pBoost.pTree.maxDepth=5; 
opts.pBoost.discrete=0;
opts.pBoost.pTree.fracFtrs=1/4; 
opts.first_nNeg = 30000;
opts.nNeg=5000; opts.nAccNeg=50000;

pLoad = getTags(dataInfo.dataSet);

opts.pLoad = [pLoad 'hRng', modelSetup.hRng, 'vRng',modelSetup.vRngBBT, ...
    'squarify', modelSetup.squarify];
opts.roidb_train = roidb_train_BF;
opts.roidb_test = roidb_test_BF;
opts.imdb_train = dataset.imdb_train{1};
opts.imdb_test = dataset.imdb_test;
opts.fg_thres_hi = 1;
opts.fg_thres_lo = 0.8; %[lo, hi)
opts.bg_thres_hi = 0.5;
opts.bg_thres_lo = 0; %[lo hi)
opts.dataDir = posGtDir;
opts.caffe_net = caffe_net;
opts.conf = conf;
opts.exp_name = exp_name;
opts.fg_nms_thres = 1;
opts.fg_use_gt = true;
opts.bg_nms_thres = 1;
opts.max_rois_num_in_gpu = 3000;
opts.init_detector = '';
opts.load_gt = false;
opts.ratio = 1.0;
opts.nms_thres =  modelSetup.nmsThresh;

% forward an image to check error and get the feature length
img = imread(dataset.imdb_test.image_at(1));
tic;
tmp_box = roidb_test_BF.rois(1).boxes;
retain_num = round(size(tmp_box, 1) * opts.bg_hard_min_ratio(end));
retain_idx = randperm(size(tmp_box, 1), retain_num);
sel_idx = true(size(tmp_box, 1), 1);
sel_idx = sel_idx(retain_idx);

if opts.bg_nms_thres < 1
    sel_box = roidb_test_BF.rois(1).boxes(sel_idx, :);
    sel_scores = roidb_test_BF.rois(1).scores(sel_idx, :);
    nms_sel_idxes = nms([sel_box sel_scores], opts.bg_nms_thres);
    sel_idx = sel_idx(nms_sel_idxes);
end

tmp_box = roidb_test_BF.rois(1).boxes(sel_idx, :);
feat = rois_get_features_ratio(conf, caffe_net, img, tmp_box, opts.max_rois_num_in_gpu, opts.ratio);
toc;
opts.feat_len = length(feat);

fs=bbGt('getFiles',{posGtDir});
train_gts = cell(length(fs), 1);
for i = 1:length(fs)
    [~,train_gts{i}]=bbGt('bbLoad',fs{i},opts.pLoad);
end
opts.train_gts = train_gts;

% set for test
opts.pLoad = [pLoad 'hRng', modelSetup.hRng, 'vRng',modelSetup.vRng];

if(~isempty(modelSetup.yRng)),  opts.pLoad = [opts.pLoad 'yRng', modelSetup.yRng]; end
if(~isempty(modelSetup.xRng)),  opts.pLoad = [opts.pLoad 'xRng', modelSetup.xRng]; end

% train BF detector
detector = DeepTrain_otf_trans_ratio_jaad( opts );
gather_bf_model(model);
caffe.reset_all();
end
function gather_bf_model(model)

BF_cachedir = fullfile(pwd, 'output', model.modelName, 'bf_cachedir');
BF_prototxt_path = fullfile('models', 'VGG16_caltech', 'bf_prototxts', 'test_feat_conv34atrous_v2.prototxt');
BF_prototxt_test_path = fullfile('models', 'VGG16_caltech','bf_test.prototxt');
bf_model  = [BF_cachedir '/' model.modelName 'Detector.mat'];
dest_folder = fullfile(pwd, 'output',model.modelName);

copyfile(BF_prototxt_path, dest_folder);
copyfile(BF_prototxt_test_path ,dest_folder);
copyfile(bf_model,dest_folder);

end
function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file,scales)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size_caltech(conf, test_net_def_file);
    anchors                = proposal_generate_anchors_caltech(cache_name, ...
                                    'scales',  scales, ...
                                    'ratios', [1 / 0.41], ...
                                    'exp_name', conf.exp_name);
end