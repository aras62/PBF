function conf = proposal_config_jaad(varargin)
% conf = proposal_config(varargin)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    ip = inputParser;
    
    %% training
    ip.addParameter('use_gpu',         gpuDeviceCount > 0, ...            
                                                        @islogical);
                                    
    % whether drop the anchors that has edges outside of the image boundary
    ip.addParameter('drop_boxes_runoff_image', ...
                                        true,           @islogical);
    
    % Image scales -- the short edge of input image                                                                                                
    ip.addParameter('scales',          600,            @ismatrix);
    % Max pixel size of a scaled input image
    ip.addParameter('max_size',        1000,           @isscalar);
    % Images per batch, only supports ims_per_batch = 1 currently
    ip.addParameter('ims_per_batch',   1,              @isscalar);
    % Minibatch size
    ip.addParameter('batch_size',      256,            @isscalar);
    % Fraction of minibatch that is foreground labeled (class > 0)
    ip.addParameter('fg_fraction',     0.5,           @isscalar);
    % weight of background samples, when weight of foreground samples is
    % 1.0
    ip.addParameter('bg_weight',       1.0,            @isscalar);
    % Overlap threshold for a ROI to be considered foreground (if >= fg_thresh)
    ip.addParameter('fg_thresh',       0.7,            @isscalar);
    % Overlap threshold for a ROI to be considered background (class = 0 if
    % overlap in [bg_thresh_lo, bg_thresh_hi))
    ip.addParameter('bg_thresh_hi',    0.3,            @isscalar);
    ip.addParameter('bg_thresh_lo',    0,              @isscalar);
    % mean image, in RGB order
    ip.addParameter('image_means',     128,            @ismatrix);
    % Use horizontally-flipped images during training?
    ip.addParameter('use_flipped',     true,           @islogical);
    % Stride in input image pixels at ROI pooling level (network specific)
    % 16 is true for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
    ip.addParameter('feat_stride',     16,             @isscalar);
    % train proposal target only to labled ground-truths or also include
    % other proposal results (selective search, etc.)
    ip.addParameter('target_only_gt',  true,           @islogical);

    % random seed                    
    ip.addParameter('rng_seed',        6,              @isscalar);

    
    %% testing
    ip.addParameter('test_scales',     600,            @isscalar);
    ip.addParameter('test_max_size',   1000,           @isscalar);
    ip.addParameter('test_nms',        0.3,            @isscalar);
    ip.addParameter('test_binary',     false,          @islogical);
    ip.addParameter('test_min_box_size',16,            @isscalar);
    ip.addParameter('test_min_box_height',50,            @isscalar);
    ip.addParameter('test_drop_boxes_runoff_image', ...
                                        false,          @islogical);
    ip.addParameter('val_interval',2000,            @isscalar);
    ip.addParameter('rpn_bbox_pred',    'proposal_bbox_pred',        @ischar);
    ip.addParameter('rpn_cls_score',    'proposal_cls_score',        @ischar);
    
    ip.parse(varargin{:});
    conf = ip.Results;
    
    assert(conf.ims_per_batch == 1, 'currently rpn only supports ims_per_batch == 1');
    
    % if image_means is a file, load it
    if ischar(conf.image_means)
        s = load(conf.image_means);
        s_fieldnames = fieldnames(s);
        assert(length(s_fieldnames) == 1);
        conf.image_means = s.(s_fieldnames{1});
    end
end