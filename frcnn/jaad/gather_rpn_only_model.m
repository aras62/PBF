function gather_rpn_only_model(conf_proposal, conf_fast_rcnn, model, dataset)
    cachedir = fullfile(pwd, 'output', 'faster_rcnn_final', model.final_model.cache_name);
    mkdir_if_missing(cachedir);
    
    % find latest model for rpn and fast rcnn
    [rpn_test_net_def_file, rpn_output_model_file] = find_last_output_model_file(model.stage1_rpn);
    proposal_detection_model.classes = dataset.imdb_test.classes;
    proposal_detection_model.image_means = conf_proposal.image_means;
    proposal_detection_model.conf_proposal = conf_proposal;
    
    % copy rpn and fast rcnn models into cachedir
    [~, test_net_proposal_name, test_net_proposal_ext] = fileparts(rpn_test_net_def_file);
    proposal_detection_model.proposal_net_def = ['proposal_', test_net_proposal_name, test_net_proposal_ext];
    [~, proposal_model_name, proposal_model_ext] = fileparts(rpn_output_model_file);
    proposal_detection_model.proposal_net = ['proposal_', proposal_model_name, proposal_model_ext];
    copyfile(rpn_test_net_def_file, fullfile(cachedir, proposal_detection_model.proposal_net_def));
    copyfile(rpn_output_model_file, fullfile(cachedir, proposal_detection_model.proposal_net));
    save(fullfile(cachedir, 'model'), 'proposal_detection_model');
end

function [test_net_def_file, output_model_file] = find_last_output_model_file(stage1)
    if isfield(stage1, 'output_model_file') && exist(stage1.output_model_file, 'file')
        output_model_file = stage1.output_model_file;
        test_net_def_file = stage1.test_net_def_file;
        return;
    end
    error('find_last_output_model_file:: no trained models');
end