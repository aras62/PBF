function sds_train(dataInfo, modelSetup)

if modelSetup.regenImdbFile, modelSetup.regenRoiFile = 1; end;

rpn_config = config_rpn(dataInfo, modelSetup);
% train proposal
out_jargin = train_rpn_jaad(rpn_config);


%train classification
rcnn_config = config_rcnn(dataInfo, modelSetup);
train_rcnn_jaad(rcnn_config, out_jargin.output_dir, ...
        out_jargin.final_model_path)

end

