function window_file = generateTrainScript(trainPhase,varargin)

ip = inputParser;
ip.addParameter('caffePath',          'ms-cnn/caffe/build/tools/caffe',          @isstr);
ip.addParameter('trainRootDir',          'ms-cnn/output',          @isstr);
ip.addParameter('outputRoot', 'jaad_720_960', @isstr);
ip.addParameter('gpu', 0, @isscalar);

if strcmpi(trainPhase, 'phase1')
    ip.addParameter('weights',          'ms-cnn/pretrained/VGG_ILSVRC_16_layers.caffemodel',   @isstr);
    ip.addParameter('solverState',  '', @isstr);
    ip.addParameter('solverName', 'solver_1st', @isstr);
    ip.addParameter('logName',        'log_1st',       @isstr);
    
elseif strcmpi(trainPhase, 'phase2')
    ip.addParameter('weights',          'phase1/snapshot/train',   @isstr);
    ip.addParameter('solverState',  '', @isstr);
    ip.addParameter('solverName', 'solver_2nd', @isstr);
    ip.addParameter('logName',        'log_2nd',       @isstr);
end

ip.parse(varargin{:});
opts = ip.Results;

log_path = fullfile(opts.trainRootDir, opts.outputRoot, trainPhase);
window_file = sprintf(fullfile(log_path , ['train_' trainPhase '.sh']));
fid = fopen(window_file, 'wt');

if isempty(opts.solverState)
    weights = opts.weights;
else
    weights = (fullfile(log_path , 'snapshot',[opts.solverState '.solverstate']));
end

fprintf(fid, '#!/usr/bin/env sh\n');
fprintf(fid, 'TOOLS=%s\n', opts.caffePath);
fprintf(fid, 'GLOG_logtostderr=1 $TOOLS train \\\n');
fprintf(fid, '--solver=%s \\\n',fullfile(log_path,[opts.solverName,'.prototxt']));
fprintf(fid, '--weights=%s \\\n',weights);
fprintf(fid, '--gpu=%d ',opts.gpu);
fprintf(fid, '2>&1 | tee -a %s.txt\n', fullfile(log_path,opts.logName));

fclose(fid);
end

