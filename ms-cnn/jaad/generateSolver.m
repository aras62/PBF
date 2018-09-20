function pathTofinalWeight = generateSolver(trainPhase,varargin)

ip = inputParser;
ip.addParameter('outputRoot', 'jaad', @isstr);
ip.addParameter('modelRootDir',          'ms-cnn/jaad/models/',          @isstr);
ip.addParameter('trainModelDir',       '720_960',  @isstr);

if strcmpi(trainPhase, 'phase1')
    ip.addParameter('train_net',          'trainval_1st',          @isstr);
    ip.addParameter('test_iter',        1000,       @isscalar);
    ip.addParameter('test_interval',        1000,       @isscalar);
    ip.addParameter('base_lr',        0.00005,       @isscalar);
    ip.addParameter('lr_policy',          'step',          @isstr);
    ip.addParameter('gamma',        0.1,       @isscalar);
    ip.addParameter('stepsize',        10000,       @isscalar);
    ip.addParameter('display',        50,       @isscalar);
    ip.addParameter('max_iter',        10000,       @isscalar);
    ip.addParameter('momentum',        0.9,       @isscalar);
    ip.addParameter('weight_decay',        0.0005,       @isscalar);
    ip.addParameter('snapshot',        10000,       @isscalar);
    ip.addParameter('snapshot_prefix',          'mscnn_jaad_p1',          @isstr);
    ip.addParameter('solver_mode',          'GPU',          @isstr);
    ip.addParameter('solverName',          'solver_1st',          @isstr);
    ip.addParameter('random_seed',        1704,       @isscalar);
    
elseif strcmpi(trainPhase, 'phase2')
    ip.addParameter('train_net',          'trainval_2nd',          @isstr);
    ip.addParameter('test_iter',        1000,       @isscalar);
    ip.addParameter('test_interval',        1000,       @isscalar);
    ip.addParameter('base_lr',        0.0002,       @isscalar);
    ip.addParameter('lr_policy',          'step',          @isstr);
    ip.addParameter('gamma',        0.1,       @isscalar);
    ip.addParameter('stepsize',        10000,       @isscalar);
    ip.addParameter('display',        50,       @isscalar);
    ip.addParameter('max_iter',        25000,       @isscalar);
    ip.addParameter('momentum',        0.9,       @isscalar);
    ip.addParameter('weight_decay',        0.0005,       @isscalar);
    ip.addParameter('snapshot',        10000,       @isscalar);
    ip.addParameter('snapshot_prefix',          'mscnn_jaad_p2',          @isstr);
    ip.addParameter('solver_mode',          'GPU',          @isstr);
    ip.addParameter('solverName',          'solver_2nd',          @isstr);
    ip.addParameter('random_seed',        1706,       @isscalar);
    
end
ip.parse(varargin{:});
opts = ip.Results;

base_dir = 'ms-cnn/output';
resultsRoot = fullfile(base_dir,opts.outputRoot);
if ~exist(resultsRoot, 'dir')
    mkdir(resultsRoot);
end

snapshotRootDir = fullfile(resultsRoot,trainPhase,'snapshot');
if ~exist(snapshotRootDir, 'dir')
    mkdir(snapshotRootDir);
end
window_file = sprintf(fullfile(resultsRoot,trainPhase,[ opts.solverName '.prototxt']));

fid = fopen(window_file, 'wt');

fprintf(fid, 'net: "%s"\n', fullfile(opts.modelRootDir,opts.trainModelDir, [opts.train_net '.prototxt']));
fprintf(fid, 'test_iter: %d\n', opts.test_iter);
fprintf(fid, 'test_interval: %d\n', opts.test_interval);
fprintf(fid, 'base_lr: %d\n', opts.base_lr);
fprintf(fid, 'lr_policy: "%s"\n', opts.lr_policy);
fprintf(fid, 'gamma: %d\n', opts.gamma);
fprintf(fid, 'stepsize: %d\n', opts.stepsize);
fprintf(fid, 'display: %d\n', opts.display);
fprintf(fid, 'max_iter: %d\n', opts.max_iter);
fprintf(fid, 'momentum: %d\n', opts.momentum);
fprintf(fid, 'weight_decay: %d\n', opts.weight_decay);
fprintf(fid, 'snapshot: %d\n', opts.snapshot);
fprintf(fid, 'snapshot_prefix: "%s"\n', fullfile(snapshotRootDir, opts.snapshot_prefix));
fprintf(fid, 'solver_mode: %s\n', opts.solver_mode);
fprintf(fid, 'random_seed: %d\n', opts.random_seed);

fclose(fid);

pathTofinalWeight = fullfile(snapshotRootDir,...
    [opts.snapshot_prefix '_iter_' num2str(opts.max_iter) '.caffemodel']); 
end

