function  mscnn_train(dataInfo,modelSetup)

generateWindowFile(dataInfo, modelSetup,'train');
generateWindowFile(dataInfo, modelSetup,'val');

if modelSetup.useGpu, solverMode = 'GPU'; else solverMode = 'CPU'; end;
trainModelDir = [num2str(modelSetup.scales) '_' num2str(modelSetup.max_size)];
if ~exist(fullfile('ms-cnn','jaad','models',trainModelDir), 'dir')
    error('Wrong scale or max_size values are selected'); end;

%%%%%%% phase1 %%%%%%%
disp('Starting phase 1 ...')
disp('Generating solver file....')
finalPhase1Model = generateSolver('phase1',...
    'outputRoot',modelSetup.modelName,...
    'trainModelDir',trainModelDir ,...
    'solver_mode', solverMode);
if ~exist(finalPhase1Model,'file')
    if modelSetup.phase1Only || (~modelSetup.phase1Only&& ~modelSetup.phase2Only)
        trainScript = generateTrainScript('phase1',...
            'outputRoot',modelSetup.modelName,...
            'gpu', modelSetup.gpuId-1,...
            'solverState', modelSetup.phase1SolverState);
        system(['sh ' trainScript]);
    end;
else
    disp('Phase 1 model exists, skipping to phase 2.');
end

%%%%%%% phase2 %%%%%%%
disp('Starting phase 2 ...')
disp('Generating solver file....')
finalModel = generateSolver('phase2',...
    'outputRoot',modelSetup.modelName,...
    'trainModelDir',trainModelDir ,...
    'solver_mode', solverMode);
if ~exist(finalModel,'file')
    if modelSetup.phase2Only || (~modelSetup.phase1Only&& ~modelSetup.phase2Only)
        trainScript = generateTrainScript('phase2',...
            'outputRoot',modelSetup.modelName,...
            'weights', finalPhase1Model,...
            'gpu', modelSetup.gpuId-1,...
            'solverState', modelSetup.phase2SolverState);
         system(['sh ' trainScript]);
        gather_bf_model(finalModel,trainModelDir, modelSetup.modelName, modelSetup.modelPath)
    end
else
    disp('Final model exists, terminating training.');
end

end

function gather_bf_model(finalModel,trainModelDir, modelName, modelPath)
deployModel = fullfile('ms-cnn','jaad','models',trainModelDir, 'mscnn_deploy.prototxt');
dest_folder = fullfile('ms-cnn',modelPath,modelName);
copyfile(deployModel, dest_folder);
copyfile(finalModel ,fullfile(dest_folder,'final.caffemodel'));
end