function dataset = jaad_traintest(dataset,dataInfo,usage, modelSetup)
% Pascal voc 2012 test set
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install


switch usage
    case {'train'}
        dataset.imdb_train    = {imdb_from_data(dataInfo, 'train', modelSetup) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x,modelSetup,dataInfo), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_data(dataInfo, 'test', modelSetup) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test,modelSetup,dataInfo);
    otherwise
        error('usage = ''train'' or ''test''');
end

end
