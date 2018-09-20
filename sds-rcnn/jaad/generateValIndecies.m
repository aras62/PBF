function val_indecies = generateValIndecies(numValData,testDir)


[imgNms,~] = bbGt('getFiles',{testDir});

val_indecies = datasample([1:length(imgNms)], numValData);


end