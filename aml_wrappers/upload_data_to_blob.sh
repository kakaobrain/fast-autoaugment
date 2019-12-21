export AZCOPY_CRED_TYPE=Anonymous;
azcopy copy "/home/dedey/DeyPortable1/torchvision_data_dir/imagenet/" "https://petridishdata.blob.core.windows.net/datasets/?se=2019-10-20T21%3A56%3A32Z&sp=rwl&sv=2018-03-28&sr=c&sig=vyMmPb2epiRkpq63G%2FMTlIFnZANEjyjEsIy2DqbJqAk%3D" --overwrite=false --follow-symlinks --recursive --from-to=LocalBlob --blob-type=BlockBlob --put-md5;
unset AZCOPY_CRED_TYPE;