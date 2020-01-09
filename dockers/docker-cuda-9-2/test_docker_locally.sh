# This command maps /home/dedey to /root/dedey inside the docker and drops one into 
# a terminal inside the docker. Then one can just run the training command.
docker run --runtime=nvidia -it --rm -v /home/dedey:/root/dedey debadeepta/petridishpytorchcuda92:latest 