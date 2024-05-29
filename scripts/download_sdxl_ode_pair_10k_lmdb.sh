$CHECKPOINT_PATH=$1

wget https://huggingface.co/tianweiy/DMD2/resolve/main/data/laion/sdxl_ode_pair_10k_lmdb.zip?download=true -O $CHECKPOINT_PATH/sdxl_ode_pair_10k_lmdb.zip sdxl_ode_pair_10k_lmdb.zip 

unzip $CHECKPOINT_PATH/sdxl_ode_pair_10k_lmdb.zip -d $CHECKPOINT_PATH