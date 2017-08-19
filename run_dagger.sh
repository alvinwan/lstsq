export n_atari=$1
export n_dagger=$1

for i in `seq 0 5`
do
    source $ENV3 && \
    echo "========== Trying $i ==========" && \
    cd /data/alvin/tensorpack/examples/A3C-Gym/ && \
    python dagger.py --n_atari=$n_atari --n_dagger=$n_dagger $i && \
    DAGGER=True python train-atari.py --env SpaceInvaders-v0 --task play-ls --load /data/alvin/models/SpaceInvaders-v0.tfmodel --i $i --N $n_atari --N_d $n_dagger
done
