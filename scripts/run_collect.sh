source $ENV3 && \
cd .. && \
python train-atari.py --task play --load /data/alvin/models/SpaceInvaders-v0.tfmodel --env SpaceInvaders-v0
