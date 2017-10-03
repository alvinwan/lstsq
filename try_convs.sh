for i in 1 2 3 4 5 6 7 8 9
do
export a=$[148-$i]
python convify.py 1 SpaceInvaders-v0 $i $a && python train.py 1_conv_$i\_$a
done
