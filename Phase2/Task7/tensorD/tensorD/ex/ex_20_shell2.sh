export PYTHONPATH=/root/tensorD:$PYTHONPATH



run_tucker_ex() {
    N1=$1
    N2=$2
    N3=$3
    gR=$4
    dR=$5
    time=0
    while [ "$time" != 20 ]
    do
        time=$(($time+1))
        python3 tucker_ex.py $N1 $N2 $N3 $gR $dR $time
    done
}


# N=20
N=20
gR=10
dR=16
while [ "$dR" != 21 ]
do
    run_tucker_ex $N $N $N $gR $dR
    dR=$(($dR+1))
done


# N=40
N=40
gR=10
dR=16
while [ "$dR" != 42 ]
do
    run_tucker_ex $N $N $N $gR $dR
    dR=$(($dR+2))
done


gR=20
dR=26
while [ "$dR" != 42 ]
do
    run_tucker_ex $N $N $N $gR $dR
    dR=$(($dR+2))
done



# N=80
N=80
gR=10
dR=20
while [ "$dR" != 85 ]
do
    run_tucker_ex $N $N $N $gR $dR
    dR=$(($dR+5))
done


gR=20
dR=30
while [ "$dR" != 85 ]
do
    run_tucker_ex $N $N $N $gR $dR
    dR=$(($dR+5))
done
