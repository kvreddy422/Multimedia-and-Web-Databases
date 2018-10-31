export PYTHONPATH=/root/tensorD:$PYTHONPATH


run_cp_ex() {
    N1=$1
    N2=$2
    N3=$3
    gR=$4
    dR=$5
    time=0
    while [ "$time" != 10 ]
    do
        time=$(($time+1))
        python3 cp_ex.py $N1 $N2 $N3 $gR $dR $time
    done
}

run_ncp_ex() {
    N1=$1
    N2=$2
    N3=$3
    gR=$4
    dR=$5
    time=0
    while [ "$time" != 10 ]
    do
        time=$(($time+1))
        python3 ncp_ex.py $N1 $N2 $N3 $gR $dR $time
    done
}

run_ntucker_ex() {
    N1=$1
    N2=$2
    N3=$3
    gR=$4
    dR=$5
    time=0
    while [ "$time" != 10 ]
    do
        time=$(($time+1))
        python3 ntucker_ex.py $N1 $N2 $N3 $gR $dR $time
    done
}





N=20
while [ "$N" != 160 ]
do
    gR=10
    dR=10
    run_cp_ex $N $N $N $gR $dR
    run_ncp_ex $N $N $N $gR $dR
    run_ntucker_ex $N $N $N $gR $dR


    gR=20
    dR=20
    run_cp_ex $N $N $N $gR $dR
    run_ncp_ex $N $N $N $gR $dR
    run_ntucker_ex $N $N $N $gR $dR

    N=$(($N*2))
done




