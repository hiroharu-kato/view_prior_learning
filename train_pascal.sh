export NI=80000
export LI=1000

export CID="aeroplane"
export PARAMS="-cls ${CID} -li ${LI} -ni ${NI} -linf 0.00003 -lp 0.01 -ld 0.5 -sm 1"
python mesh_reconstruction/train.py -ds pascal -eid pascal_proposed_color_airplane_0 $PARAMS -rs 0

export CID="car"
export PARAMS="-cls ${CID} -li ${LI} -ni ${NI} -linf 0.00003 -lp 0.01 -ld 0.5 -sm 1"
python mesh_reconstruction/train.py -ds pascal -eid pascal_proposed_color_car $PARAMS -rs 0

export CID="chair"
export PARAMS="-cls ${CID} -li ${LI} -ni ${NI} -linf 0.00003 -lp 0.01 -ld 0.5 -sm 1"
python mesh_reconstruction/train.py -ds pascal -eid pascal_proposed_color_chair $PARAMS -rs 0

export NI=5000
export LI=1000
export CID="aeroplane"
export PARAMS="-cls ${CID} -li ${LI} -ni ${NI} -linf 0.00003 -lp 0.01 -sm 1"
python mesh_reconstruction/train.py -ds pascal -eid pascal_baseline_color_airplane $PARAMS -rs 0

export CID="car"
export PARAMS="-cls ${CID} -li ${LI} -ni ${NI} -linf 0.00003 -lp 0.01 -sm 1"
python mesh_reconstruction/train.py -ds pascal -eid pascal_baseline_color_car $PARAMS -rs 0

export CID="chair"
export PARAMS="-cls ${CID} -li ${LI} -ni ${NI} -linf 0.00003 -lp 0.01 -sm 1"
python mesh_reconstruction/train.py -ds pascal -eid pascal_baseline_color_chair $PARAMS -rs 0