####################################################################################################
# Single-view

# silhouettes, w/o regularization
python mesh_reconstruction/train.py -eid shapenet_single_sil_nv1 -dd $DD -ni 50000 -ds shapenet -svt 1 -nv 1 -sm 1 

# silhouettes, w/ regularization
python mesh_reconstruction/train.py -eid shapenet_single_sil_nv1_uvr -dd $DD -ni 100000 -ds shapenet -svt 1 -nv 1 -ld 0.1 -sm 1  
python mesh_reconstruction/train.py -eid shapenet_single_sil_nv1_uvr_cc -dd $DD -ni 100000 -ds shapenet -svt 1 -nv 1 -ld 0.2 -cc 1 -sm 1 

# color, w/o regularization
python mesh_reconstruction/train.py -eid shapenet_single_color_nv1 -dd $DD -ni 50000 -ds shapenet -svt 1 -nv 1 -lp 0.5 -sm 1 

# color, w/ regularization
python mesh_reconstruction/train.py -eid shapenet_single_color_nv1_uvr -dd $DD -ni 100000 -ds shapenet -svt 1 -nv 1 -lp 0.5 -ld 2 -sm 1 
python mesh_reconstruction/train.py -eid shapenet_single_color_nv1_uvr_cc -dd $DD -ni 100000 -ds shapenet -svt 1 -nv 1 -lp 0.5 -ld 2 -cc 1 -sm 1  

####################################################################################################
# Multi-view

# silhouettes
python mesh_reconstruction/train.py -eid shapenet_multi_sil_nv20 -dd $DD -ni 500000 -ds shapenet -sm 1  
python mesh_reconstruction/train.py -eid shapenet_multi_sil_nv10 -dd $DD -ni 250000 -ds shapenet -nv 10 -sm 1 
python mesh_reconstruction/train.py -eid shapenet_multi_sil_nv5 -dd $DD -ni 125000 -ds shapenet -nv 5 -sm 1 
python mesh_reconstruction/train.py -eid shapenet_multi_sil_nv3 -dd $DD -ni 75000 -ds shapenet -nv 3 -sm 1 
python mesh_reconstruction/train.py -eid shapenet_multi_sil_nv2 -dd $DD -ni 50000 -ds shapenet -nv 2 -sm 1 

# silhouettes + regularization
python mesh_reconstruction/train.py -eid shapenet_multi_sil_nv20_uvr_cc_long -dd $DD -ni 1000000 -ds shapenet -ld 0.03 -cc 1 -sm 1
python mesh_reconstruction/train.py -eid shapenet_multi_sil_nv20_uvr_cc -dd $DD -ni 500000 -ds shapenet -ld 0.03 -cc 1 -sm 1
python mesh_reconstruction/train.py -eid shapenet_multi_sil_nv10_uvr_cc -dd $DD -ni 500000 -ds shapenet -nv 10 -ld 0.03 -cc 1 -sm 1
python mesh_reconstruction/train.py -eid shapenet_multi_sil_nv5_uvr_cc -dd $DD -ni 250000 -ds shapenet -nv 5 -ld 0.03 -cc 1 -sm 1
python mesh_reconstruction/train.py -eid shapenet_multi_sil_nv3_uvr_cc -dd $DD -ni 150000 -ds shapenet -nv 3 -ld 0.03 -cc 1 -sm 1
python mesh_reconstruction/train.py -eid shapenet_multi_sil_nv2_uvr_cc -dd $DD -ni 100000 -ds shapenet -nv 2 -ld 0.03 -cc 1 -sm 1

# color
python mesh_reconstruction/train.py -eid shapenet_multi_color_nv20 -dd $DD -ni 500000 -ds shapenet -lp 0.1 -sm 1 
python mesh_reconstruction/train.py -eid shapenet_multi_color_nv2 -dd $DD -ni 50000 -ds shapenet -lp 0.1 -nv 2 -sm 1  

# color + regularization
python mesh_reconstruction/train.py -eid shapenet_multi_color_nv20_uvr_cc_long -dd $DD -ni 1000000 -ds shapenet -lp 0.1 -nv 20 -ld 0.3 -cc 1 -sm 1
python mesh_reconstruction/train.py -eid shapenet_multi_color_nv20_uvr_cc -dd $DD -ni 500000 -ds shapenet -lp 0.1 -nv 20 -ld 0.3 -cc 1 -sm 1
python mesh_reconstruction/train.py -eid shapenet_multi_color_nv2_uvr_cc -dd $DD -ni 100000 -ds shapenet -lp 0.1 -nv 2 -ld 0.3 -cc 1 -sm 1

