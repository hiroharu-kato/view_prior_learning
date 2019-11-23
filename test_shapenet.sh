# Best model on ShapeNet
# VPL, texture prediction, #views=20
python ./mesh_reconstruction/test.py -eid shapenet_multi_color_nv20_uvr_cc -ds shapenet -nt 0

# single: #views = 1
# multi: 1 < #views
# nvx: #view = x
# color: learning with silhouettes & colors
# silhouettes: learning with only silhouettes
# uvr: with VPL
# cc: with class conditioning
# long: more training iterations
python ./mesh_reconstruction/test.py -eid shapenet_multi_color_nv20_uvr_cc -ds shapenet -nt 0
python ./mesh_reconstruction/test.py -eid shapenet_multi_color_nv20 -ds shapenet -nt 0
python ./mesh_reconstruction/test.py -eid shapenet_multi_color_nv2_uvr_cc -ds shapenet -nt 0
python ./mesh_reconstruction/test.py -eid shapenet_multi_sil_nv20_uvr_cc_long -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_multi_sil_nv20_uvr_cc -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_multi_sil_nv10_uvr_cc -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_multi_sil_nv5_uvr_cc -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_multi_sil_nv3_uvr_cc -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_multi_sil_nv2_uvr_cc -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_multi_sil_nv20 -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_multi_sil_nv10 -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_multi_sil_nv5 -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_multi_sil_nv3 -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_multi_sil_nv2 -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_single_sil_nv1 -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_single_sil_nv1_uvr -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_single_sil_nv1_uvr_cc -ds shapenet
python ./mesh_reconstruction/test.py -eid shapenet_single_color_nv1 -ds shapenet -nt 0
python ./mesh_reconstruction/test.py -eid shapenet_single_color_nv1_uvr -ds shapenet -nt 0
python ./mesh_reconstruction/test.py -eid shapenet_single_color_nv1_uvr_cc -ds shapenet -nt 0
