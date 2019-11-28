# Best model on ShapeNet
# VPL, texture prediction, #views=20
python ./mesh_reconstruction/test.py -ds shapenet -nt 0 -eid shapenet_multi_color_nv20_uvr_cc_long

# single: #views = 1
# multi: 1 < #views
# nvx: #view = x
# color: learning with silhouettes & colors
# silhouettes: learning with only silhouettes
# uvr: with VPL
# cc: with class conditioning
# long: more training iterations
python ./mesh_reconstruction/test.py -ds shapenet -nt 0 -eid shapenet_multi_color_nv20_uvr_cc
python ./mesh_reconstruction/test.py -ds shapenet -nt 0 -eid shapenet_multi_color_nv20
python ./mesh_reconstruction/test.py -ds shapenet -nt 0 -eid shapenet_multi_color_nv2_uvr_cc
python ./mesh_reconstruction/test.py -ds shapenet -nt 0 -eid shapenet_multi_color_nv2
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_multi_sil_nv20_uvr_cc_long
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_multi_sil_nv20_uvr_cc
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_multi_sil_nv10_uvr_cc
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_multi_sil_nv5_uvr_cc
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_multi_sil_nv3_uvr_cc
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_multi_sil_nv2_uvr_cc
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_multi_sil_nv20
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_multi_sil_nv10
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_multi_sil_nv5
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_multi_sil_nv3
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_multi_sil_nv2
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_single_sil_nv1
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_single_sil_nv1_uvr
python ./mesh_reconstruction/test.py -ds shapenet -eid shapenet_single_sil_nv1_uvr_cc
python ./mesh_reconstruction/test.py -ds shapenet -nt 0 -eid shapenet_single_color_nv1
python ./mesh_reconstruction/test.py -ds shapenet -nt 0 -eid shapenet_single_color_nv1_uvr
python ./mesh_reconstruction/test.py -ds shapenet -nt 0 -eid shapenet_single_color_nv1_uvr_cc
