# category-specific models
python ./mesh_reconstruction/test.py -ds pascal -c aeroplane -eid pascal_proposed_color_airplane
python ./mesh_reconstruction/test.py -ds pascal -c car -eid pascal_proposed_color_car
python ./mesh_reconstruction/test.py -ds pascal -c chair -eid pascal_proposed_color_chair

python ./mesh_reconstruction/test.py -ds pascal -c aeroplane -eid pascal_baseline_color_airplane
python ./mesh_reconstruction/test.py -ds pascal -c car -eid pascal_baseline_color_car
python ./mesh_reconstruction/test.py -ds pascal -c chair -eid pascal_baseline_color_chair
