# num of views = 20
# with texture prediction

OBJECT_IDS='
    460cf3a75d8467d1bb579d1d8d989550 fbed7adcac3217912056b4bd5d870b47 a49eec529b5c44eaac00fd1150223027
    b4715a33fe6d1bb7f63ee8a34069b7c5 4231883e92a3c1a21c62d11641ffbd35 f6722edbe92ed28f37b2bb75885cfc44
    230efad5abd6b56bfcb8d8c6d4df8143 4f0173cf9b1472c37d4f87d95d70ab8 f2e592962a9df3de1d27f57617a4976d
    d9bb77735ff0963ae7e684d25d4dcaf0 f772e5b89475fd2e4719088c8e42c6ab 112cdf6f3466e35fa36266c295c27a25
    48cfd8b4bc4919e6cbc6ff5546f4ec42
'
for OID in $OBJECT_IDS
do
    EID=shapenet_multi_color_nv20_uvr_cc
    python mesh_reconstruction/make_gif.py -ds shapenet -eid $EID -nt 0 -oid $OID -vid 0
done;
