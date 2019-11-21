import string

import chainer
import cupy as cp


def _voxelize_sub1(faces, size):
    batch_size, num_faces = faces.shape[:2]
    faces = cp.ascontiguousarray(faces)
    voxels = chainer.cuda.elementwise(
        'int32 j, raw T faces',
        'int32 voxels',
        string.Template('''
            float z1 = i % ${vs};
            float y1 = (i / ${vs}) % ${vs};
            float x1 = (i / (${vs} * ${vs})) % ${vs};
            float x2 = x1 + 1;
            float y2 = y1;
            float z2 = z1;
            float x3 = x1;
            float y3 = y1 + 1;
            float z3 = z1;
            float x4 = x1;
            float y4 = y1;
            float z4 = z1 + 1;
            float x5 = x1 + 1;
            float y5 = y1 + 1;
            float z5 = z1;
            float x6 = x1;
            float y6 = y1 + 1;
            float z6 = z1 + 1;
            float x7 = x1 + 1;
            float y7 = y1;
            float z7 = z1 + 1;
            float x8 = x1 + 1;
            float y8 = y1 + 1;
            float z8 = z1 + 1;
            int bn = i / (${vs} * ${vs} * ${vs});

            bool ok1 = false;
            for (int fn = 0; fn < ${nf}; fn++){
                float* face = (float*)&faces[(bn * ${nf} + fn) * 9];
                float xp1 = face[0];
                float xp2 = face[3];
                float xp3 = face[6];
                float yp1 = face[1];
                float yp2 = face[4];
                float yp3 = face[7];
                float zp1 = face[2];
                float zp2 = face[5];
                float zp3 = face[8];
                
                bool ok2 = true;
                for (int ln = 0; ln < 13; ln++) {
                    float xl, yl, zl;
                    if (ln == 0) {
                        xl = 1;
                        yl = 0;
                        zl = 0;
                    } else if (ln == 1) {
                        xl = 0;
                        yl = 1;
                        zl = 0;
                    } else if (ln == 2) {
                        xl = 0;
                        yl = 0;
                        zl = 1;
                    } else {
                        float xl1, yl1, zl1, xl2, yl2, zl2;
                        if (3 <= ln && ln < 6) {
                            xl1 = 1;
                            yl1 = 0;
                            zl1 = 0;
                        } else if (6 <= ln && ln < 9) {
                            xl1 = 0;
                            yl1 = 1;
                            zl1 = 0;
                        } else if (9 <= ln && ln < 12) {
                            xl1 = 0;
                            yl1 = 0;
                            zl1 = 1;
                        }
                        if (ln % 3 == 0) {
                            xl2 = xp2 - xp1;
                            yl2 = yp2 - yp1;
                            zl2 = zp2 - zp1;
                        } else if (ln % 3 == 1) {
                            xl2 = xp3 - xp2;
                            yl2 = yp3 - yp2;
                            zl2 = zp3 - zp2;
                        } else if (ln % 3 == 2) {
                            xl2 = xp1 - xp3;
                            yl2 = yp1 - yp3;
                            zl2 = zp1 - zp3;
                        }
                        if (ln == 12) {
                            xl1 = xp2 - xp1;
                            yl1 = yp2 - yp1;
                            zl1 = zp2 - zp1;
                            xl2 = xp3 - xp2;
                            yl2 = yp3 - yp2;
                            zl2 = zp3 - zp2;
                        }
                        xl = yl1 * zl2 - yl2 * zl1;
                        yl = zl1 * xl2 - zl2 * xl1;
                        zl = xl1 * yl2 - xl2 * yl1;
                    }
                    float pp1 = xp1 * xl + yp1 * yl + zp1 * zl;
                    float pp2 = xp2 * xl + yp2 * yl + zp2 * zl;
                    float pp3 = xp3 * xl + yp3 * yl + zp3 * zl;
                    float pv1 = x1 * xl + y1 * yl + z1 * zl;
                    float pv2 = x2 * xl + y2 * yl + z2 * zl;
                    float pv3 = x3 * xl + y3 * yl + z3 * zl;
                    float pv4 = x4 * xl + y4 * yl + z4 * zl;
                    float pv5 = x5 * xl + y5 * yl + z5 * zl;
                    float pv6 = x6 * xl + y6 * yl + z6 * zl;
                    float pv7 = x7 * xl + y7 * yl + z7 * zl;
                    float pv8 = x8 * xl + y8 * yl + z8 * zl;
                    float pp_min = min(min(pp1, pp2), pp3);
                    float pp_max = max(max(pp1, pp2), pp3);
                    float pv_min = min(min(min(min(min(min(min(pv1, pv2), pv3), pv4), pv5), pv6), pv7), pv8);
                    float pv_max = max(max(max(max(max(max(max(pv1, pv2), pv3), pv4), pv5), pv6), pv7), pv8);
                    if ((pp_max < pv_min) || (pv_max < pp_min)) {
                        ok2 = false;
                        break;
                    }
                }
                if (ok2) {
                    ok1 = true;
                    break;
                }
            }
            if (ok1) {
                voxels = 1;
            } else {
                voxels = 0;
            }
        ''').substitute(
            bs=batch_size,
            nf=num_faces,
            vs=size,
        ),
        'function',
    )(cp.arange(batch_size * size * size * size).astype('int32'), faces)
    voxels = voxels.reshape((batch_size, size, size, size))

    return voxels


def _voxelize_sub4(voxels):
    # fill in
    bs, vs = voxels.shape[:2]
    voxels = cp.ascontiguousarray(voxels)
    visible = cp.zeros_like(voxels, 'int32')
    chainer.cuda.elementwise(
        'int32 j, raw int32 bs, raw int32 vs',
        'raw int32 voxels, raw int32 visible',
        '''
            int z = j % vs;
            int x = (j / vs) % vs;
            int y = (j / (vs * vs)) % vs;
            int bn = j / (vs * vs * vs);
            int pn = j;
            if ((y == 0) || (y == vs - 1) || (x == 0) || (x == vs - 1) || (z == 0) || (z == vs - 1)) {
                if (voxels[pn] == 0) visible[pn] = 1;
            }
        ''',
        'function',
    )(cp.arange(bs * vs * vs * vs).astype('int32'), bs, vs, voxels, visible)

    sum_visible = visible.sum()
    while True:
        chainer.cuda.elementwise(
            'int32 j, raw int32 bs, raw int32 vs',
            'raw int32 voxels, raw int32 visible',
            '''
                int z = j % vs;
                int x = (j / vs) % vs;
                int y = (j / (vs * vs)) % vs;
                int bn = j / (vs * vs * vs);
                int pn = j;
                if ((y == 0) || (y == vs - 1) || (x == 0) || (x == vs - 1) || (z == 0) || (z == vs - 1)) return;
                if (voxels[pn] == 0 && visible[pn] == 0) {
                    int yi, xi, zi;
                    yi = y - 1;
                    xi = x;
                    zi = z;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                    yi = y + 1;
                    xi = x;
                    zi = z;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                    yi = y;
                    xi = x - 1;
                    zi = z;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                    yi = y;
                    xi = x + 1;
                    zi = z;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                    yi = y;
                    xi = x;
                    zi = z - 1;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                    yi = y;
                    xi = x;
                    zi = z + 1;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                }
            ''',
            'function',
        )(cp.arange(bs * vs * vs * vs).astype('int32'), bs, vs, voxels, visible)
        if visible.sum() == sum_visible:
            break
        else:
            sum_visible = visible.sum()
    return 1 - visible


def voxelize(faces, size):
    faces = cp.copy(faces)
    faces *= size

    voxels = _voxelize_sub1(faces, size)
    voxels = _voxelize_sub4(voxels)

    return voxels
