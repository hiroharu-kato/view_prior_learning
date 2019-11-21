import os
import imageio


def save_obj(filename, vertices, faces, vertices_t=None, faces_t=None, textures=None):
    assert vertices.ndim == 2
    assert faces.ndim == 2

    if textures is not None:
        filename_mtl = filename[:-4] + '.mtl'
        filename_texture = filename[:-4] + '.png'
        material_name = 'material_1'
        textures = textures[:, ::-1, :]
        imageio.imwrite(filename_texture, textures.transpose((1, 2, 0)))

    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('\n')

        if textures is not None:
            f.write('mtllib %s\n\n' % os.path.basename(filename_mtl))

        for vertex in vertices:
            f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
        f.write('\n')

        if textures is not None:
            vertices_t[:, 0] /= (textures.shape[2] - 1)
            vertices_t[:, 1] /= (textures.shape[1] - 1)
            for vertex in vertices_t.reshape((-1, 2)):
                f.write('vt %.8f %.8f\n' % (vertex[0], vertex[1]))
            f.write('\n')

            f.write('usemtl %s\n' % material_name)
            for i, (face, face_t) in enumerate(zip(faces, faces_t)):
                f.write('f %d/%d %d/%d %d/%d\n' % (
                    face[0] + 1, face_t[0] + 1, face[1] + 1, face_t[1] + 1, face[2] + 1, face_t[2] + 1))
            f.write('\n')
        else:
            for face in faces:
                f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))

    if textures is not None:
        with open(filename_mtl, 'w') as f:
            f.write('newmtl %s\n' % material_name)
            f.write('map_Kd %s\n' % os.path.basename(filename_texture))
