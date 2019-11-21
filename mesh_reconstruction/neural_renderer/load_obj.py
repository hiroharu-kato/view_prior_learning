import os

import numpy as np
import imageio


def load_mtl(filename_mtl):
    # load color (Kd) and filename of textures from *.mtl
    material_name = ''
    materials = {}
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                command = line.split()[0]
                if command == 'newmtl':
                    material_name = line.split()[1]
                    materials[material_name] = {}
                elif command == 'map_Kd':
                    materials[material_name]['texture_filename'] = line.split()[1]
                elif command == 'Kd':
                    materials[material_name]['color'] = np.array(list(map(float, line.split()[1:4])))
    return materials


def load_textures_func(filename_obj, filename_mtl):
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        for line in f.readlines():
            if len(line.split()) != 0 and line.split()[0] == 'vt':
                vertices.append([float(v) for v in line.split()[1:3]])
    vertices = np.vstack(vertices).astype('float32')

    # load faces for textures
    faces = []
    material_names = []
    material_name = ''
    with open(filename_obj) as f:
        for line in f.readlines():
            if len(line.split()) == 0:
                continue
            elif line.split()[0] == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                if '/' in vs[0]:
                    v0 = int(vs[0].split('/')[1])
                else:
                    v0 = 0
                for i in range(nv - 2):
                    if '/' in vs[i + 1]:
                        v1 = int(vs[i + 1].split('/')[1])
                    else:
                        v1 = 0
                    if '/' in vs[i + 2]:
                        v2 = int(vs[i + 2].split('/')[1])
                    else:
                        v2 = 0
                    faces.append((v0, v1, v2))
                    material_names.append(material_name)
            elif line.split()[0] == 'usemtl':
                material_name = line.split()[1]
    faces = np.vstack(faces).astype('int32') - 1

    # load mtl file
    materials = load_mtl(filename_mtl)

    # load textures
    pos = 0
    textures = np.zeros((3, 0, 0), 'float32')
    for material_name, material in materials.items():
        # load texture
        if 'texture_filename' in material:
            texture = imageio.imread(os.path.join(os.path.dirname(filename_mtl), material['texture_filename']))
            texture = texture.astype('float32') / 255.
            texture = texture.transpose((2, 0, 1))
            texture = texture[:, ::-1, ::1]

            # modify texture vertices
            indices = np.unique(faces[np.array(material_names) == material_name].flatten())
            vertices[indices, 0] *= texture.shape[2] - 1  # x
            vertices[indices, 1] *= texture.shape[1] - 1  # y
            vertices[indices, 1] += pos
        else:
            color = material['color']
            texture = np.ones((3, 2, 2), 'float32') * np.array(color)[:, None, None]
            vertices = np.concatenate((vertices, np.zeros((3, 2), 'float32')), axis=0)
            vertices[-3, 0] = 0
            vertices[-3, 1] = pos
            vertices[-2, 0] = 0
            vertices[-2, 1] = pos + 1
            vertices[-1, 0] = 1
            vertices[-1, 1] = pos + 1
            faces[np.array(material_names) == material_name] = np.array(
                [vertices.shape[0] - 3, vertices.shape[0] - 2, vertices.shape[0] - 1])

        pos += texture.shape[1]
        if textures.shape[2] < texture.shape[2]:
            textures = np.concatenate(
                (textures, np.zeros((3, textures.shape[1], texture.shape[2] - textures.shape[2]))),
                axis=2,
            )
        elif texture.shape[2] < textures.shape[2]:
            texture = np.concatenate(
                (texture, np.zeros((3, texture.shape[1], textures.shape[2] - texture.shape[2]))),
                axis=2,
            )
        textures = np.concatenate((textures, texture), axis=1)
        textures = textures.astype('float32')

    return vertices, faces, textures


def load_obj(filename_obj, normalization=True, load_textures=False):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        for line in f.readlines():
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'v':
                vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype('float32')

    # load faces
    faces = []
    with open(filename_obj) as f:
        for line in f.readlines():
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                v0 = int(vs[0].split('/')[0])
                for i in range(nv - 2):
                    v1 = int(vs[i + 1].split('/')[0])
                    v2 = int(vs[i + 2].split('/')[0])
                    faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype('int32') - 1

    # load textures
    textures = None
    if load_textures:
        with open(filename_obj) as f:
            for line in f.readlines():
                if line.startswith('mtllib'):
                    filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
                    vertices_t, faces_t, textures = load_textures_func(filename_obj, filename_mtl)
        if textures is None:
            raise Exception('Failed to load textures.')

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[None, :]
        vertices /= np.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[None, :] / 2

    if load_textures:
        return vertices, faces, vertices_t, faces_t, textures
    else:
        return vertices, faces
