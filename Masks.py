import numpy as np
from PIL import Image
from PIL import ImageDraw
from MaskSys import MaskObj

def getMask(Q):

    for data in Q:
        src_tri_coords = np.array(data[0])
        dst_v = data[1]
        Q = data[2]
        cropped_img = data[3]

        arr_img = np.asarray(cropped_img)
        shape = arr_img.shape

        z = np.full((shape[0],shape[1],1),255 , dtype=np.uint8)
        arr_img = np.concatenate((arr_img , z), axis=2)
        cropped_img = Image.fromarray(arr_img)

        dst_tri = np.array([dst_v[0], dst_v[1], dst_v[2]])
        bb2 = calcBoundingBoxTri(dst_tri)
        origin_2 = [bb2[0][0], bb2[0][1]]

        bb = calcBoundingBoxTri(src_tri_coords)
        origin_1 = [bb[0][0], bb[0][1]]

        b_r = realign(dst_v, origin_2)
        a_r = realign(src_tri_coords, origin_1)

        M = np.array([
            [b_r[0][0], b_r[0][1], 1, 0, 0, 0],
            [b_r[1][0], b_r[1][1], 1, 0, 0, 0],
            [b_r[2][0], b_r[2][1], 1, 0, 0, 0],
            [0, 0, 0, b_r[0][0], b_r[0][1], 1],
            [0, 0, 0, b_r[1][0], b_r[1][1], 1],
            [0, 0, 0, b_r[2][0], b_r[2][1], 1]
        ])

        y = np.array([a_r[0][0], a_r[1][0], a_r[2][0], a_r[0][1], a_r[1][1], a_r[2][1]])


        A = np.linalg.solve(M, y)

        dst_tri = (tuple(dst_tri[0]), tuple(dst_tri[1]), tuple(dst_tri[2]))
        origin_dst = [bb2[0][0], bb2[0][1]]

        dst_img_size = getlenXY(bb2)


        transformed = cropped_img.transform((dst_img_size[0], dst_img_size[1]), Image.AFFINE, A, resample=Image.BICUBIC, fillcolor=(0,0,0,0))

        adjusted_tri = realign(dst_tri, origin_dst)

        adjusted_tri = (tuple(adjusted_tri[0]), tuple(adjusted_tri[1]), tuple(adjusted_tri[2]))

        mask = Image.new('1', dst_img_size)
        maskdraw = ImageDraw.Draw(mask)
        maskdraw.polygon(adjusted_tri, fill=255)

        Q.put(MaskObj.MaskObj(mask, transformed, origin_dst, bb2, dst_tri))




def getPixelCoords(wrld_xyz, intrinsics_mtrx, extrinsics_mtrx):

    a = np.matmul(intrinsics_mtrx, extrinsics_mtrx)

    b = np.matmul(a, wrld_xyz)

    if (b[2] != 1):
        b[0] = b[0] / b[2]
        b[1] = b[1] / b[2]

    return b[:2]

def realign(tri, origin):
    new_tri = []
    for x in tri:
        temp = []
        for y in range(len(origin)):
            temp.append(x[y] - origin[y])
        new_tri.append(temp)
    return new_tri

def calcBoundingBoxTri(triangle):

    c1 = triangle[0]
    c2 = triangle[1]
    c3 = triangle[2]

    xmin = min(c1[0],c2[0],c3[0])
    ymin = min(c1[1],c2[1],c3[1])
    xmax = max(c1[0],c2[0],c3[0])
    ymax = max(c1[1],c2[1],c3[1])

    return [[(int)(xmin),(int)(ymin)],[(int)(xmax),(int)(ymax)]]

def getlenXY(bb):
    return [(int) (bb[1][0]-bb[0][0])+1, (int)(bb[1][1]-bb[0][1])+1]
