import collections
import json
import sys
import time
import shutil
import subprocess
import os
import multiprocessing
from multiprocessing import Process, Queue, Manager, Pool, shared_memory
import queue as qq
from PIL import Image
from PIL import ImageOps
import Masks as CT
import pickle as pkl
import objReader
import numpy as np
from datetime import datetime
from MaskSys import MaskQueue
import sys
import random
from numba import jit, njit  # , cuda, float32
from numba import prange as pr
from numba.typed import List, Dict
import scipy
from scipy import spatial
import math
cameras = {}
intrinsics = {}
poses = {}
mesh = None
from operator import itemgetter
import Grouper
from pyoctree import pyoctree as ot

meshroom_sfm_file_location = None
meshroom_sfm = None
mesh_location = None
mesh_obj = None
xatlas_dir = None
sfm_json = None

@jit(nopython=True)
def calcBoundingBoxTri(c1,c2,c3):

    xmin = min([c1[0],c2[0],c3[0]])
    ymin = min([c1[1],c2[1],c3[1]])
    xmax = max([c1[0],c2[0],c3[0]])
    ymax = max([c1[1],c2[1],c3[1]])

    return np.array([[(int)(xmin[0]),(int)(ymin[0])],[(int)(xmax[0]),(int)(ymax[0])]], dtype=np.float64)


@njit
def matmult(matrix1,matrix2):

    x = matrix1.shape[0]
    y = matrix2.shape[1]
    rmatrix = np.zeros(shape=(x,y), dtype=np.float64)

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                rmatrix[i][j] += matrix1[i][k] * matrix2[k][j]
    return rmatrix

@njit
def getPixelCoords(wrld_xyz, intrinsics_mtrx, extrinsics_mtrx):

    a = matmult(intrinsics_mtrx, extrinsics_mtrx)

    b = matmult(a, wrld_xyz)

    if (b[2] != 1):
        b[0] = b[0] / b[2]
        b[1] = b[1] / b[2]

    return b[:2]

@njit
def checkfit(face_coords_np, im, em, h, w):
    source = List()
    count = -1
    for y in face_coords_np:
        count+=1
        xyz1 = np.zeros(shape=(4,1), dtype=np.float64)
        xyz1[0] = y[0]
        xyz1[1] = y[1]
        xyz1[2] = y[2]
        xyz1[3] = 1

        c = getPixelCoords(xyz1, im, em)
        source.append(c)

        if c[0] < 0:
            return 0
        elif c[1] < 0:
            return 0
        elif (c[1] > h):
            return 0
        elif (c[0] > w):
            return 0

    bb = calcBoundingBoxTri(source[0],source[1],source[2])
    if (bb[0][0] == bb[1][0]):
        return 0

    elif (bb[0][1] == bb[1][1]):
        return 0

    return 1

@njit(fastmath=True)
def isbetween(face_coords_np, c, y):
    for z in face_coords_np:
        if( np.sqrt(np.square(y[0]-c[0]) + np.square(y[1]-c[1]) + np.square(y[2]-c[2])) < np.sqrt(np.square(z[0]-c[0]) + np.square(z[1]-c[1]) + np.square(z[2]-c[2]))):
            return True
    return False

def selectViewTaskV3(data):
    args = data[0]
    cameras = data[1]

    poses = data[2]

    intrinsics = data[3]

    occluded = data[4]

    cam_score2 = np.zeros(len(cameras))
    results = []

    status = []
    for z in args:
        status.append([])

    for z in status:
        for y in cameras:
            z.append(None)
    count2 = -1
    cr = -1
    for z in args:
        cr += 1
        if(cr==10):
            cr=0
        count2 += 1
        face_coords_np = np.array(z[0], dtype=np.float64)
        count = -1
        for cam in cameras:
            count += 1
            status[count2][count] = checkfit(face_coords_np,
                                            np.array(intrinsics[cameras[cam][1]][0], dtype=np.float64),
                                            np.array(poses[cameras[cam][0]][0], dtype=np.float64),
                                            intrinsics[cameras[cam][1]][2][1],
                                            intrinsics[cameras[cam][1]][2][0])

            angle = checkAngle2(face_coords_np, np.array(poses[cameras[cam][0]][2]))
            correctstr = str(sorted(z[1]))
            if status[count2][count] == 1:
                if (cam in occluded[correctstr]):
                    cam_score2[count] += -3
                else:
                    cam_score2[count] += -angle
            else:
                cam_score2[count] += -2

    ####################################################

    count2 = -1
    for z in args:
        count2 += 1
        min = -1
        face = z[1]
        face_coords_np = np.array(z[0], dtype=np.float64)

        count = -1
        for cam in cameras:
            count += 1

            # if(int(occluded[count])==1):
            #     count+=1
            #     continue
            # else:
            #     count+=1

            if status[count2][count] == 0:
                continue
            angle = checkAngle2(face_coords_np, np.array(poses[cameras[cam][0]][2]))
            score = cam_score2[count]*angle
            if min == -1:
                min = [score,  cam, face, count]
            elif score < min[0]:
                min = [score, cam, face, count]


        results.append(min)

    return results

@njit
def dist(p1,p2):
    d = p2-p1
    dist = np.sqrt(np.square(d[0]) + np.square(d[1]) + np.square(d[2]))
    return dist

@njit
def dist2(c,y):
    return np.sqrt(np.square(y[0] - c[0]) + np.square(y[1] - c[1]) + np.square(y[2] - c[2]))

@njit
def cent(triangle):
    c = np.zeros(3)
    c[0] = (triangle[0][0] + triangle[1][0] + triangle[2][0])/3
    c[1] = (triangle[0][1] + triangle[1][1] + triangle[2][1])/3
    c[2] = (triangle[0][2] + triangle[1][2] + triangle[2][2])/3
    return c

def occlussionProcess(vert2, face3, poses, face, face2, cameras):
    p = Dict()
    for x in poses:
        p[x] = np.array(poses[x][1], dtype=np.float32)
    face3 = np.array(face3, dtype=np.int32)
    tree = ot.PyOctree(vert2, face2)
    occluded = {}
    cnt = -1
    for f in face3:
        sortedstr = str(sorted(face[cnt]))
        cnt += 1
        for cam in cameras:
            id = cameras[cam][0]
            ray = getRays(vert2, f, p[id])
            i = []
            interPoints = []
            inter = tree.rayIntersection(ray)
            for intersections in inter:
                interPoints.append([intersections.p, tree.polyList[intersections.triLabel].N])
            rv = unit_vector_points(ray[0], ray[1])
            for it in interPoints:
                add = 1
                if (np.dot(it[1], rv) < 0):
                    if isbetween(np.array([vert2[f[0]],vert2[f[1]],vert2[f[2]]]), np.array(p[id]), np.array(it[0])):
                        add = 1
                    else:
                        add = 0
                    # if(add==1):
                    #     for vv in f:
                    #         if (it[2] == vert2[vv]).all():
                    #             add = 0
                    #             break
                    if (add == 1):
                       i.append(it[0])

            if(len(i)>1):
                if sortedstr in occluded:
                    occluded[sortedstr].append(cam)
                else:
                    occluded[sortedstr] = [cam]
            else:
                if sortedstr in occluded:
                    continue
                else:
                    occluded[sortedstr] = []

    return occluded

def groupMesh(mesh):
    #face_c = mesh.faces_coords()
    face = mesh.faces()

    vert = mesh.mesh_vertices()

    max_p = 10
    max_p2 = 10
    batches = 6
    batch_size = max_p*batches
    parts = 15
    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Creating Partitions....")
    grouper = Grouper.partition(vert, face, parts)
    partitions = grouper.getParts()
    pkl.dump(partitions, open("partitions.p", "wb"))
    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Done")
    partitions = pkl.load(open("partitions.p", "rb"))
    parts = len(partitions)
    vert_parts = []


    for x in range(parts):
        vert_parts.append([])

    for x in range(parts):
        for y in partitions[x]:
            v = [[vert[y[0]-1],  vert[y[1]-1], vert[y[2]-1]], np.array(y, dtype=int)]
            vert_parts[x].append(v)

    results = []

    face2 = []

    for x in face:
        face2.append([x[0] - 1, x[1] - 1, x[2] - 1])
    face2 = np.array(face2, dtype=np.int32)
    vert2 = np.array(vert, dtype=float)
    pool2 = Pool(max_p2)
    args3 = []
    faceparts = []
    face3 = []
    for x in range(max_p2):
        faceparts.append([])
        face3.append([])

    count3 = 0
    for x in range(len(face2)):
        faceparts[count3].append(face2[x])
        face3[count3].append(face[x])
        count3+=1
        if(count3==max_p2):
            count3=0

    for x in range(max_p2):
        args3.append([vert2, faceparts[x], poses, face3[x], face2, cameras])

    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Calculating Occlussions....")
    occs = pool2.starmap(occlussionProcess,args3)
    pool2.close()
    pool2.join()
    pkl.dump(occs, open("occs.p", "wb"))
    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Done Occlussion Check")

    occluded = {}

    #occs = pkl.load(open("occs.p", "rb"))
    for x in occs:
        for y in x:
            occluded[str(y)] = x[y]

    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Assigning views to mesh...")
    itr = 0
    while itr < parts:

        args2 = []
        for x in range(batch_size):
            if(itr+x>=parts):
                break
            args2.append([])

        for x in range(batch_size):
            if(itr+x>=parts):
                break
            args2[x].append([vert_parts[itr+x], cameras, poses, intrinsics, occluded])

        pool = Pool(max_p, maxtasksperchild=batches)
        tempResults = pool.starmap(selectViewTaskV3,args2)
        pool.close()
        pool.join()
        results.extend(tempResults)
        itr += batch_size
    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Done")

    new_results = []
    for x in results:
        for z in x:
            new_results.append(z)
    pkl.dump(new_results, open("meshToView.p", "wb"))


@njit
def getRays(vert, f, c):
    rays = np.zeros(shape=(2, 3), dtype=np.float32)
    v = np.zeros(shape=(1, 3), dtype=np.float32)
    rays[0][0] = c[0]
    rays[0][1] = c[1]
    rays[0][2] = c[2]
    v[0] = vert[f[0]]
    v[1] = vert[f[1]]
    v[2] = vert[f[2]]
    rays[1] = cent(v)
    return rays

@jit(nopython=True)
def surface_normal_newell(poly):

    n = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    for i, v_curr in enumerate(poly):
        v_next = poly[(i+1) % len(poly),:]
        n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    n = n / np.linalg.norm(n)
    return n

@jit(nopython=True)
def checkAngle(mesh, clv):
    v1_u = surface_normal_newell(mesh)
    v2_u = clv

    rad = np.arccos(np.dot(v1_u, v2_u))
    degrees = math.degrees(rad)
    return degrees

@njit
def checkAngle2(mesh, clv):
    v1_u = surface_normal_newell(mesh)
    v2_u = clv
    return np.dot(v1_u, v2_u)

@njit
def unit_vector(vector):
    magnitude = np.sqrt(np.square(vector[0]) + np.square(vector[1]) + np.square(vector[2]))
    return np.divide(vector, magnitude)

@njit
def unit_vector_points(p1, p2):
    vector = p2-p1
    magnitude = np.sqrt(np.square(vector[0]) + np.square(vector[1]) + np.square(vector[2]))
    return np.divide(vector, magnitude)

def consumer(queue, closed, size):

    qqq = MaskQueue
    qqq.start(size)

    while True:
        if queue.qsize() <= 0 and closed.get() == 1:
            qqq.close()
            break
        else:
            try:
                mask = queue.get_nowait()

                qqq.addMask(mask)

                del mask

            except qq.Empty:
                pass

    textureMap = qqq.join()
    textureMap = ImageOps.flip(textureMap)
    textureMap.save("texture.png", "PNG")
    qqq.kill()


def createTextureMap(s):
    lst = pkl.load(open("meshToView.p", "rb"))
    Q = Manager().Queue()
    size = s
    meshes = mesh.faces()
    vertices = mesh.mesh_vertices()
    uv = mesh.uv_vertices()
    closed = Manager().Value('1', 0)

    pictures = {}

    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Loading Images...")
    for i in cameras:
        pictures[i] = Image.open(cameras[i][2], 'r')

    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Done")

    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Loading Data to Memory...")


    lst = sorted(lst, key=itemgetter(1))

    args = []
    p_max = 4
    current_img_id = -1
    p_iter = 0

    for x in range(p_max):
        args.append([])

    for m in lst:
        imgId = m[1]

        if(imgId != current_img_id):
         if(current_img_id!=-1):
             del pictures[current_img_id]
         pictures[imgId].load()
         current_img_id = imgId

        src_v = []
        dst_v = []

        for y in m[2]:
            src_v.append(vertices[y-1])
            tmp = []

            for z in uv[y-1]:
                n = z * size

                #if(np.mod(n,1) < 0.5):
                #    n = (int)(n)

                tmp.append(n)
            dst_v.append(tmp)

        src_tri_coords = []

        for v in src_v:
            xyz1 = np.array([v[0],v[1],v[2],1], dtype=np.float64)

            im = np.array(intrinsics[cameras[imgId][1]][0], dtype=np.float64)
            em = np.array(poses[cameras[imgId][0]][0], dtype=np.float64)
            #dist = np.array(intrinsics[cameras[imgId][1]][1], dtype=np.float64)
            n = CT.getPixelCoords(xyz1, im, em)
            #n = CT.undistortcoords(n,im,dist)
            src_tri_coords.append(n)

        bb = CT.calcBoundingBoxTri(np.array(src_tri_coords))
        crop_size = ((bb[0][0]), (bb[0][1]), (bb[1][0]), (bb[1][1]))

        args[p_iter].append([src_tri_coords, dst_v, Q, pictures[imgId].crop(crop_size)])
        p_iter += 1
        if p_iter == p_max:
            p_iter = 0

    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Done")
    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Total Triangles: " + str(len(meshes)))


    del pictures
    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Creating Masks...")


    #del args
    del vertices
    del meshes

    ppl = Pool(p_max, maxtasksperchild=5000)
    ppl.imap(CT.getMask, args)

    ppl.close()

    ppl.join()

    consumer_process = Process(target=consumer, args=(Q, closed, size))
    consumer_process.start()
    closed.set(1)

    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Done...")
    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Creating Map...")


    consumer_process.join()



def Start(m,s,v):
    global mesh
    global poses
    global cameras
    global intrinsics
    texturesize = v
    meshroom_sfm_file_location = s
    meshroom_sfm = open(meshroom_sfm_file_location, "r")
    mesh_location = m
    mesh_obj = open(mesh_location, "r")
    sfm_json = json.loads(meshroom_sfm.read())
    xatlas_dir = sys.path[0] + "\\" + "xatlas\\"
    mesh = objReader
    print("#########################################################")
    print(datetime.now().strftime("%H:%M:%S") + ": Creating UV...")

    shutil.copy2(mesh_location,sys.path[0])
    subprocess.run([xatlas_dir+"meshuv","mesh.obj"],stdout=open(os.devnull, 'wb'))
    os.remove("mesh.obj")
    print(datetime.now().strftime("%H:%M:%S") + ": Done")

    print(datetime.now().strftime("%H:%M:%S") + ": Loading .OBJ...")

    mesh.readOBJ("output.obj")

    print(datetime.now().strftime("%H:%M:%S") + ": Done")

    print(datetime.now().strftime("%H:%M:%S") + ": Processing Camera Parameters...")
    for x in sfm_json['views']:
        cameras[x['viewId']] = [x['poseId'], x['intrinsicId'], x['path']]

    for x in sfm_json['intrinsics']:
        mat = [[float(x['pxFocalLength']), 0, float(x['principalPoint'][0]),0],
               [0, float(x['pxFocalLength']), float(x['principalPoint'][1]),0],
               [0, 0, 1,0]]

        dist = np.array([float(x['distortionParams'][0]),float(x['distortionParams'][1]), 0, 0, float(x['distortionParams'][2])],dtype=np.float32)
        size = ((int)(x['width']),(int)(x['height']))
        intrinsics[x['intrinsicId']] = [mat, dist, size]

    for x in sfm_json['poses']:
        mat = [[float(x['pose']['transform']['rotation'][0]), float(x['pose']['transform']['rotation'][1]),
                float(x['pose']['transform']['rotation'][2])],
               [float(x['pose']['transform']['rotation'][3]), float(x['pose']['transform']['rotation'][4]),
                float(x['pose']['transform']['rotation'][5])],
               [float(x['pose']['transform']['rotation'][6]), float(x['pose']['transform']['rotation'][7]),
                float(x['pose']['transform']['rotation'][8])]]

        c = [float(x['pose']['transform']['center'][0]),
             float(x['pose']['transform']['center'][1]),
             float(x['pose']['transform']['center'][2])]

        mat = np.transpose(np.array(mat))

        t = -1 * np.matmul(mat,c)

        new_em = []
        for y in range(3):
            temp = []
            for z in range(4):
                if (z < 3):
                    temp.append(mat[y][z])
                else:
                    temp.append(t[y])
            new_em.append(temp)
        new_em.append([0,0,0,1])
        mat = new_em

        poses[x['poseId']] = [mat,c,0]

    to_remove = []
    for x in cameras:
        try:
            principal = [[intrinsics[cameras[x][1]][0][0][2]], [intrinsics[cameras[x][1]][0][1][2]], [1]]

            ext = poses[cameras[x][0]][0]

            temp = np.matmul(intrinsics[cameras[x][1]][0], ext)

            principal[0] -= temp[0][3]
            principal[1] -= temp[1][3]
            principal[2] -= temp[2][3]

            p_world = np.linalg.solve(temp[:3, :3], principal)

            cameras[x] = [cameras[x][0], cameras[x][1], cameras[x][2], p_world]
        except:
            to_remove.append(x)

    for x in to_remove:
        cameras.pop(x, None)

    for x in cameras:
        lv = np.subtract(np.transpose(cameras[x][3]), poses[cameras[x][0]][1])
        lv = lv[0]
        lv = unit_vector(lv)
        poses[cameras[x][0]][2] = np.array(lv, dtype=np.float32)


    print(datetime.now().strftime("%H:%M:%S") + ": Done")

    print(datetime.now().strftime("%H:%M:%S") + ": Processing Meshes...")
    groupMesh(mesh)
    print(datetime.now().strftime("%H:%M:%S") + ": Done")

    print(datetime.now().strftime("%H:%M:%S") + ": Creating Texture Map...")
    createTextureMap(texturesize)
    print(datetime.now().strftime("%H:%M:%S") + ": Done")
    print(datetime.now().strftime("%H:%M:%S") + ": File saved as Texture.png in root folder")
