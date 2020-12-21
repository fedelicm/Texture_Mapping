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

import pymetis
from PIL import Image
from PIL import ImageOps
import Masks as CT
import pickle as pkl
import objReader
import numpy as np
from datetime import datetime
from MaskSys import MaskQueue
meshroom_sfm_file_location = None
meshroom_sfm = None
mesh_location = None
mesh_obj = None
xatlas_dir = None
sfm_json = None
import sys
#import numba
from numba import jit, njit  # , cuda, float32
import scipy
from scipy import spatial
import math
cameras = {}
intrinsics = {}
poses = {}
mesh = None
from operator import itemgetter
from Geometry import KDTree
#import pyvista as pv

@njit(fastmath=True)
def checkocclussion(face, face2, camera_centre):

    vectors = np.array([[0,0,0],[0,0,0],[0,0,0]],dtype=np.float32)

    vectors[0] = np.subtract(face[0], camera_centre)
    vectors[1] = np.subtract(face[1], camera_centre)
    vectors[2] = np.subtract(face[2], camera_centre)

    for x in vectors:
        temp = np.subtract(x,camera_centre)
        camera_direction = temp / np.linalg.norm(temp)
        v0 = face2[0]
        v1 = face2[1]
        v2 = face2[2]

        v0v1 = np.subtract(v1, v0)
        v0v2 = np.subtract(v2, v0)
        pvec = np.cross(camera_direction, v0v2)

        det = np.dot(v0v1, pvec)

        if det < 0.000001:
            return 0

        invDet = 1.0 / det
        tvec = np.subtract(camera_centre, v0)
        u = np.multiply(np.dot(tvec, pvec), invDet)

        if u < 0 or u > 1:
            return 0

        qvec = np.cross(tvec, v0v1)
        v = np.multiply(np.dot(camera_direction, qvec), invDet)

        if v < 0 or u + v > 1:
            return 0

        t = np.multiply(np.dot(v0v2, qvec), invDet)
        if t >= 0:
            return 1

    return 0

def checkocclussion2(face, face2, camera_centre):

    vectors = np.array([[0,0,0],[0,0,0],[0,0,0]],dtype=np.float32)

    vectors[0] = np.subtract(face[0], camera_centre)
    vectors[1] = np.subtract(face[1], camera_centre)
    vectors[2] = np.subtract(face[2], camera_centre)

    for x in vectors:
        temp = np.subtract(x,camera_centre)
        camera_direction = temp / np.linalg.norm(temp)
        v0 = face2[0]
        v1 = face2[1]
        v2 = face2[2]

        v0v1 = np.subtract(v1, v0)
        v0v2 = np.subtract(v2, v0)
        pvec = np.cross(camera_direction, v0v2)

        det = np.dot(v0v1, pvec)

        if det < 0.000001:
            return 0

        invDet = 1.0 / det
        tvec = np.subtract(camera_centre, v0)
        u = np.multiply(np.dot(tvec, pvec), invDet)

        if u < 0 or u > 1:
            return 0

        qvec = np.cross(tvec, v0v1)
        v = np.multiply(np.dot(camera_direction, qvec), invDet)

        if v < 0 or u + v > 1:
            return 0

        t = np.multiply(np.dot(v0v2, qvec), invDet)
        if t >= 0:
            return 1

    return 0

@njit(parallel=True)
def occlussioncheck(face, meshes, cameras, length):

    results = np.zeros(length, dtype=np.int16)

    for x in range(length):
        camera_centre = cameras[x]
        for y in meshes:
            a = -1
            a = checkocclussion(face, y, camera_centre)

            if (a == 1):
                results[x] = 1
                break
            results[x] = 0

    return results

def selectViewTaskV3(data):
    args = data[0]
    cameras = data[1]

    poses = data[2]

    intrinsics = data[4]

    #tree = KDTree.triangleTree()
    y = np.array(data[3], dtype=np.float32)
    # tree.addTriangles(triangles=y)
    # tree.buildTree()
    cam_score = np.zeros(len(cameras))
    cam_score2 = np.zeros(len(cameras))
    results = []

    #1st pass

    for z in args:
        face_coords_np = np.array(z[0], dtype=np.float32)
        count = 0
        for cam in cameras:
            count+=1
            fit = -1
            source = []
            for y in face_coords_np:
                xyz1 = np.array([y[0], y[1], y[2], 1], dtype=np.float32)
                im = np.array(intrinsics[cameras[cam][1]][0], dtype=np.float32)
                em = np.array(poses[cameras[cam][0]][0], dtype=np.float32)
                c = CT.getPixelCoords(xyz1, im, em)
                source.append(c)

                if c[0] < 0:
                    fit = 0
                    break
                elif c[1] < 0:
                    fit = 0
                    break
                elif (c[1] > intrinsics[cameras[cam][1]][2][1]):
                    fit = 0
                    break
                elif (c[0] > intrinsics[cameras[cam][1]][2][0]):
                    fit = 0
                    break

            if (fit == 0):
                continue

            if (fit == -1):
                bb = CT.calcBoundingBoxTri(source)
                if (bb[0][0] == bb[1][0]):
                    continue
                elif (bb[0][1] == bb[1][1]):
                    continue

            cam_score2[count-1] += 1
    ####################################################

    for z in args:
        min = -1
        face = z[1]
        face_coords_np = np.array(z[0], dtype=np.float32)
        #nearby = tree.getpoints_within_d(face_coords_np, 0.05)
        # nearby = tree.getpoints_between_range(face_coords_np, 0.09, 0.1)


        # centres = []
        # for x in cameras:
        #     centres.append(np.array(poses[cameras[x][0]][1], dtype=np.float32))
        #
        # occluded = occlussioncheck(face_coords_np, np.array(nearby, dtype=np.float32), np.array(centres, dtype=np.float32), len(centres))

        count = 0
        for cam in cameras:
            # if(int(occluded[count])==1):
            #     count+=1
            #     continue
            # else:
            #     count+=1
            fit = -1
            b = checkAngle(face_coords_np, poses[cameras[cam][0]][2])
            source = []
            for y in face_coords_np:
                xyz1 = np.array([y[0],y[1],y[2],1], dtype=np.float32)
                im = np.array(intrinsics[cameras[cam][1]][0], dtype=np.float32)
                em = np.array(poses[cameras[cam][0]][0], dtype=np.float32)
                #dist = np.array(intrinsics[cameras[cam][1]][1], dtype=np.float32)
                c = CT.getPixelCoords(xyz1, im, em)
                ##c = CT.undistortcoords(c, im, dist)
                source.append(c)

                if c[0] < 0:
                    fit = 0
                    break
                elif c[1] < 0:
                    fit = 0
                    break
                elif(c[1] > intrinsics[cameras[cam][1]][2][1]):
                    fit = 0
                    break
                elif (c[0] > intrinsics[cameras[cam][1]][2][0]):
                    fit = 0
                    break

            if (fit == 0):
                continue

            if(fit==-1):
                bb = CT.calcBoundingBoxTri(source)
                if(bb[0][0] == bb[1][0]):
                    continue
                elif(bb[0][1] == bb[1][1]):
                    continue
            d = dist(face_coords_np,poses[cameras[cam][0]][1])
            h = b + (np.sqrt(cam_score2[count])/np.square(d)*2)
            if b < 0:
                continue
            elif min == -1:
                min = [h,  cam, face, count]
            else:
                if h > min[0]:
                    min = [h, cam, face, count]
            count+=1
        #cam_score[min[3]]+=1
        results.append(min)

    return results


def dist(p1,p2):
    dist = np.linalg.norm(p2-p1)
    return dist

def groupMesh(mesh):
    faces = mesh.faces_coords()
    face = mesh.faces()
    max_p = 10
    pool = Pool(max_p)
    args = []

    n_cuts, membership = pymetis.part_graph(max_p, adjacency=face)

    iter = 0
    for x in range(max_p):
        args.append([])

    args2 = []
    for x in range(max_p):
        args2.append([])

    # for y in range(len(faces)):
    #     args[iter].append([faces[y], face[y]])
    #     iter+=1
    #     if(iter == max_p):
    #         iter=0

    iter = 0
    for y in face:
        args[membership[y[0]]].append( [faces[iter] , y])
        iter+=1

    for y in range(max_p):
        args2[y].append([args[y], cameras, poses, faces, intrinsics])


    results = pool.starmap(selectViewTaskV3,args2)
    pool.close()
    pool.join()

    new_results = []
    f = open("meshToView.txt", "a")
    for x in results:
        for z in x:
            new_results.append(z)
            f.write(str(z))
            f.write("\n")
    f.close()
    pkl.dump(new_results, open("meshToView.p", "wb"))


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

    degrees = math.degrees(np.arccos(np.minimum(1.0, np.maximum(np.dot(v1_u, v2_u), -1.0))))
    #rad = np.arccos(np.dot(v1_u, v2_u))
    #degrees = math.degrees(rad)
    return degrees

@jit(nopython=True)
def unit_vector(vector):
    return np.divide(vector , np.linalg.norm(vector))

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
    # should change this later on
    list = pkl.load(open("meshToView.p", "rb"))
    Q = Manager().Queue()
    size = s
    meshes = mesh.faces()
    vertices = mesh.mesh_vertices()
    uv = mesh.uv_vertices()
    processes = []
    closed = Manager().Value('1', 0)

    pictures = {}

    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Loading Images...")
    for i in cameras:
        pictures[i] = Image.open(cameras[i][2], 'r')

    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Done")

    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Loading Data to Memory...")


    ignored = 0
    outside_region = 0
    line_crops = 0

    list = sorted(list, key=itemgetter(1))

    args = []
    p_max = 4
    current_img_id = -1
    p_iter = 0

    for x in range(p_max):
        args.append([])

    for m in list:

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

        b_b = CT.calcBoundingBoxTri(dst_v)

        src_tri_coords = []
        skip = 0

        src_size = pictures[imgId].size
        for v in src_v:
            xyz1 = np.array([v[0],v[1],v[2],1], dtype=np.float32)

            im = np.array(intrinsics[cameras[imgId][1]][0], dtype=np.float32)
            em = np.array(poses[cameras[imgId][0]][0], dtype=np.float32)
            #dist = np.array(intrinsics[cameras[imgId][1]][1], dtype=np.float32)
            n = CT.getPixelCoords(xyz1, im, em)
            #n = CT.undistortcoords(n,im,dist)
            src_tri_coords.append(n)

        bb = CT.calcBoundingBoxTri(src_tri_coords)
        crop_size = ((bb[0][0]), (bb[0][1]), (bb[1][0]), (bb[1][1]))
        counted = -1
        for n in bb:
            for x in range(len(src_size)):
                if(n[x]>src_size[x]):
                    skip = 1
                    if(counted == -1):
                        outside_region += 1
                        counted = 1
                elif(n[x]<0):
                    skip = 1
                    if (counted == -1):
                        outside_region += 1
                        counted = 1

        if crop_size[0] == crop_size[2]:
            skip = 1
            line_crops += 1
        elif crop_size[1] == crop_size[3]:
            skip = 1
            line_crops += 1

        if skip == 1:
            ignored += 1
            continue

        args[p_iter].append([src_tri_coords, dst_v, Q, pictures[imgId].crop(crop_size)])
        p_iter += 1
        if p_iter == p_max:
            p_iter = 0

    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Done")
    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Total Meshes: " + str(len(meshes)))
    print(" - " + datetime.now().strftime("%H:%M:%S") + ": Error Meshes: " + str(ignored)
        + " [ Outside Region: " + str(outside_region) + ", Line Crops: " + str(line_crops) + " ]"
        )


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


    for x in cameras:

        principal = [[intrinsics[cameras[x][1]][0][0][2]], [intrinsics[cameras[x][1]][0][1][2]], [1]]

        ext = poses[cameras[x][0]][0]

        temp = np.matmul(intrinsics[cameras[x][1]][0], ext)

        principal[0] -= temp[0][3]
        principal[1] -= temp[1][3]
        principal[2] -= temp[2][3]

        p_world = np.linalg.solve(temp[:3, :3], principal)

        cameras[x] = [cameras[x][0], cameras[x][1], cameras[x][2], p_world]

    for x in cameras:
        lv = np.subtract(np.transpose(cameras[x][3]), poses[cameras[x][0]][1])
        lv = lv[0]
        lv = unit_vector(lv)
        #lv = np.multiply(lv,-1)
        poses[cameras[x][0]][2] = np.array(lv, dtype=np.float32)

    # pvmesh = pv.read(mesh_location)
    # plotter = pv.Plotter()
    # plotter.add_mesh(pvmesh, color='red')
    #
    # for x in poses.values():
    #      #point = x[1]
    #      lv = x[2]
    #      line = [x[1], x[1] + (np.multiply(lv, 1))]
    #      #plotter.add_points(np.array(point))
    #      plotter.add_lines(np.array(line))
    #
    # for x in cameras.values():
    #     point = x[3]
    #     plotter.add_points(np.array(point).transpose(), color="red")
    # plotter.show()

    print(datetime.now().strftime("%H:%M:%S") + ": Done")

    print(datetime.now().strftime("%H:%M:%S") + ": Processing Meshes...")
    groupMesh(mesh)
    print(datetime.now().strftime("%H:%M:%S") + ": Done")

    print(datetime.now().strftime("%H:%M:%S") + ": Creating Texture Map...")
    createTextureMap(texturesize)
    print(datetime.now().strftime("%H:%M:%S") + ": Done")
    print(datetime.now().strftime("%H:%M:%S") + ": File saved as Texture.png in root folder")
