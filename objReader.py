import re
import numpy as np

mesh = None

vertices_mesh_arr = []
vertices_uv_arr = []
faces_arr = []
faces_id = []

def readOBJ(obj):

    mesh = open(obj, "r")

    for x in mesh:
        if(re.search("v (.)", x)):

            arr = []
            txt = x[2:].split(" ")

            for n in txt:
                arr.append(np.float32(n))

            vertices_mesh_arr.append(arr)

        elif (re.search("vt (.)", x)):
            arr = []
            txt = x[3:].split(" ")
            for n in txt:
                arr.append(np.float32(n))
            vertices_uv_arr.append(arr)

        elif(re.search("f (.)", x)):
            arr = []
            txt = x[2:].split(" ")
            for n in txt:
                arr.append(int(n.split("/")[0]))

            faces_id.append(arr)

    for x in faces_id:
        arr = []
        for y in x:
            arr.append(vertices_mesh_arr[y - 1])
        faces_arr.append(arr)

def faces_coords():
    return faces_arr

def faces():
    return faces_id

def mesh_vertices():
    return vertices_mesh_arr

def uv_vertices():
    return vertices_uv_arr









