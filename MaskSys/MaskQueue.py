from multiprocessing import Process, Manager, Queue, Pool, Array, shared_memory
import numpy as np
import queue as qq
from PIL import Image
from datetime import datetime

img = None
processing_list = None
pending_masks = None
done_masks = None
done_add = None

count = None
max_p = 3
process_pool = None

closed = None

cleaner_process = None
parallel_done = None
not_added = None
add_process = None
X = None
X_shape = None
started = False
shr = None

def addMask(mask):
    global not_added
    not_added.put(mask)

def checkForOverlap(mask, processing_dict):
    d = processing_dict.values()
    for x in d:
        if checkBoundingBox(mask.getBoundingBox(), x):
            return False
    return True

def checkBoundingBox(R1, R2):

    if (R1[0][0] >= R2[1][0] or R2[0][0]>= R1[1][0]):
        return False

    if (R1[0][1] <= R2[1][1] or R2[0][1] <= R1[1][1]):
        return False

    return True


def start(img_size):
    global started
    global X_shape
    global shr
    global X
    global not_added
    global done_masks
    global closed
    global done_add
    global closed
    global count
    global parallel_done
    global processing_list
    global process_pool
    global pending_masks

    if started:
        return

    started = True

    img = Image.new('RGBA', (img_size, img_size),color=(0,0,0,0))
    size = img.size

    X_shape = (size[1], size[0], 4)
    X_shape2 = (size[0], size[1], 4)
    shr = shared_memory.SharedMemory(create=True, size=X_shape[0]*X_shape[1]*X_shape[2] )

    np_array = np.frombuffer(shr.buf, np.uint8, count= X_shape[0] * X_shape[1] * X_shape[2] ).reshape(X_shape)
    np_array[:] = (imgToArr(img))[:]

    pending_masks = Manager().Queue()
    done_masks = Manager().Queue()
    not_added = Manager().Queue()
    not_added.maxsize = 10000
    closed = Manager().Value('i', 0)
    done_add = Manager().Value('i', 0)
    count = Manager().Value('i', 0)
    parallel_done = Manager().Value('i', 0)
    processing_list = Manager().dict()
    process_pool = []

    startAddProcess()
    name = shr.name

    for i in range(max_p):
        process_pool.append(Process(target=maskTask,args=(pending_masks, done_masks, done_add, name, X_shape2)))

        process_pool[i].start()

    startCleaner()

def imgToArr(img):
    return np.transpose(np.array(img, np.uint8), (1,0,2))

def startAddProcess():
    global add_process
    add_process = Process(target=addTask,args=(not_added, closed, done_add, pending_masks, count, processing_list))
    add_process.start()

def addTask(Q, closed, done, pending_masks, count, processing_dict):
    while True:
        if closed.get() == 1 and Q.qsize() == 0:
            done.set(1)
            break
        try:

            mask = Q.get_nowait()
            if checkForOverlap(mask, processing_dict):

                c = count.get()
                bb = mask.getBoundingBox()

                processing_dict[c] = bb

                pending_masks.put([c, mask])
                count.set(c+1)

            else:
                 Q.put(mask)
            del mask
        except qq.Empty:
            pass

def fillTransparent(img, locXY):
    size = img.shape
    total = 0
    total_colourR = 0
    total_colourG = 0
    total_colourB = 0
    if(img[locXY[1]+1][locXY[0]][4]!=0):
        total+=1
        total_colourR += img[locXY[1]+1][locXY[0]][0]
        total_colourG += img[locXY[1]+1][locXY[0]][1]
        total_colourB += img[locXY[1]+1][locXY[0]][2]
    if(img[locXY[1]][locXY[0]+1][4]!=0):
        total+=1
        total_colourR += img[locXY[1]][locXY[0]+1][0]
        total_colourG += img[locXY[1]][locXY[0]+1][1]
        total_colourB += img[locXY[1]][locXY[0]+1][2]
    if(img[locXY[1]-1][locXY[0]][4]!=0):
        total+=1
        total_colourR += img[locXY[1]-1][locXY[0]][0]
        total_colourG += img[locXY[1]-1][locXY[0]][1]
        total_colourB += img[locXY[1]-1][locXY[0]][2]
    if(img[locXY[1]][locXY[0]-1][4]!=0):
        total+=1
        total_colourR += img[locXY[1]][locXY[0]-1][0]
        total_colourG += img[locXY[1]][locXY[0]-1][1]
        total_colourB += img[locXY[1]][locXY[0]-1][2]

#@jit(nopython=True)
def maskPasteNP(arr1, arr2, xy, mask):
    #max_gap = 2
    #arr2 = np.array(arr2)
    #mask = np.array(mask)

    #if(mask.shape[0] != arr2.shape[0]  or mask.shape[1] != arr2.shape[1]):
    #    print("Dimensions of Mask and Source Image do not match")
    #    print(arr2.shape)
    #    print(mask.shape)

    size = arr2.shape
    for y in range(size[1]):
        last_was_black = False
        for x in range(size[0]):
            if(mask[x][y]==1):
                if(arr2[x][y][3]!=0):
                    arr1[xy[1] + x][xy[0] + y] = arr2[x][y]
                #     if (last_was_black and x - 2 >= 0):
                #         if (arr1[xy[1] + x - 2][xy[0] + y][0] != 0
                #                 and arr1[xy[1] + x - 2][xy[0] + y][1] != 0
                #                 and arr1[xy[1] + x - 2][xy[0] + y][2] != 0
                #         ):
                #             arr1[xy[1] + x - 1][xy[0] + y][0] = np.uint8(
                #                 int((int(arr1[xy[1] + x - 2][xy[0] + y][0]) + int(arr1[xy[1] + x][xy[0] + y][0])) / 2))
                #             arr1[xy[1] + x - 1][xy[0] + y][1] = np.uint8(
                #                 int((int(arr1[xy[1] + x - 2][xy[0] + y][1]) + int(arr1[xy[1] + x][xy[0] + y][1])) / 2))
                #             arr1[xy[1] + x - 1][xy[0] + y][2] = np.uint8(
                #                 int((int(arr1[xy[1] + x - 2][xy[0] + y][2]) + int(arr1[xy[1] + x][xy[0] + y][2])) / 2))
                #             arr1[xy[1] + x - 1][xy[0] + y][3] = np.uint8(int(255))
                #     last_was_black = False
                # else:
                #     if (last_was_black is False and x + 1 < size[0]):
                #         if (arr1[xy[1] + x + 1][xy[0] + y][0] != 0
                #                 and arr1[xy[1] + x + 1][xy[0] + y][1] != 0
                #                 and arr1[xy[1] + x + 1][xy[0] + y][2] != 0
                #         ):
                #             arr1[xy[1] + x][xy[0] + y][0] = np.uint8(int(
                #                 (int(arr1[xy[1] + x - 1][xy[0] + y][0]) + int(arr1[xy[1] + x + 1][xy[0] + y][0])) / 2))
                #             arr1[xy[1] + x][xy[0] + y][1] = np.uint8(int(
                #                 (int(arr1[xy[1] + x - 1][xy[0] + y][1]) + int(arr1[xy[1] + x + 1][xy[0] + y][1])) / 2))
                #             arr1[xy[1] + x][xy[0] + y][2] = np.uint8(int(
                #                 (int(arr1[xy[1] + x - 1][xy[0] + y][2]) + int(arr1[xy[1] + x + 1][xy[0] + y][2])) / 2))
                #             arr1[xy[1] + x][xy[0] + y][3] = np.uint8(int(255))
                #     last_was_black = True


def arrToImg(arr, shape):
    X_shape2 = [shape[1],shape[0]]
    return Image.frombuffer('RGBA', X_shape2[:2], arr )

def imgPasteNP(arr1,arr2,loc):
    size = arr2.shape
    arr1[loc[1]:loc[1]+size[0],loc[0]:loc[0]+size[1]] = arr2[:]


def maskTask(Q, Q_done, done_add, shr_name, X_shape):
    xbuff = shared_memory.SharedMemory(name=shr_name, create=False)
    img_arr = np.frombuffer(xbuff.buf, np.uint8, count=X_shape[0] * X_shape[1] * X_shape[2]).reshape(X_shape)

    while True:
        if done_add.get() == 1 and Q.qsize() == 0:
            break

        try:
            a = Q.get_nowait()

            bb = a[1].getBoundingBox()

            loc = np.array([bb[0][0], bb[0][1]], dtype=np.int32)

            maskPasteNP(img_arr, np.array(a[1].getTransformed(), dtype=np.uint8), loc, np.array(a[1].getMask(), dtype=np.uint8))

            Q_done.put(a[0])
        except qq.Empty:
            pass


def realign(tri, origin):
    new_tri = []
    for x in tri:
        temp = []
        for y in range(len(origin)):
            temp.append(x[y] - origin[y])
        new_tri.append(temp)
    return new_tri

def startCleaner():
    global cleaner_process
    cleaner_process = Process(target=cleanerTask, args=(done_masks,not_added,parallel_done, processing_list))
    cleaner_process.start()

def cleanerTask(Q, Q2, done, processing_dict):
    num = 0
    while True:
        if done.get() == 1 and Q2.qsize() == 0:
            break
        try:
            a = Q.get_nowait()

            processing_dict.pop(a)
            num += 1
            if(num>=100000):
                print(" - " + datetime.now().strftime("%H:%M:%S") + ": Masks Processed: " + str(a+1))
                num-=100000
        except qq.Empty:
            pass

def join():
    global add_process
    global process_pool
    global parallel_done
    global cleaner_process
    global shr

    add_process.join()

    for x in range(max_p):
        process_pool[x].join()

    parallel_done.set(1)
    cleaner_process.join()

    img_arr = np.frombuffer(shr.buf, np.uint8, count=X_shape[0] * X_shape[1] * X_shape[2])
    img = arrToImg(img_arr,X_shape)
    return img

def kill():
    global shr
    shr.close()
    shr.unlink()


def close():
    global closed
    closed.set(1)
