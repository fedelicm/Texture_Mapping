class MaskObj():

    origin = None
    boundingBbox = None
    mask = None
    transformed = None
    dst_tri = None

    def __init__(self, mask, transformed, origin, boundingBox, dst_tri):
        self.mask = mask
        self.origin = origin
        self.boundingBox = boundingBox
        self.dst_tri = dst_tri
        self.transformed = transformed

    def getBoundingBox(self):
        return self.boundingBox

    def getMask(self):
        return self.mask

    def getOrigin(self):
        return self.origin

    def getTransformed(self):
        return self.transformed

    def getDstTri(self):
        return self.dst_tri
