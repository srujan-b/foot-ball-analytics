def getCenterOfBbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def getBboxWidth(bbox):
    return bbox[2] - bbox[0]

def measureDistance(p1,p2):

    return ((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2 )**0.5
