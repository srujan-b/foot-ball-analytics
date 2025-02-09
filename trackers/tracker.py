from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append('../')
from utils import getBboxWidth,getCenterOfBbox
import cv2
import numpy as np
import pandas as pd


class Tracker:

    def __init__(self,modelPath):
        self.model = YOLO(modelPath)
        self.tracker = sv.ByteTrack()

    def ballInterpolation(self,ballPositions):
        ballPositions = [x.get(1,{}).get('bbox',[]) for x in ballPositions]
        dfBallPositions = pd.DataFrame(ballPositions,columns=['x1','y1','x2','y2'])
        dfBallPositions = dfBallPositions.interpolate()
        dfBallPositions = dfBallPositions.bfill()

        ballPositions = [{1:{'bbox':x}} for x in dfBallPositions.to_numpy().tolist()]

        return ballPositions
    
    def detectFrames(self, frames):
        batchSize = 20
        detections = []

        for i in range(0,len(frames),batchSize):

            detectionBatch = self.model.predict(frames[i:i+batchSize],conf=0.1)
            detections += detectionBatch
            
        return detections
    
    def getObjectTracks(self, frames , readFromStubs = False , stubPath = None):


        if readFromStubs and stubPath is not None and os.path.exists(stubPath):
            with open(stubPath,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        tracks = {
           "players" :[],
           "referees":[],
           "ball":[]
        }


        detections = self.detectFrames(frames)

        for frameNum,detection in enumerate(detections) :
            clsNames = detection.names
            clsNameInv = {v:k for k,v in clsNames.items()}
            # convert to supervision detection format

            detectionSupervision = sv.Detections.from_ultralytics(detection)


            for objInd, classId in enumerate(detectionSupervision.class_id):
                if clsNames[classId] == "goalkeeper":
                    detectionSupervision.class_id[objInd] = clsNameInv["player"]

            detectionWithTracks = self.tracker.update_with_detections(detectionSupervision)
            

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frameDetection in detectionWithTracks:

                bbox = frameDetection[0].tolist()
                clsId = frameDetection[3]
                trackId = frameDetection[4]

                if clsId ==  clsNameInv['player'] or clsId ==  clsNameInv['goalkeeper']:

                    tracks["players"][frameNum][trackId] = {"bbox":bbox}
                
                if clsId ==  clsNameInv['referee']:
                    
                    tracks["referees"][frameNum][trackId] = {"bbox":bbox}
            
            for frameDetection in detectionSupervision:

                bbox = frameDetection[0].tolist()
                clsId = frameDetection[3]
                trackId = frameDetection[4]

                if clsId == clsNameInv['ball']:

                    tracks["ball"][frameNum][1] = {"bbox":bbox}
            

        if stubPath is not None:
            with open(stubPath,'wb') as f:
                pickle.dump(tracks,f)    
        return tracks

    def drawEllipse(self,frame,bbox,color,trackId=None):
        y2 = int(bbox[3])
        xCenter, yCenter = getCenterOfBbox(bbox)
        width = getBboxWidth(bbox)

        cv2.ellipse(
            frame,
            center=(xCenter,y2),
            axes=(int(width),int(0.35*width)),
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )

        rectangleWidth = 40
        rectangleHeight = 20
        x1Rect  = xCenter - rectangleWidth//2
        x2Rect  = xCenter + rectangleWidth//2
        y1React = (y2 - rectangleHeight//2) +15
        y2React = (y2 + rectangleHeight//2) +15

        if trackId is not None:

            cv2.rectangle(frame,
                          (int(x1Rect),int(y1React)),
                          (int(x2Rect),int(y2React)),
                          color,
                          cv2.FILLED)
            
            x1Text = x1Rect+12

            if trackId > 99:
                x1Text -=10
            cv2.putText(
                frame,f"{trackId}",
                (int(x1Text),int(y1React+15)),cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    
    def drawTrianle(self,frame,bbox,color):

        y = int(bbox[1])
        x,_ = getCenterOfBbox(bbox)

        trianglePoints = np.array([[x,y],
                                   [x-10,y-20],
                                   [x+10,y-20]])

        cv2.drawContours(frame,[trianglePoints],0,color,cv2.FILLED)
        cv2.drawContours(frame,[trianglePoints],0,(0,0,0),2)

        return frame

    def drawAnnotations(self,videoFrames,tracks):



        outputVideoFrames = []

        for frameNum,frame in enumerate(videoFrames):

            frame = frame.copy()
            
            playerDict = tracks["players"][frameNum]
            refreesDict = tracks["referees"][frameNum]
            ballDict = tracks["ball"][frameNum]

            #draw players 
            for trackId,player in playerDict.items():
                color = player.get("teamColor",(255,255,255))
                frame = self.drawEllipse(frame, player["bbox"],color, trackId)

                if player.get('hasBall',False):

                    frame = self.drawTrianle(frame,player["bbox"],(0,0,255))
            
            
            for trackID,refrees in refreesDict.items():
                frame = self.drawEllipse(frame, refrees["bbox"],(0,255,255), None)
            
            # draw ball 
            for _,ball in ballDict.items():
                frame = self.drawTrianle(frame,ball["bbox"],(0,255,0))
            
            
            
            outputVideoFrames.append(frame)
        
        return outputVideoFrames

    