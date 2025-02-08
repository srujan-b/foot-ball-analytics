from ultralytics import YOLO
import supervision as sv
import pickle
import os


class Tracker:

    def __init__(self,modelPath):
        self.model = YOLO(modelPath)
        self.tracker = sv.ByteTrack()
    
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
