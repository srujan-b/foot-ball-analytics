from sklearn.cluster import KMeans


class TeamAssigner:

    def __init__(self):
        self.teamColors = {}
        self.playerTeamDict ={}
    


    def getClustringModel(self,image):

        # reshape th eimge into 2d array 
        image2D = image.reshape(-1,3)

        # perform the clusting 
        kmeans = KMeans(n_clusters = 2,init = "k-means++",n_init=1).fit(image2D)

        return kmeans

    def getPlayerColor(self, frame,bbox):

        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        topHalfImage = image[0:int(image.shape[0]/2),:]
        
        #get clustring model
        kmeans = self.getClustringModel(topHalfImage)

        # get clustered labels for each pixel 

        labels = kmeans.labels_

        # reshape the labels to the image shape

        clusterdImage = labels.reshape(topHalfImage.shape[0],topHalfImage.shape[1])

        # get the player cluster

        cornerClusters = [clusterdImage[0,0],
                          clusterdImage[0,-1],
                          clusterdImage[-1,0],
                          clusterdImage[-1,-1]]
        nonPlayerCluster = max(set(cornerClusters),key=cornerClusters.count)

        playerCluster = 1-nonPlayerCluster

        playerColor = kmeans.cluster_centers_[playerCluster]

        return playerColor

    def assignTeamColor(self,frame,playerDetections):

        playerColors = []

        for _,playerDetection in playerDetections.items():

            bbox = playerDetection['bbox']
            playerColor = self.getPlayerColor(frame,bbox)

            playerColors.append(playerColor)
        
        kmeans = KMeans(n_clusters = 2,init = "k-means++",n_init=1).fit(playerColors)


        self.kmeans = kmeans
        self.teamColors[1] = kmeans.cluster_centers_[0]
        self.teamColors[2] = kmeans.cluster_centers_[1]

    def getPlayerTeam(self,frame,playerBbox,playerId):

        if playerId in self.playerTeamDict:
            return self.playerTeamDict[playerId]
        else:

            playerColor = self.getPlayerColor(frame,playerBbox)

            teamId = self.kmeans.predict(playerColor.reshape(1,-1))[0]

            teamId+=1

            self.playerTeamDict[playerId] = teamId

            return teamId

