from utils import read_video,save_video
from trackers import Tracker

def main():
    #read video
    videoFrames = read_video('/mnt/c/Users/somes/Documents/foot-ball-analytics/video/ipVideo.mp4')

    # Initilize the tracker
    tracker = Tracker('/mnt/c/Users/somes/Documents/foot-ball-analytics/models/best.pt')

    tracks = tracker.getObjectTracks(videoFrames,
                            readFromStubs=True,
                            stubPath='/mnt/c/Users/somes/Documents/foot-ball-analytics/stubs/track_stubs.pkl')
    
    # draw o/p 
    # draw object track
    
    outputVideoFrames = tracker.drawAnnotations(videoFrames,tracks)



    # save video
    save_video(outputVideoFrames,'/mnt/c/Users/somes/Documents/foot-ball-analytics/output_videos/output_video.avi')

if __name__ == '__main__':
    main()