from utils import read_video,save_video
from trackers import Tracker

def main():
    #read video
    videoFrames = read_video('/mnt/c/Users/somes/Documents/foot-ball-analytics/video/08fd33_4.mp4')

    # Initilize the tracker
    tracker = Tracker('/mnt/c/Users/somes/Documents/foot-ball-analytics/models/best.pt')

    tracker.getObjectTracks(videoFrames,
                            readFromStubs=True,
                            stubPath='/mnt/c/Users/somes/Documents/foot-ball-analytics/stubs/track_stubs.pkl')
    
    
    # save video
    save_video(videoFrames,'/mnt/c/Users/somes/Documents/foot-ball-analytics/output_videos/output_video.avi')

if __name__ == '__main__':
    main()