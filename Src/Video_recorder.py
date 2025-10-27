import cv2
import os
from datetime import datetime
import csv
import time

class VideoRecorder:
    def __init__(self, resolution=(640, 480), fps=20.0):
        self.resolution = resolution
        self.fps = fps
        self.recording = False
        self.video_writer = None
        self.recording_start_time = 0
        self.last_frame_capture = 0
        self.frame_capture_interval = 5.0  # Capture frames every 5 seconds
        self.frame_count = 0
        
        # Create directories if they don't exist
        if not os.path.exists('videos'):
            os.makedirs('videos')
        if not os.path.exists('captured_frames'):
            os.makedirs('captured_frames')
    
    def start_recording(self):
        """Start recording video and frames"""
        if not self.recording:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"videos/rover_navigation_{timestamp}.mp4"
            
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_filename, fourcc, self.fps, self.resolution)
            self.recording_start_time = time.time()
            self.last_frame_capture = self.recording_start_time
            self.recording = True
            self.frame_count = 0
            print(f"🎥 Started recording: {video_filename}")
            return True
        return False
    
    def stop_recording(self):
        """Stop recording video"""
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("🎥 Recording stopped")
            return True
        return False
    
    def capture_frame(self, frame):
        """Capture individual frame if interval has passed"""
        current_time = time.time()
        if current_time - self.last_frame_capture >= self.frame_capture_interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_filename = f"captured_frames/frame_{timestamp}_{self.frame_count:06d}.jpg"
            cv2.imwrite(frame_filename, frame)
            self.last_frame_capture = current_time
            print(f"📸 Captured frame: {frame_filename}")
    
    def write_frame(self, frame):
        """Write frame to video and capture individual frames if recording"""
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)
            self.capture_frame(frame)
            self.frame_count += 1
    
    def is_recording(self):
        """Check if currently recording"""
        return self.recording
    
    def add_recording_indicator(self, frame):
        """Add recording indicator to frame"""
        if self.recording:
            cv2.putText(frame, "RECORDING", (10, self.resolution[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Frames: {self.frame_count}", (10, self.resolution[1] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame
    
    def get_stats(self):
        """Get recording statistics"""
        return {
            'recording': self.recording,
            'frame_count': self.frame_count,
            'duration': time.time() - self.recording_start_time if self.recording else 0
        }