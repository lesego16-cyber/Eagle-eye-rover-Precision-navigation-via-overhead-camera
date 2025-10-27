import cv2
import numpy as np
import time
import math
import csv
from datetime import datetime

# --- Camera Calibration ---
class CalibratedCamera:
    def __init__(self, calibration_file='camera_calibration.npz'):
        try:
            data = np.load(calibration_file)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            self.camera_height = None
            self.marker_height = None
            self.measured = False
            print("✅ Camera calibration loaded successfully")
            print(f"   Camera matrix: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
        except Exception as e:
            print(f"❌ Camera calibration failed: {e}")
            self.camera_matrix = None
            self.dist_coeffs = None
            self.fx = self.fy = 500.0
            self.cx = 320.0
            self.cy = 240.0
            self.measured = False

    def set_physical_parameters(self, camera_height, marker_height):
        self.camera_height = float(camera_height) 
        self.marker_height = float(marker_height) 
        self.measured = True
        print(f"📏 Physical parameters set: Camera height={self.camera_height}cm, Marker height={self.marker_height}cm")

    def pixel_to_ground_cm(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates (u, v) to ground-plane coordinates (X, Y) in cm
        using the calibrated intrinsic parameters and the known camera height.
        Assumes the ground is a flat plane (Z_W = 0).
        """
        if not self.measured:
            return pixel_x, pixel_y

        # Normalize pixel to camera coordinates
        x = (pixel_x - self.cx) / self.fx
        y = (pixel_y - self.cy) / self.fy

        # Scale normalized ray by height to get ground intersection
        # For an overhead camera looking straight down, Zc ≈ camera_height
        X_ground = x * self.camera_height
        Y_ground = y * self.camera_height

        return X_ground, Y_ground

    def calculate_distance(self, pixel_height):
        """Calculate distance to object based on its pixel height"""
        if not self.measured or pixel_height <= 0:
            return 0
            
        distance_cm = (self.marker_height * self.fy) / pixel_height
        return distance_cm

    def calculate_ground_distance(self, point1_cm, point2_cm):
        """Calculate real-world distance between two points in cm"""
        return np.linalg.norm(np.array(point1_cm) - np.array(point2_cm))

    def undistort(self, image):
        """Simple undistort - use calibrated parameters if available"""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            try:
                return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
            except Exception as e:
                print(f"❌ Undistortion failed: {e}")
                return image
        return image

# --- Target Tracking ---
class TargetTracker:
    def __init__(self):
        self.target_position = None
        self.is_tracking = False
        
    def set_target(self, x, y):
        """Set a static target position"""
        self.target_position = (x, y)
        self.is_tracking = True
        print(f"🎯 Static target set at ({x}, {y})")
        return True
        
    def update_target(self, frame):
        """For static target, just return the set position"""
        if self.is_tracking and self.target_position:
            return self.target_position, None
        return None, None
        
    def draw_target(self, frame, target_position):
        """Draw static target with yellow circle"""
        if target_position:
            yellow_color = (0, 255, 255)
            cv2.circle(frame, target_position, 25, yellow_color, 3)
            cv2.circle(frame, target_position, 8, (0, 0, 255), -1)
            cv2.circle(frame, target_position, 12, (0, 0, 255), 2)
        return frame

    def reset_tracking(self):
        """Reset tracking state"""
        self.is_tracking = False
        self.target_position = None

# Compute distance and heading calculations
def calculate_real_distance(rover_front_real_position, target_real_position):
    """Calculate distance between rover FRONT and target in centimeters - EXACT SAME AS NAVIGATION"""
    if rover_front_real_position is None or target_real_position is None:
        return None
    
    dx = target_real_position[0] - rover_front_real_position[0]
    dy = target_real_position[1] - rover_front_real_position[1]
    real_distance = math.hypot(dx, dy)
    
    return real_distance

def calculate_heading_error(rover_front_real_position, rover_heading, target_real_position):
    """Calculate heading error between rover and target - EXACT SAME AS NAVIGATION"""
    if rover_front_real_position is None or target_real_position is None or rover_heading is None:
        return None
    
    dx = target_real_position[0] - rover_front_real_position[0]
    dy = target_real_position[1] - rover_front_real_position[1]
    desired_heading = math.degrees(math.atan2(dy, dx)) % 360
    heading_error = desired_heading - (rover_heading % 360)
    
    # Normalize heading error to [-180, 180] 
    if heading_error > 180:
        heading_error -= 360
    elif heading_error < -180:
        heading_error += 360
        
    return heading_error

# Detect the rover using features
def detect_rover(frame, green_lower, green_upper, calibrated_camera):
    start_time = time.time()
    
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect blue marker which indicates the rover's front 
    blue_lower = np.array([90, 120, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    kernel = np.ones((3, 3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blue_center = None
    if blue_contours:
        blue_contour = max(blue_contours, key=cv2.contourArea)
        if cv2.contourArea(blue_contour) > 5:
            M = cv2.moments(blue_contour)
            if M['m00'] != 0:
                blue_x = int(M['m10'] / M['m00'])
                blue_y = int(M['m01'] / M['m00'])
                blue_center = (blue_x, blue_y)
                cv2.circle(frame, blue_center, 7, (255, 0, 0), -1)
    
    # Detect green markers
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rover_data = []
    green_positions = []
    
    for c in green_contours:
        area = cv2.contourArea(c)
        if area > 10:
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                green_positions.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    
    # Detect rover using at least 3 markers
    if blue_center is not None and len(green_positions) >= 3:
        # Use only the first 4 green markers if more are detected
        green_positions = green_positions[:4]
        
        # Calculate centroid from green markers
        centroid = np.mean(green_positions, axis=0)
        x, y = centroid

        # Use blue center as front position
        front_x, front_y = blue_center

        real_world_position = None
        real_world_front_position = None
        
        if calibrated_camera.measured:
            # Calculate real world coordinates for centroid and front
            x_cm, y_cm = calibrated_camera.pixel_to_ground_cm(x, y)
            real_world_position = (x_cm, y_cm)
            
            # Calculate real world coordinates for front position
            front_x_cm, front_y_cm = calibrated_camera.pixel_to_ground_cm(front_x, front_y)
            real_world_front_position = (front_x_cm, front_y_cm)

        # Heading calculation: from combined marker centroid to blue marker
        dx = blue_center[0] - x
        dy = blue_center[1] - y
        
        # Calculate heading directly 
        heading = np.degrees(np.arctan2(dy, dx)) % 360

        rover_data.append({
            'pixel_position': (x, y),
            'pixel_front_position': (front_x, front_y),
            'real_position': real_world_position,
            'real_front_position': real_world_front_position,
            'heading': heading,
            'distance_cm': None
        })

        # Draw red arrow from centroid to blue marker to indicate the rover's orientation
        arrow_length = 60
        end_x = int(x + arrow_length * np.cos(np.radians(heading)))
        end_y = int(y + arrow_length * np.sin(np.radians(heading)))

    
        cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y), (0, 0, 255), 4)

    processing_time = (time.time() - start_time) * 1000
    return frame, rover_data, processing_time

# Provide navigation  information on display
def draw_navigation_info(frame, rover_data, target_position, calibrated_camera):
    """Draw navigation information using EXACT SAME calculations as navigation system"""
    if not rover_data or not target_position:
        return frame

    rover = rover_data[0]
    
    if rover['real_front_position'] and calibrated_camera.measured:
       
        rover_front_cm = rover['real_front_position']
        
        # Convert target pixel to real world coordinates
        target_cm = calibrated_camera.pixel_to_ground_cm(target_position[0], target_position[1])
        # Compute distance and heading errors
        distance_error = calculate_real_distance(rover_front_cm, target_cm)
        
        heading_error = calculate_heading_error(rover_front_cm, rover['heading'], target_cm)

        # Draw green navigation line from rover front to target
        cv2.line(frame, 
                rover['pixel_front_position'], 
                target_position, 
                (0, 255, 0), 2)

        text_y = 30
        cv2.putText(frame, f"Distance Error: {distance_error:.1f}cm", 
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Heading Error: {heading_error:.1f}°", 
                   (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

# Logs all performance parameters to CSV files
class PerformanceLogger:
    def __init__(self, filename='performance.csv'):  
        self.filename = filename
        self.processing_times = []
        self.transmission_times = []
        self.command_times = []
        self.execution_times = []
        self.frame_count = 0
        self.test_start_time = None
        self.csv_initialized = False
        self.init_csv()
        
    def init_csv(self):
        """Initialize CSV file with headers if file doesn't exist"""
        try:
            # Check if file exists and has content
            import os
            if not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0:
                with open(self.filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Timestamp', 'Test_ID', 'Avg_Processing_Time_ms', 'Avg_Transmission_Time_ms',
                        'Avg_Command_Time_ms', 'Avg_Execution_Time_ms', 'Total_Frames',
                        'Test_Duration_Seconds'
                    ])
                print(f"✅ Performance CSV initialized: {self.filename}")
            else:
                print(f"📁 Performance CSV already exists: {self.filename}")
            self.csv_initialized = True
        except Exception as e:
            print(f"❌ Failed to initialize performance CSV: {e}")
            self.filename = None

    def start_test(self):
        """Start timing for a new test"""
        print(f"🔧 PERFORMANCE LOGGER: Starting new test")
        self.test_start_time = time.time()
        self.processing_times = []
        self.transmission_times = []
        self.command_times = []
        self.execution_times = []
        self.frame_count = 0

    def add_processing_time(self, time_ms):
        """Add frame processing time"""
        self.processing_times.append(time_ms)
        
    def add_transmission_time(self, time_ms):
        """Add video transmission time"""
        self.transmission_times.append(time_ms)
        
    def add_command_time(self, time_ms):
        """Add command transmission time"""
        self.command_times.append(time_ms)
        
    def add_execution_time(self, time_ms):
        """Add rover command execution time"""
        self.execution_times.append(time_ms)
        
    def increment_frame_count(self):
        """Increment frame counter"""
        self.frame_count += 1
        
    def log_final_performance(self, test_id):
        """Log final performance metrics ONLY when target is reached"""
        print(f"🔧 PERFORMANCE LOGGER: Attempting to log for test {test_id}")
        print(f"🔧 File: {self.filename}")
        print(f"🔧 Test start time: {self.test_start_time}")
        
        if not self.filename:
            print(f"❌ Cannot log performance: No filename set")
            return
            
        if not self.test_start_time:
            print(f"❌ Cannot log performance: No test active (test_start_time is None)")
            return
            
        try:
            # Calculate averages
            avg_processing = np.mean(self.processing_times) if self.processing_times else 0
            avg_transmission = np.mean(self.transmission_times) if self.transmission_times else 0
            avg_command = np.mean(self.command_times) if self.command_times else 0
            avg_execution = np.mean(self.execution_times) if self.execution_times else 0
            
            test_duration = time.time() - self.test_start_time
            
            print(f"🔧 Writing to {self.filename}...")
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    test_id,
                    f"{avg_processing:.2f}",
                    f"{avg_transmission:.2f}",
                    f"{avg_command:.2f}",
                    f"{avg_execution:.2f}",
                    self.frame_count,
                    f"{test_duration:.2f}"
                ])
            
            print(f"💾 FINAL PERFORMANCE LOGGED for test {test_id}")
            print(f"   Test Duration: {test_duration:.1f}s")
            print(f"   Avg Processing: {avg_processing:.2f}ms")
            print(f"   Avg Transmission: {avg_transmission:.2f}ms")
            print(f"   Avg Command: {avg_command:.2f}ms")
            print(f"   Avg Execution: {avg_execution:.2f}ms")
            print(f"   Total Frames: {self.frame_count}")
            
            # Reset test timing
            self.test_start_time = None
            print(f"🔧 Performance logger reset complete")
            
        except Exception as e:
            print(f"❌ Failed to log performance: {e}")
            import traceback
            traceback.print_exc()

    def reset(self):
        """Reset metrics for new test"""
        self.start_test()

# Global performance logger instance
performance_logger = PerformanceLogger()

# Logs all speed performance parameters
class SpeedDistanceLogger:
    def __init__(self, filename='speed_distance.csv'):
        self.filename = filename
        self.data = [] 
        self.init_csv()
        
    def init_csv(self):
        """Initialize CSV file with headers"""
        try:
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'Distance_cm', 'Speed', 'Navigation_Mode', 'Command'
                ])
            print(f"✅ Speed-Distance CSV initialized: {self.filename}")
        except Exception as e:
            print(f"❌ Failed to initialize speed-distance CSV: {e}")
            self.filename = None

    def log_data(self, distance, speed, navigation_mode, command):
        """Log current distance and speed"""
        if not self.filename:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.data.append((timestamp, distance, speed, navigation_mode, command))
            
            # Append to CSV file
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    f"{distance:.2f}",
                    speed,
                    navigation_mode,
                    command
                ])
                
        except Exception as e:
            print(f"❌ Failed to log speed-distance data: {e}")

    def reset(self):
        """Reset data for new test"""
        self.data = []

# Global speed-distance logger instance
speed_distance_logger = SpeedDistanceLogger()