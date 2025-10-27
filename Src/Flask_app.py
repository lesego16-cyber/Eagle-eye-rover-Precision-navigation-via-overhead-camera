from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import threading
import time
import socket
import logging
import math
from datetime import datetime

# Import created modules
from NavigationC import VectorNavigator
from Controller import controller
from Image_processing import detect_rover, CalibratedCamera, TargetTracker, draw_navigation_info, PerformanceLogger, calculate_real_distance, calculate_heading_error, SpeedDistanceLogger
from Video_recorder import VideoRecorder

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# Configure the camera
CAMERA_INDEX = 1
RESOLUTION = (640, 480)
FPS = 20.0

# Gloabl variables
latest_frame = None
frame_lock = threading.Lock()
stream_active = True
frame_count = 0
current_target_pixels = None
current_target_real = None
rover_pixel_position = None
rover_real_position = None
rover_front_real_position = None
rover_heading = None
scanning_active = False
last_debug_print = 0
test_start_time = None
current_test_id = "test_1"

# Initialize modules 
navigator = VectorNavigator()
video_recorder = VideoRecorder(RESOLUTION, FPS)
calibrated_camera = CalibratedCamera()
target_tracker = TargetTracker()
performance_logger = PerformanceLogger()
speed_distance_logger = SpeedDistanceLogger()  # NEW: Speed-distance logger

# Set physical measurements
calibrated_camera.set_physical_parameters(
    camera_height=250.0
    marker_height=5.5
)

GREEN_LOWER = np.array([35, 40, 40])
GREEN_UPPER = np.array([90, 255, 255])

# Helper functions 
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except: return "localhost"

def camera_thread():
    global latest_frame, rover_pixel_position, rover_real_position, rover_heading, current_target_pixels, current_target_real
    global rover_front_real_position, frame_count
    
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)
    print(f"✅ Camera started. Connect at http://{get_local_ip()}:5000")
    print(f"   Resolution: {RESOLUTION[0]}x{RESOLUTION[1]}")

    while stream_active:
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Undistort the frame for both display and calculations
        undistorted_frame = calibrated_camera.undistort(frame)
        
        # Process rover detection on undistorted frame
        processed_frame, rover_data, processing_time = detect_rover(undistorted_frame, GREEN_LOWER, GREEN_UPPER, calibrated_camera)
        performance_logger.add_processing_time(processing_time)
        performance_logger.increment_frame_count()
        frame_count += 1
        
        # Store rover data for navigation and display
        if rover_data:
            rover = rover_data[0]
            rover_pixel_position = rover['pixel_position']
            rover_real_position = rover['real_position']
            rover_front_real_position = rover['real_front_position']
            rover_heading = rover['heading']

        # Update target position using static target tracking on undistorted frame
        target_position, target_bbox = target_tracker.update_target(undistorted_frame)
        
        if target_tracker.is_tracking and target_position:
            current_target_pixels = target_position
            
            # Convert target pixel to real world coordinates
            if calibrated_camera.measured:
                target_x_cm, target_y_cm = calibrated_camera.pixel_to_ground_cm(target_position[0], target_position[1])
                current_target_real = (target_x_cm, target_y_cm)
            else:
                current_target_real = None
            
            # Calculate and display navigation errors 
            if rover_front_real_position and current_target_real and rover_heading:
                # Draw navigation info with distance and heading errors
                processed_frame = draw_navigation_info(processed_frame, rover_data, target_position, calibrated_camera)
            
            # Draw static target visualization
            processed_frame = target_tracker.draw_target(processed_frame, target_position)
            
        elif current_target_pixels and not target_tracker.is_tracking:
            cv2.circle(processed_frame, (int(current_target_pixels[0]), int(current_target_pixels[1])), 8, (0, 255, 255), -1)
            cv2.circle(processed_frame, (int(current_target_pixels[0]), int(current_target_pixels[1])), 12, (0, 255, 255), 2)
            cv2.putText(processed_frame, "TARGET SET", 
                       (int(current_target_pixels[0]) + 20, int(current_target_pixels[1]) - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        status_y = 30
        if target_tracker.is_tracking:
            status_text = "TARGET SET"
            color = (0, 255, 255)  # Yellow for static target
            cv2.putText(processed_frame, status_text, (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        video_recorder.write_frame(processed_frame)

        # Measure transmission time
        transmission_start = time.time()
        with frame_lock:
            _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY),70])
            latest_frame = buffer.tobytes()
        transmission_time = (time.time() - transmission_start) * 1000
        performance_logger.add_transmission_time(transmission_time)
        
        time.sleep(0.05)

def navigation_thread():
    global scanning_active
    while True:
        if (navigator.navigation_active and rover_front_real_position and
            rover_heading and current_target_real and not scanning_active):
            
            current_rover_front = rover_front_real_position
            current_rover_heading = rover_heading
            current_target = current_target_real
            
            # Calculate current distance
            current_distance = calculate_real_distance(current_rover_front, current_target)
            
            # DEBUG: Print navigation input every 5 seconds
            current_time = time.time()
            if hasattr(navigation_thread, 'last_debug_time'):
                if current_time - navigation_thread.last_debug_time > 5.0:
                    heading_error = calculate_heading_error(current_rover_front, current_rover_heading, current_target)
                    print(f"🧭 REAL-WORLD NAVIGATION INPUT:")
                    print(f"   Rover FRONT real: {current_rover_front}")
                    print(f"   Target real: {current_target}")
                    print(f"   Real distance: {current_distance:.1f}cm")
                    print(f"   Heading error: {heading_error:.1f}°")
                    navigation_thread.last_debug_time = current_time
            else:
                navigation_thread.last_debug_time = current_time
                heading_error = calculate_heading_error(current_rover_front, current_rover_heading, current_target)
                print(f"🧭 REAL-WORLD NAVIGATION INPUT:")
                print(f"   Rover FRONT real: {current_rover_front}")
                print(f"   Target real: {current_target}")
                print(f"   Real distance: {current_distance:.1f}cm")
                print(f"   Heading error: {heading_error:.1f}°")
            
            # Measure command transmission time
            command_start = time.time()
            result = navigator.execute_navigation(current_rover_front, current_rover_heading, current_target)
            command_time = (time.time() - command_start) * 1000
            performance_logger.add_command_time(command_time)
            
            # Log distance and speed
            if hasattr(navigator, 'last_command') and hasattr(navigator, 'last_speed'):
                speed_distance_logger.log_data(
                    distance=current_distance,
                    speed=navigator.last_speed,
                    navigation_mode=navigator.navigation_mode,
                    command=navigator.last_command
                )
            
            if result == "SCAN_ACTIVE":
                scanning_active = True
                print("⏳ Rover scanning environment...")
            elif result == "TARGET_REACHED":
                # Log final performance when target is reached
                performance_logger.log_final_performance(test_id=current_test_id)
                navigator.stop_navigation()

        # Check if rover has finished scanning the environment
        if scanning_active:
            time.sleep(3.0)
            scanning_active = False
            print("✅ Scan complete, resuming navigation")
                
        time.sleep(0.1)

# Flask routes 
@app.route('/')
def index(): 
    return render_template('index.html', local_ip=get_local_ip(), rover_ip=navigator.rover_ip)

@app.route('/set_target', methods=['POST'])
def set_target():
    global current_target_pixels, current_target_real
    data = request.get_json()
    x = data.get('x'); y = data.get('y')
    if x is None or y is None: 
        return jsonify({'status':'error'}), 400
    
    # Set target position
    if target_tracker.set_target(int(x), int(y)):
        current_target_pixels = (float(x), float(y))
        
        # Convert to real coordinates immediately
        if calibrated_camera.measured:
            target_x_cm, target_y_cm = calibrated_camera.pixel_to_ground_cm(x, y)
            current_target_real = (target_x_cm, target_y_cm)
        else:
            current_target_real = None
            
        print(f"✅ Static target set at pixels: {current_target_pixels}")
        print(f"   Real coordinates: {current_target_real}")
        return jsonify({
            'status':'success',
            'target':current_target_pixels,
            'tracking_initialized': True
        })
    
    return jsonify({'status':'error', 'message': 'Failed to set target'}), 400

@app.route('/clear_target')
def clear_target():
    global current_target_pixels, current_target_real
    current_target_pixels = None
    current_target_real = None
    target_tracker.reset_tracking()
    navigator.stop_navigation()
    print("🗑️ Target cleared")
    return jsonify({'status':'success'})

@app.route('/start_navigation')
def start_navigation():
    global test_start_time, scanning_active
    if scanning_active:
        return jsonify({'status':'scan_in_progress'}), 503
    if current_target_real and rover_front_real_position:
        # Reset performance logger for new test
        performance_logger.start_test()
        speed_distance_logger.reset() 
        test_start_time = time.time()
        
        # DEBUG: Print initial conditions
        real_distance = calculate_real_distance(rover_front_real_position, current_target_real)
        heading_error = calculate_heading_error(rover_front_real_position, rover_heading, current_target_real)
        print(f"🚀 STARTING REAL-WORLD NAVIGATION:")
        print(f"   Initial real distance: {real_distance:.1f}cm")
        print(f"   Initial heading error: {heading_error:.1f}°")
        print(f"   Rover FRONT position: {rover_front_real_position}")
        print(f"   Target position: {current_target_real}")
        
        success = navigator.start_navigation()
        return jsonify({'status':'navigation_started'} if success else {'status':'rover_connection_failed'})
    return jsonify({'status':'no_target'}), 400

@app.route('/stop_navigation')
def stop_navigation():
    global test_start_time
    navigator.stop_navigation()
    test_start_time = None  
    print("🛑 Navigation stopped manually - performance not logged")
    return jsonify({'status':'navigation_stopped'})

@app.route('/start_recording')
def start_recording():
    return jsonify({'status':'recording_started'} if video_recorder.start_recording() else {'status':'already_recording'})

@app.route('/stop_recording')
def stop_recording():
    return jsonify({'status':'recording_stopped'} if video_recorder.stop_recording() else {'status':'not_recording'})

@app.route('/set_test_id', methods=['POST'])
def set_test_id():
    global current_test_id
    data = request.get_json()
    test_id = data.get('test_id', 'test_1')
    current_test_id = test_id
    print(f"📝 Test ID set to: {current_test_id}")
    return jsonify({'status': 'success', 'test_id': current_test_id})

@app.route('/get_status')
def get_status():
    actual_distance = None
    heading_error = None
    if rover_front_real_position and current_target_real:
        actual_distance = calculate_real_distance(rover_front_real_position, current_target_real)
        heading_error = calculate_heading_error(rover_front_real_position, rover_heading, current_target_real)
    
    return jsonify({
        'rover_position': {'x': rover_front_real_position[0], 'y': rover_front_real_position[1]} if rover_front_real_position else None,
        'rover_heading': rover_heading if rover_heading else None,
        'target': {'x': current_target_real[0], 'y': current_target_real[1]} if current_target_real else None,
        'actual_distance': actual_distance,
        'heading_error': heading_error,
        'recording': video_recorder.is_recording(),
        'rover_connected': navigator.check_rover_connection(),
        'scanning': scanning_active,
        'rover_ip': navigator.rover_ip,
        'target_tracking': target_tracker.is_tracking,
        'tracking_confidence': 100,
        'target_lost': False,
        'test_id': current_test_id
    })

@app.route('/video_feed')
def video_feed():
    def generate():
        while stream_active:
            with frame_lock:
                if latest_frame: 
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n'
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main function
if __name__ == '__main__':
    print(f"🎯 REAL-WORLD Navigation System")
    print(f"   Target: 10.0cm real distance")
    print(f"   Performance logging: ENABLED")
    print(f"   Speed-Distance logging: ENABLED")
    
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=navigation_thread, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)