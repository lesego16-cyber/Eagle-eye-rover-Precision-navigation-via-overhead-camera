import math
import time
import cv2
import numpy as np
import sys
import os
import csv
import random

# Add the path to modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from NavigationC import VectorNavigator
from distance_controller import controller

# Simulation parameters 
WINDOW_SIZE = (800, 600)
FPS = 20
ROVER_RADIUS = 10
TARGET_RADIUS = 15

#  Movement parameters based on actual rover behavior
BASE_MOVEMENT_PER_COMMAND = 1.5  
BASE_TURN_PER_COMMAND = 2.0      

# Real-world modelling parameters obtained from physical experiments
SENSOR_NOISE_DISTANCE = 0.8      # ±0.8cm noise in distance measurements
SENSOR_NOISE_HEADING = 2.5       # ±2.5° noise in heading measurements
CONTROL_DELAY = 0.3              # 300ms delay between command and execution
MOMENTUM_FACTOR = 0.7            
MIN_EFFECTIVE_TURN = 1.5         # Minimum turn that actually affects heading 
TURN_SLIPPAGE = 0.3              
SPEED_VARIABILITY = 0.15         # ±15% speed variation per command

# Assumed rover dynamics
MAX_ACCELERATION = 0.5           
CURRENT_SPEED = 0               
CURRENT_TURN_RATE = 0          

# imulated rover state 
rover_pos = np.array([200.0, 200.0], dtype=float)
rover_heading = 0.0  
target_pos = np.array([600.0, 400.0], dtype=float)
navigation_start_time = 0

PIXELS_PER_CM = 1.0

# State tracking
command_queue = [] 
last_actual_movement = 0
sensor_readings = []  

# Initialize navigator 
navigator = VectorNavigator()

# Simulation state
navigation_active = False
final_results = None
last_command = None
last_speed = 0
distance_history = []
heading_error_history = []
last_controller_time = 0
CONTROLLER_INTERVAL = 0.5

# CSV Performance logging
performance_csv_file = "sim_performance.csv"

original_send_to_rover = None
original_check_connection = None
original_set_obstacle = None

def add_sensor_noise(true_value, noise_level):
    """Add realistic sensor noise with some persistence"""
    global sensor_readings
    
    noise = random.gauss(0, noise_level)
    
    if sensor_readings:
        recent_bias = np.mean(sensor_readings[-3:]) if len(sensor_readings) >= 3 else 0
        noise = noise * 0.7 + recent_bias * 0.3
    
    sensor_readings.append(noise)
    if len(sensor_readings) > 10:
        sensor_readings.pop(0)
    
    return true_value + noise

def apply_momentum(current_speed, commanded_speed, acceleration_limit):
    """Simulate momentum and acceleration limits"""
    speed_diff = commanded_speed - current_speed
    # Limit acceleration
    if abs(speed_diff) > acceleration_limit:
        speed_diff = np.sign(speed_diff) * acceleration_limit
    
    return current_speed + speed_diff

def simulated_rover_movement(cmd, speed):
    """REALISTIC movement with momentum, slippage, and variability"""
    global rover_pos, rover_heading, CURRENT_SPEED, CURRENT_TURN_RATE, last_actual_movement
    
    # Apply speed variability since real motors arent perfect
    actual_speed = speed * (1 + random.uniform(-SPEED_VARIABILITY, SPEED_VARIABILITY))
    
    # Apply momentum to speed changes
    CURRENT_SPEED = apply_momentum(CURRENT_SPEED, actual_speed, MAX_ACCELERATION)
    
    # Calculate current distance to target 
    dx = target_pos[0] - rover_pos[0]
    dy = target_pos[1] - rover_pos[1]
    distance = math.hypot(dx, dy)
    
    # Movement scaling based on distance 
    if distance > 80.0:
        movement_scale = 1.0
    elif distance > 60.0:
        movement_scale = 0.9
    elif distance > 40.0:
        movement_scale = 0.8
    elif distance > 30.0:
        movement_scale = 0.7
    elif distance > 20.0:
        movement_scale = 0.5
    elif distance > 15.0:
        movement_scale = 0.3
    elif distance > 12.0:
        movement_scale = 0.2
    elif distance > 10.5:
        movement_scale = 0.1
    else:
        movement_scale = 0.05
    
    # Scale by speed
    speed_ratio = CURRENT_SPEED / 120.0
    movement_scale *= speed_ratio
    

    movement_executed = 0
    
    if cmd == 'F':
        movement_distance = BASE_MOVEMENT_PER_COMMAND * movement_scale
        
        # Add some random variation to forward movement
        movement_distance *= (1 + random.uniform(-0.1, 0.1))
        
        rad = math.radians(rover_heading)
        rover_pos[0] += movement_distance * math.cos(rad) * PIXELS_PER_CM
        rover_pos[1] += movement_distance * math.sin(rad) * PIXELS_PER_CM
        movement_executed = movement_distance
        CURRENT_TURN_RATE = 0
        print(f"🤖 SIM: Moving FORWARD {movement_distance:.1f}cm at speed {CURRENT_SPEED:.0f} (scale: {movement_scale:.2f})")
        
    elif cmd == 'CW':

        turn_angle = BASE_TURN_PER_COMMAND * speed_ratio
        turn_angle *= (1 - TURN_SLIPPAGE) 
        
        # Minimum effective turn
        if turn_angle < MIN_EFFECTIVE_TURN:
            turn_angle = MIN_EFFECTIVE_TURN if random.random() > 0.3 else 0
        
        rover_heading += turn_angle
        CURRENT_TURN_RATE = turn_angle
        movement_executed = turn_angle
        print(f"🤖 SIM: Turning CW {turn_angle:.1f}° at speed {CURRENT_SPEED:.0f} (slippage: {TURN_SLIPPAGE:.1f})")
        
    elif cmd == 'CCW':
        turn_angle = BASE_TURN_PER_COMMAND * speed_ratio
        turn_angle *= (1 - TURN_SLIPPAGE)
        
        if turn_angle < MIN_EFFECTIVE_TURN:
            turn_angle = MIN_EFFECTIVE_TURN if random.random() > 0.3 else 0
            
        rover_heading -= turn_angle
        CURRENT_TURN_RATE = -turn_angle
        movement_executed = turn_angle
        print(f"🤖 SIM: Turning CCW {turn_angle:.1f}° at speed {CURRENT_SPEED:.0f} (slippage: {TURN_SLIPPAGE:.1f})")
        
    elif cmd == 'R':
        # Smaller turns with even more variability
        turn_angle = BASE_TURN_PER_COMMAND * speed_ratio * 0.6
        turn_angle *= (1 - TURN_SLIPPAGE - 0.1)
        if turn_angle < MIN_EFFECTIVE_TURN * 0.7:
            turn_angle = MIN_EFFECTIVE_TURN * 0.7 if random.random() > 0.4 else 0
            
        rover_heading += turn_angle
        CURRENT_TURN_RATE = turn_angle
        movement_executed = turn_angle
        print(f"🤖 SIM: Turning RIGHT {turn_angle:.1f}° at speed {CURRENT_SPEED:.0f}")
        
    elif cmd == 'L':
        turn_angle = BASE_TURN_PER_COMMAND * speed_ratio * 0.6
        turn_angle *= (1 - TURN_SLIPPAGE - 0.1)
        
        if turn_angle < MIN_EFFECTIVE_TURN * 0.7:
            turn_angle = MIN_EFFECTIVE_TURN * 0.7 if random.random() > 0.4 else 0
            
        rover_heading -= turn_angle
        CURRENT_TURN_RATE = -turn_angle
        movement_executed = turn_angle
        print(f"🤖 SIM: Turning LEFT {turn_angle:.1f}° at speed {CURRENT_SPEED:.0f}")
        
    elif cmd == 'S' or cmd == 'STOP':
        # Gradual stopping with momentum
        CURRENT_SPEED = apply_momentum(CURRENT_SPEED, 0, MAX_ACCELERATION * 2)
        CURRENT_TURN_RATE = 0
        print(f"🤖 SIM: STOPPING - current speed: {CURRENT_SPEED:.0f}")
    
    # Keep heading in 0-360
    rover_heading %= 360
    last_actual_movement = movement_executed

def get_noisy_sensor_readings():
    """Return sensor readings with realistic noise"""
    dx = target_pos[0] - rover_pos[0]
    dy = target_pos[1] - rover_pos[1]
    true_distance = math.hypot(dx, dy)
    true_desired_heading = math.degrees(math.atan2(dy, dx)) % 360
    
    # Adds sensor noise
    noisy_distance = add_sensor_noise(true_distance, SENSOR_NOISE_DISTANCE)
    noisy_rover_heading = add_sensor_noise(rover_heading, SENSOR_NOISE_HEADING)
    
    # Calculate noisy heading error
    noisy_error_heading = true_desired_heading - (noisy_rover_heading % 360)
    if noisy_error_heading > 180:
        noisy_error_heading -= 360
    elif noisy_error_heading < -180:
        noisy_error_heading += 360
    
    return noisy_distance, noisy_error_heading, noisy_rover_heading

def log_final_performance(final_distance, final_heading_error, final_speed, rover_x, rover_y, target_x, target_y, total_time):
    """Log only final performance data to CSV"""
    try:
        need_header = not os.path.exists(performance_csv_file)
        
        with open(performance_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if need_header or f.tell() == 0:
                writer.writerow(['timestamp', 'final_distance_cm', 'final_heading_error_deg', 'final_speed', 
                               'rover_x', 'rover_y', 'target_x', 'target_y', 'total_time_s'])
                print(f"📊 Created/wrote header to CSV file: {performance_csv_file}")
            
            # Write the data row
            data_row = [time.time(), final_distance, final_heading_error, final_speed,
                       rover_x, rover_y, target_x, target_y, total_time]
            writer.writerow(data_row)
            print(f"📊 Appended data to CSV: {data_row}")
        
        print(f"📊 Final performance logged to: {performance_csv_file}")
        
    except Exception as e:
        print(f"❌ Failed to log performance: {e}")
        # Try to create the file if it doesn't exist
        try:
            with open(performance_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'final_distance_cm', 'final_heading_error_deg', 'final_speed', 
                               'rover_x', 'rover_y', 'target_x', 'target_y', 'total_time_s'])
            print(f"📊 Created new CSV file: {performance_csv_file}")
        except Exception as e2:
            print(f"❌ Could not create CSV file: {e2}")

def setup_simulation_navigator():
    """Temporarily modify the navigator for REALISTIC simulation"""
    global original_send_to_rover, original_check_connection, original_set_obstacle
    
    original_send_to_rover = navigator.send_to_rover
    original_check_connection = navigator.check_rover_connection
    original_set_obstacle = navigator.set_obstacle_detection_mode
    
    def simulation_send_to_rover(cmd, speed=120, rover_position=None, heading_param=None, target_position=None):
        global last_command, last_speed, rover_heading, command_queue
        
        # Store the command and speed from the navigation system
        last_command = cmd
        last_speed = speed
        
        # Convert command to Arduino format 
        short_cmd = navigator._map_cmd_for_arduino(cmd)
        
        # Get noisy sensor readings 
        noisy_distance, noisy_heading_error, noisy_heading = get_noisy_sensor_readings()

        print(f"➡️ Sending command to rover: {short_cmd} @ speed {speed}")
        print(f"   📏 Distance to target: {noisy_distance/PIXELS_PER_CM:.1f}cm")
        
        execute_time = time.time() + CONTROL_DELAY
        command_queue.append((execute_time, short_cmd, speed))
        
        # Process any pending commands that are ready
        current_time = time.time()
        commands_to_execute = [cmd for cmd in command_queue if cmd[0] <= current_time]
        for cmd_data in commands_to_execute:
            _, cmd_to_execute, speed_to_execute = cmd_data
            simulated_rover_movement(cmd_to_execute, speed_to_execute)
            command_queue.remove(cmd_data)
        
        command_queue = [cmd for cmd in command_queue if current_time - cmd[0] < 1.0]
        
        # Simulate successful command execution
        navigator.connection_errors = 0
        navigator.last_successful_command_time = time.time()
        
        # Update navigator status 
        if rover_position is not None:
            # Add noise to reported position
            noisy_rover_pos = rover_position + np.random.normal(0, 0.5, 2)
            navigator.rover_status['position'] = noisy_rover_pos
        if heading_param is not None:
            navigator.rover_status['heading'] = noisy_heading
        navigator.rover_status['is_moving_forward'] = short_cmd == 'F'
        
        return True
    
    navigator.send_to_rover = simulation_send_to_rover
    navigator.check_rover_connection = lambda: True  
    navigator.set_obstacle_detection_mode = lambda enable: True  

def restore_navigator():
    """Restore the original navigator methods"""
    global original_send_to_rover, original_check_connection, original_set_obstacle
    if original_send_to_rover:
        navigator.send_to_rover = original_send_to_rover
    if original_check_connection:
        navigator.check_rover_connection = original_check_connection
    if original_set_obstacle:
        navigator.set_obstacle_detection_mode = original_set_obstacle

def start_navigation():
    """Start navigation using the ACTUAL navigation system"""
    global navigation_active, distance_history, heading_error_history, navigation_start_time
    global CURRENT_SPEED, CURRENT_TURN_RATE, command_queue, sensor_readings
    
    # Reset real-world simulation state
    navigation_active = True
    distance_history = []
    heading_error_history = []
    navigation_start_time = time.time()
    CURRENT_SPEED = 0
    CURRENT_TURN_RATE = 0
    command_queue = []
    sensor_readings = []
    
    # Ensures CSV file exists and has header
    try:
        if not os.path.exists(performance_csv_file):
            with open(performance_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'final_distance_cm', 'final_heading_error_deg', 'final_speed', 
                               'rover_x', 'rover_y', 'target_x', 'target_y', 'total_time_s'])
            print(f"📊 Created new CSV file: {performance_csv_file}")
        else:
            print(f"📊 Using existing CSV file: {performance_csv_file}")
    except Exception as e:
        print(f"❌ CSV file setup failed: {e}")
    
    # Start navigation using the navigation module
    result = navigator.start_navigation()
    if result:
        print("🚀 Navigation started using ACTUAL navigation system")
        print("🔧 REAL-WORLD SIMULATION ENABLED:")
        print(f"   - Sensor noise: ±{SENSOR_NOISE_DISTANCE}cm distance, ±{SENSOR_NOISE_HEADING}° heading")
        print(f"   - Control delay: {CONTROL_DELAY}s")
        print(f"   - Momentum factor: {MOMENTUM_FACTOR}")
        print(f"   - Turn slippage: {TURN_SLIPPAGE*100}%")
    return result

def stop_navigation():
    """Stop navigation using the ACTUAL navigation system"""
    global navigation_active, navigation_start_time
    
    # Calculate final metrics for logging
    dx = target_pos[0] - rover_pos[0]
    dy = target_pos[1] - rover_pos[1]
    final_distance = math.hypot(dx, dy)
    desired_heading = math.degrees(math.atan2(dy, dx)) % 360
    final_heading_error = desired_heading - (rover_heading % 360)
    if final_heading_error > 180:
        final_heading_error -= 360
    elif final_heading_error < -180:
        final_heading_error += 360
    
    # Calculate total time 
    total_time = time.time() - navigation_start_time
    
    # Log final performance to CSV
    log_final_performance(
        final_distance=final_distance,
        final_heading_error=final_heading_error,
        final_speed=last_speed,
        rover_x=rover_pos[0] / PIXELS_PER_CM,
        rover_y=rover_pos[1] / PIXELS_PER_CM,
        target_x=target_pos[0] / PIXELS_PER_CM,
        target_y=target_pos[1] / PIXELS_PER_CM,
        total_time=total_time
    )
    
    print(f"📊 FINAL RESULTS:")
    print(f"   Distance: {final_distance:.2f}cm")
    print(f"   Heading Error: {final_heading_error:.2f}°")
    print(f"   Final Speed: {last_speed}")
    print(f"   Rover Position: ({rover_pos[0]/PIXELS_PER_CM:.1f}, {rover_pos[1]/PIXELS_PER_CM:.1f})")
    print(f"   Target Position: ({target_pos[0]/PIXELS_PER_CM:.1f}, {target_pos[1]/PIXELS_PER_CM:.1f})")
    print(f"   Total Time: {total_time:.2f}s")
    
    # Stop navigation 
    result = navigator.stop_navigation()
    if result:
        print("🛑 Navigation stopped using ACTUAL navigation system")
    return result

def draw_simulation_frame():
    """Draw the complete simulation visualization"""
    # Create blank image
    frame = np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)
    
    for x in range(0, WINDOW_SIZE[0], 50):
        cv2.line(frame, (x, 0), (x, WINDOW_SIZE[1]), (50, 50, 50), 1)
    for y in range(0, WINDOW_SIZE[1], 50):
        cv2.line(frame, (0, y), (WINDOW_SIZE[0], y), (50, 50, 50), 1)

    # Draw target
    cv2.circle(frame, (int(target_pos[0]), int(target_pos[1])), TARGET_RADIUS, (0, 255, 255), -1)
    cv2.circle(frame, (int(target_pos[0]), int(target_pos[1])), int(10 * PIXELS_PER_CM), (0, 200, 200), 2)
    cv2.circle(frame, (int(target_pos[0]), int(target_pos[1])), int(9.8 * PIXELS_PER_CM), (0, 150, 150), 1)
    cv2.circle(frame, (int(target_pos[0]), int(target_pos[1])), int(10.2 * PIXELS_PER_CM), (0, 150, 150), 1)

    # Draw rover
    rover_color = (0, 255, 0) if navigation_active else (100, 100, 100)
    cv2.circle(frame, (int(rover_pos[0]), int(rover_pos[1])), ROVER_RADIUS, rover_color, -1)

    # Draw rover heading
    front_x = int(rover_pos[0] + ROVER_RADIUS * 2 * math.cos(math.radians(rover_heading)))
    front_y = int(rover_pos[1] + ROVER_RADIUS * 2 * math.sin(math.radians(rover_heading)))
    cv2.line(frame, (int(rover_pos[0]), int(rover_pos[1])), (front_x, front_y), rover_color, 2)

    # Draw line to target
    cv2.line(frame, (int(rover_pos[0]), int(rover_pos[1])), (int(target_pos[0]), int(target_pos[1])), (0, 0, 255), 1)

    # Calculate current metrics
    dx = target_pos[0] - rover_pos[0]
    dy = target_pos[1] - rover_pos[1]
    distance = math.hypot(dx, dy)
    desired_heading = math.degrees(math.atan2(dy, dx)) % 360
    current_heading_error = desired_heading - (rover_heading % 360)
    if current_heading_error > 180:
        current_heading_error -= 360
    elif current_heading_error < -180:
        current_heading_error += 360

    # Update history 
    distance_history.append(distance / PIXELS_PER_CM)
    heading_error_history.append(current_heading_error)
    if len(distance_history) > 100:
        distance_history.pop(0)
        heading_error_history.pop(0)

    # Draw history graph
    graph_height = 80
    graph_width = 200
    graph_x = 10
    graph_y = WINDOW_SIZE[1] - graph_height - 10
    
    # Distance graph
    if distance_history:
        max_dist = max(100, max(distance_history))
        for i in range(1, len(distance_history)):
            x1 = graph_x + (i-1) * (graph_width / len(distance_history))
            y1 = graph_y + graph_height - (distance_history[i-1] / max_dist) * graph_height
            x2 = graph_x + i * (graph_width / len(distance_history))
            y2 = graph_y + graph_height - (distance_history[i] / max_dist) * graph_height
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    
    # Heading error graph
    graph_y2 = graph_y - graph_height - 10
    if heading_error_history:
        max_error = max(45, max(abs(h) for h in heading_error_history))
        for i in range(1, len(heading_error_history)):
            x1 = graph_x + (i-1) * (graph_width / len(heading_error_history))
            y1 = graph_y2 + graph_height - ((heading_error_history[i-1] + max_error) / (2 * max_error)) * graph_height
            x2 = graph_x + i * (graph_width / len(heading_error_history))
            y2 = graph_y2 + graph_height - ((heading_error_history[i] + max_error) / (2 * max_error)) * graph_height
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Draw graph borders and labels
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), (255, 255, 255), 1)
    cv2.rectangle(frame, (graph_x, graph_y2), (graph_x + graph_width, graph_y2 + graph_height), (255, 255, 255), 1)
    cv2.putText(frame, "Distance (cm)", (graph_x, graph_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "Heading Error ()", (graph_x, graph_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Display status information
    status_y = 20
    cv2.putText(frame, f"Rover: ({rover_pos[0]/PIXELS_PER_CM:.1f}, {rover_pos[1]/PIXELS_PER_CM:.1f}) cm", 
               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Target: ({target_pos[0]/PIXELS_PER_CM:.1f}, {target_pos[1]/PIXELS_PER_CM:.1f}) cm", 
               (10, status_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Distance: {distance/PIXELS_PER_CM:.1f} cm", 
               (10, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Heading: {rover_heading:.1f}°", 
               (10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Heading Error: {current_heading_error:.1f}°", 
               (10, status_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Command: {last_command} @ {last_speed}", 
               (10, status_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Navigation: {'ACTIVE' if navigation_active else 'INACTIVE'}", 
               (10, status_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Speed Range: 120-200 (Current: {last_speed})", 
               (10, status_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame, distance, current_heading_error

def simulation_loop():
    global rover_pos, rover_heading, target_pos, navigation_active, final_results
    global last_command, last_speed, distance_history, heading_error_history, last_controller_time
    global command_queue, CURRENT_SPEED

    # Setup the navigator for simulation
    setup_simulation_navigator()
    
    cv2.namedWindow("Rover Navigation Simulation")
    
    while True:
        current_time = time.time()
        
        # Process any pending commands 
        if command_queue:
            commands_to_execute = [cmd for cmd in command_queue if cmd[0] <= current_time]
            for cmd_data in commands_to_execute:
                _, cmd_to_execute, speed_to_execute = cmd_data
                simulated_rover_movement(cmd_to_execute, speed_to_execute)
                command_queue.remove(cmd_data)
        
        # Draw simulation frame with NOISY sensor readings for display
        frame, true_distance, true_heading_error = draw_simulation_frame()
        
        noisy_distance, noisy_heading_error, _ = get_noisy_sensor_readings()

        # Run navigation system when active
        if navigation_active and current_time - last_controller_time >= CONTROLLER_INTERVAL:
           
            result = navigator.execute_navigation(
                rover_pos/PIXELS_PER_CM,  
                rover_heading,            
                target_pos/PIXELS_PER_CM
            )
            
            if result == "TARGET_REACHED":
                final_results = navigator.get_final_results()
                print("🎯 TARGET REACHED in simulation!")
                if final_results:
                    print(f"🎯 Final distance: {final_results['final_distance']:.1f}cm")
                    print(f"🎯 Final heading error: {final_results['final_heading_error']:.1f}°")
                
                stop_navigation()
                navigation_active = False
            
            last_controller_time = current_time

        # Display final results if navigation completed
        if final_results is not None and not navigation_active:
            results = final_results
            cv2.putText(frame, "=== NAVIGATION COMPLETE ===", (WINDOW_SIZE[0]//2 - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Final Distance: {results['final_distance']:.2f} cm", 
                       (WINDOW_SIZE[0]//2 - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Final Heading Error: {results['final_heading_error']:.2f}°", 
                       (WINDOW_SIZE[0]//2 - 150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display real time simulation parameters
        param_y = WINDOW_SIZE[1] - 120
        cv2.putText(frame, f"Real-world modeling:", (10, param_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"Speed: {CURRENT_SPEED:.0f} (cmd: {last_speed})", (10, param_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"Queued commands: {len(command_queue)}", (10, param_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"Sensor noise: ±{SENSOR_NOISE_DISTANCE}cm, ±{SENSOR_NOISE_HEADING}°", (10, param_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.imshow("Rover Navigation Simulation", frame)

        # Handle keyboard input
        key = cv2.waitKey(int(1000/FPS)) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  
            if navigation_active:
                stop_navigation()
            else:
                start_navigation()
        elif key == ord('r'):  # R - reset rover position
            rover_pos = np.array([100.0, 100.0], dtype=float)
            rover_heading = 0.0
            navigation_active = False
            final_results = None
            CURRENT_SPEED = 0
            CURRENT_TURN_RATE = 0
            command_queue = []
            sensor_readings = []
            print("🔄 Rover position and simulation state reset")
        elif key == ord('c'):  # C - clear history
            distance_history = []
            heading_error_history = []
            print("📊 History cleared")

    # Restore navigator before exiting
    restore_navigator()
    cv2.destroyAllWindows()

# Mouse callback for target setting
def mouse_callback(event, x, y, flags, param):
    global target_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        target_pos = np.array([x, y], dtype=float)
        print(f"🎯 Target set to: ({x/PIXELS_PER_CM:.1f}, {y/PIXELS_PER_CM:.1f}) cm")

# Main function to run simulation
if __name__ == "__main__":
    cv2.namedWindow("Rover Navigation Simulation")
    cv2.setMouseCallback("Rover Navigation Simulation", mouse_callback)
    simulation_loop()