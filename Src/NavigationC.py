import math
import requests
import time
import csv
import os
from Controller import controller

class VectorNavigator:
    def __init__(self):
        self.rover_ip = "192.168.137.211"
        self.base_url = f"http://{self.rover_ip}"
        self.navigation_active = False
        self.obstacle_detection_enabled = True
        self.final_rover_position = None
        self.final_rover_heading = None
        self.final_distance = None
        self.final_heading_error = None
        self.last_print_time = time.time()
        self.print_interval = 2.0
        

        self.navigation_mode = "APPROACH"
        self.heading_correction_start_time = 0
        self.max_heading_correction_time = 5.0  
        
        
        self.last_command = None
        self.last_speed = 0
        
        # Initialize CSV file for final navigation results logging
        self.final_results_file = "final_navigation_results.csv"
        self.initialize_csv_file()
        
        print(f"🤖 Rover IP set to: {self.rover_ip}")

    def initialize_csv_file(self):
        """Initialize CSV file with exact column headers"""
        try:
            with open(self.final_results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow([
                        "Timestamp", "TargetX", "TargetY", "FinalRoverX", "FinalRoverY", 
                        "FinalDistance", "FinalHeading", "FinalHeadingError", "Status"
                    ])
            print(f"✅ Results file ready: {self.final_results_file}")
        except Exception as e:
            print(f"⚠️ Could not create results file: {e}")
            self.final_results_file = None

    def check_rover_connection(self):
        """Check if rover is connected"""
        try:
            r = requests.get(f"{self.base_url}/", timeout=3.0)
            return r.status_code == 200 and "Rover Ready" in r.text
        except Exception as e:
            print(f"❌ Rover connection failed: {e}")
            return False

    def _map_cmd_for_arduino(self, cmd):
        if cmd is None:
            return 'S'
        c = str(cmd).strip().upper()
        if c in ('F', 'FORWARD'): return 'F'
        elif c in ('L', 'LEFT'): return 'L'
        elif c in ('R', 'RIGHT'): return 'R'
        elif c in ('B', 'BACKWARD'): return 'B'
        elif c in ('STOP', 'S'): return 'S'
        else: return 'S'

    def set_obstacle_detection_mode(self, enable_obstacle_detection):
        """Enable or disable obstacle detection"""
        if enable_obstacle_detection != self.obstacle_detection_enabled:
            mode_command = "NORMAL_MODE" if enable_obstacle_detection else "TARGET_APPROACH"
            try:
                url = f"{self.base_url}/control?cmd={mode_command}&speed=0"
                r = requests.get(url, timeout=3.0)
                if r.status_code == 200:
                    self.obstacle_detection_enabled = enable_obstacle_detection
                    print(f"🛡️ Obstacle detection: {'ENABLED' if enable_obstacle_detection else 'DISABLED'}")
            except Exception:
                pass

    def send_to_rover(self, command, speed=255):
        short_cmd = self._map_cmd_for_arduino(command)
        
        # Store the last command and speed for logging
        self.last_command = short_cmd
        self.last_speed = speed
        
        try:
            url = f"{self.base_url}/control?cmd={short_cmd}&speed={speed}"
            
            # Print status
            current_time = time.time()
            if current_time - self.last_print_time >= self.print_interval:
                print(f"➡️ [{self.navigation_mode}] {short_cmd} @ {speed}")
                self.last_print_time = current_time
            
            r = requests.get(url, timeout=5.0)
            return r.status_code == 200

        except Exception as e:
            current_time = time.time()
            if current_time - self.last_print_time >= self.print_interval:
                print(f"❌ Failed to send '{short_cmd}': {e}")
                self.last_print_time = current_time
            return False

    def save_final_results(self, target_position, status="SUCCESS"):
        """Save final results to CSV with exact parameters"""
        if self.final_rover_position and self.final_rover_heading and self.final_distance is not None:
            try:
                if self.final_results_file:
                    with open(self.final_results_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            target_position[0],
                            target_position[1],
                            self.final_rover_position[0],
                            self.final_rover_position[1],
                            self.final_distance,
                            self.final_rover_heading,
                            self.final_heading_error,
                            status
                        ])
                    print(f"💾 Final results saved: {self.final_distance:.1f}cm, heading error: {self.final_heading_error:.1f}°")
                else:
                    print(f"📊 Final results (not saved): {self.final_distance:.1f}cm, heading error: {self.final_heading_error:.1f}°")
            except Exception as e:
                print(f"❌ Failed to save final results: {e}")

    def execute_navigation(self, rover_front_position, rover_heading, target_position):
        """FIXED: Use rover FRONT position instead of centroid position"""
        if not self.navigation_active:
            return False

        # Calculate the rover and target relative distancea and heading
        dx = target_position[0] - rover_front_position[0]
        dy = target_position[1] - rover_front_position[1]
        distance = math.hypot(dx, dy)
        
        desired_heading = math.degrees(math.atan2(dy, dx)) % 360
        heading_error = desired_heading - (rover_heading % 360)
        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360

        abs_heading_error = abs(heading_error)

        # Final stopping condition
        if 9.5 <= distance <= 10.5:
            print(f"🎯 TARGET REACHED! Distance: {distance:.1f}cm, Heading: {heading_error:.1f}°")
            
            self.final_rover_position = rover_front_position
            self.final_rover_heading = rover_heading
            self.final_distance = distance
            self.final_heading_error = heading_error
            
            self.save_final_results(target_position, "SUCCESS")
            self.set_obstacle_detection_mode(True)
            self.stop_navigation()
            return "TARGET_REACHED"

        # Implements a 3-phase state machine
        current_time = time.time()
        
        # 1. APPROACH: Navigate toward target until 20cm
        if self.navigation_mode == "APPROACH":
            if distance <= 20.0:
                print("🔄 Entering HEADING_CORRECTION mode (5 seconds)")
                print(f"📊 Starting heading: {heading_error:.1f}°")
                self.navigation_mode = "HEADING_CORRECTION"
                self.heading_correction_start_time = current_time
        
        # 2. HEADING_CORRECTION: Spend 5 seconds correcting heading  to <10°
        elif self.navigation_mode == "HEADING_CORRECTION":
            time_elapsed = current_time - self.heading_correction_start_time
            
        
            if int(time_elapsed) != int(time_elapsed - 0.1): 
                print(f"⏳ Heading correction: {time_elapsed:.1f}s / 5.0s | Error: {heading_error:.1f}°")
            
            # Check if state transition must take place
            if time_elapsed >= self.max_heading_correction_time:
                print(f"⏰ TIME UP! Final heading: {heading_error:.1f}°")
                print("🔄 Switching to FINAL_APPROACH")
                self.navigation_mode = "FINAL_APPROACH"
            elif abs_heading_error <= 10.0:
                print(f"✅ HEADING PERFECT! {heading_error:.1f}°")
                print("🔄 Switching to FINAL_APPROACH")
                self.navigation_mode = "FINAL_APPROACH"
        
        # 3. FINAL_APPROACH: Move to stop at 10cm
        elif self.navigation_mode == "FINAL_APPROACH":
            if distance > 25.0:
                print("🔄 Too far from target, back to HEADING_CORRECTION")
                self.navigation_mode = "HEADING_CORRECTION"
                self.heading_correction_start_time = current_time

        # Get command movements from the controller
        cmd, speed = controller(rover_front_position, rover_heading, target_position, self.navigation_mode, 
                               self.heading_correction_start_time)
        
        # Send command
        result = self.send_to_rover(cmd, speed)
        return result

    def start_navigation(self):
        """Start navigation"""
        if not self.check_rover_connection():
            print("❌ Cannot start - rover not connected")
            return False
        
        self.navigation_mode = "APPROACH"
        self.heading_correction_start_time = 0
        self.set_obstacle_detection_mode(True)
        self.navigation_active = True
        self.last_print_time = time.time()
        print("🚀 Navigation started")
        return True

    def stop_navigation(self):
        """Stop navigation"""
        self.set_obstacle_detection_mode(True)
        self.navigation_active = False
        self.send_to_rover('S', 0)
        print("🛑 Navigation stopped")
        return True