import math
import time

def controller(rover_front_position, rover_heading, target_position, navigation_mode, heading_correction_start_time):
    if rover_front_position is None or rover_heading is None or target_position is None:
        return 'S', 0

    #Computing distance and heading errors using the rover and target positions
    dx = target_position[0] - rover_front_position[0]
    dy = target_position[1] - rover_front_position[1]
    distance = math.hypot(dx, dy)

    desired_heading = math.degrees(math.atan2(dy, dx)) % 360
    error_heading = desired_heading - (rover_heading % 360)
    if error_heading > 180:
        error_heading -= 360
    elif error_heading < -180:
        error_heading += 360

    abs_error = abs(error_heading)

    # Phase-based control 
    if navigation_mode == "APPROACH":
        # Navigate toward target with heading corrections
        if distance > 80.0:
            speed = 180
        elif distance > 50.0:
            speed = 170  
        elif distance > 30.0:
            speed = 160
        elif distance > 20.0:
            speed = 150
        else:
            speed = 140

    
        if abs_error > 25:  # Large error - turn first
            cmd = 'R' if error_heading > 0 else 'L'
        elif abs_error > 10:  # Medium error - gentle turn
            cmd = 'R' if error_heading > 0 else 'L'
            speed = max(140, speed - 20)  
        else:  # Good heading - move forward
            cmd = 'F'

    elif navigation_mode == "HEADING_CORRECTION":
        # Reducing heading error to <10° within 5 seconds
        speed = 130
        
        if abs_error > 10:  
            cmd = 'R' if error_heading > 0 else 'L'
        else:  
            cmd = 'S'

    elif navigation_mode == "FINAL_APPROACH":
        # Move forward or backward to stop at 10cm
        speed = 120
        
        if distance > 10.5:  
            cmd = 'F'
        elif distance < 9.5:  
            cmd = 'B'
        else:  # Within the target zone, stop regardless of heading
            cmd = 'S'

    else:
        cmd = 'S'
        speed = 0

    return cmd, speed