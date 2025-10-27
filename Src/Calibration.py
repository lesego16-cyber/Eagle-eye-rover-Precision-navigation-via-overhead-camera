import numpy as np
import cv2
import glob
import os

# Camera calibration using Zhang's method
def calibrate_camera(images_folder, chessboard_size, square_size):
    # Create output folder
    output_folder = "calibration_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    
    # Obtain all image paths
    images_path = os.path.join(images_folder, '*.jpg')
    images = glob.glob(images_path)
    
    if not images:
        print(f"No images found in {images_folder}")
        return None, None, None, None
    
    print(f"Found {len(images)} images for calibration")
    
    # Process each image and store images with corners for final display
    successful_calibrations = 0
    corner_images = []  
    
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"Could not read image: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            successful_calibrations += 1
            objpoints.append(objp)
            
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)
            
            # Draw and save the detected corners
            img_with_corners = cv2.drawChessboardCorners(img.copy(), chessboard_size, corners_refined, ret)
            
            # Save the images with detected corners as output
            filename = os.path.basename(fname)
            corners_output_path = os.path.join(output_folder, f"corners_{filename}")
            cv2.imwrite(corners_output_path, img_with_corners)
            
            # Store the images for final comparison
            corner_images.append((img, img_with_corners, filename))
            
            print(f"Image {i+1}/{len(images)}: Corners found and saved -> {corners_output_path}")
        else:
            print(f"Image {i+1}/{len(images)}: Could not find corners")
    
    if successful_calibrations < 5:
        print(f"Only {successful_calibrations} successful calibrations. Need at least 5 for good results.")
        return None, None, None, None
    
    print(f"\nCalibrating camera using {successful_calibrations} images...")
    
    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Calculate the reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"Calibration completed!")
    print(f"Reprojection error: {mean_error/len(objpoints):.5f} pixels")
    
    # Create final comparison images for the first few successful calibrations
    create_final_comparisons(corner_images, camera_matrix, dist_coeffs, output_folder)
    
    return camera_matrix, dist_coeffs, rvecs, tvecs

def create_final_comparisons(corner_images, camera_matrix, dist_coeffs, output_folder):
    """Create comparison images showing original, corners, and calibrated result"""
    print("\nCreating final comparison images...")
    
    # Process the first 5 images
    num_images = min(5, len(corner_images))
    
    for i in range(num_images):
        original, corners_img, filename = corner_images[i]
        
        # Obtain the calibrated version of the image
        calibrated = cv2.undistort(original, camera_matrix, dist_coeffs)
        
        # Create a triple comparison image displaying the original image + detected corners + calibrated image
        max_height = 600
        if original.shape[0] > max_height:
            scale = max_height / original.shape[0]
            width = int(original.shape[1] * scale)
            original = cv2.resize(original, (width, max_height))
            corners_img = cv2.resize(corners_img, (width, max_height))
            calibrated = cv2.resize(calibrated, (width, max_height))
        
        combined = np.hstack((original, corners_img, calibrated))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, 'Original', (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, 'Corners Detected', (original.shape[1] + 10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, 'Calibrated', (2*original.shape[1] + 10, 30), font, 0.7, (0, 255, 0), 2)
        
        cv2.line(combined, (original.shape[1], 0), (original.shape[1], combined.shape[0]), (255, 255, 255), 2)
        cv2.line(combined, (2*original.shape[1], 0), (2*original.shape[1], combined.shape[0]), (255, 255, 255), 2)
        
        # Save the comparison images
        comparison_path = os.path.join(output_folder, f"comparison_{filename}")
        cv2.imwrite(comparison_path, combined)
        print(f"Saved comparison image: {comparison_path}")
    
    # Create a summary image displaying the camera's intrinsic and extrinsic parameters
    create_parameters_image(camera_matrix, dist_coeffs, output_folder)

def create_parameters_image(camera_matrix, dist_coeffs, output_folder):
    """Create an image showing the camera calibration parameters"""
    param_img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 40
    
    # Title
    cv2.putText(param_img, 'CAMERA CALIBRATION RESULTS', (30, y_offset), font, 0.8, (0, 255, 255), 2)
    y_offset += 50
    
    # Camera matrix
    cv2.putText(param_img, 'Camera Matrix:', (30, y_offset), font, 0.6, (0, 255, 0), 1)
    y_offset += 30
    
    for i, row in enumerate(camera_matrix):
        text = f"[{row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f}]"
        cv2.putText(param_img, text, (50, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    y_offset += 20
    
    # Distortion coefficients
    cv2.putText(param_img, 'Distortion Coefficients:', (30, y_offset), font, 0.6, (0, 255, 0), 1)
    y_offset += 30
    
    dist_text = "["
    for coeff in dist_coeffs.ravel():
        dist_text += f"{coeff:7.4f} "
    dist_text += "]"
    
    cv2.putText(param_img, dist_text, (50, y_offset), font, 0.5, (255, 255, 255), 1)
    
    # Save the summary of obtained parameters image
    param_path = os.path.join(output_folder, "calibration_parameters.jpg")
    cv2.imwrite(param_path, param_img)
    print(f"Saved parameters image: {param_path}")

def save_calibration_results(camera_matrix, dist_coeffs, filename='camera_calibration.npz'):
    """Save calibration results to a file"""
    np.savez(filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Calibration results saved to {filename}")

def load_calibration_results(filename='camera_calibration.npz'):
    """Load calibration results from a file"""
    data = np.load(filename)
    return data['camera_matrix'], data['dist_coeffs']

# Main function
if __name__ == "__main__":
    images_folder = "Calibration_images" 
    chessboard_size = (10, 7)  # Number of inner corners (columns, rows) of the chessboard image
    square_size = 25.0  # Size of one square in millimeters
    
    # Calibrate camera
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
        images_folder, chessboard_size, square_size
    )
    
    if camera_matrix is not None:
        # Save calibration results
        save_calibration_results(camera_matrix, dist_coeffs)
        
        print(f"\n✅ All calibration results saved to 'calibration_output' folder!")
        print("📁 Folder contains:")
        print("   - Images with detected corners (corners_*.jpg)")
        print("   - Comparison images (comparison_*.jpg)")
        print("   - Calibration parameters (calibration_parameters.jpg)")
        print("   - Calibration data (camera_calibration.npz)")
        
    else:
        print("Camera calibration failed!")