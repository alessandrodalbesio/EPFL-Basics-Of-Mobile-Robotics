##### Modules import #####
from tdmclient import ClientAsync # Import thymio library
from time import * # Import time
import traceback # Import traceback
from utils.settings import * # Import settings
from global_navigation.main import * # Import global navigation
from local_navigation.main import * # Import local navigation
from vision.camera import * # Import camera
from vision.map import * # Import map library
from filtering.kalman_filter import * # Import Kalman filter
from utils.tools import * # Import tools
import pandas as pd

async def demo(save_video_path=None):
    save_video = save_video_path is not None

    ##### Variables definition #####
    start = goal = None
    robotPos_estimated = robotOrientation_estimated = None
    time_start = time()
    motors_speed = lambda lm, rm: { "motor.left.target": [round(lm)], "motor.right.target": [round(rm)] }
    #Wheel Speed Measures
    d_wl = d_wr = 0

    try:
        ##### Connection to the robot #####
        client = ClientAsync()
        node = await client.wait_for_node()
        await node.lock()

        ##### Map creation and variables definition #####
        cam = Camera(save_video=save_video, save_video_name=save_video_path)
        map = Map(cam, number_of_obstacles=NUMBER_OF_OBSTACLES, robot_size=ROBOT_SIZE)
        map.findObstacles()
        glob = Global(map.obstacles)
        local = Local()
        cam.obstacles = map.obstacles
        angle_hist = []
        pos_hist = []

        ##### Dataframe definition #####
        df_wheel_speeds_measured = pd.DataFrame(columns = ['left_speed_measured', 'right_speed_measured', 'camera_state'])
        df_wheel_speeds_predicted = pd.DataFrame(columns = ['left_speed_predicted', 'right_speed_predicted', 'camera_state'])
        df_wheel_speeds_commanded = pd.DataFrame(columns = ['left_speed_commanded', 'right_speed_commanded', 'camera_state'])
        df_pos_measured = pd.DataFrame(columns = ['x', 'y', 'camera_state'])
        df_pos_estimated = pd.DataFrame(columns = ['x', 'y', 'camera_state'])
        df_orientation_measured = pd.DataFrame(columns = ['theta', 'time', 'camera_state'])
        df_orientation_estimated = pd.DataFrame(columns = ['theta', 'time', 'camera_state'])

        ##### Loop #####
        while not glob.goal_reached:        
            # Final and initial position estimation and path planning
            if start is None or goal is None:
                # Find initial and final positions with the camera
                start, initialOrientation, goal = map.getInitialFinalData()
                
                # Find the optimal path with global navigation
                glob.find_optimal_path(start, goal)
                
                # Initialize the kalman filter
                initialOrientation = (initialOrientation + 2 * np.pi) % (2 * np.pi)
                if initialOrientation > np.pi:
                    initialOrientation = initialOrientation - 2 * np.pi
                kalman = KalmanFilter(map.convertToCm([start])[0], initialOrientation)
                robotPos_estimated = np.array([start[0], start[1]])
                
                # Define attributes for the real time camera display
                cam.startPosition = start
                cam.goalPosition = goal
                cam.optimalPath = glob.optimal_path

                # Define variables for sampling
                time_last_sample = time()
                time_sampling = None
                
                # Skip the first iteration
                continue

            # Sensing
            await node.wait_for_variables()
            prox_horizontal_measured = node["prox.horizontal"]
            left_speed_measured = node["motor.left.speed"]
            right_speed_measured = node["motor.right.speed"]
            robotPos_measured, robotPos_measured_cm, robotOrientation_measured = map.cameraRobotSensing() # Robot position and orientation from the camera

            # Kidnapping management
            if robotPos_estimated is not None and robotPos_measured is not None and np.linalg.norm(robotPos_measured-robotPos_estimated) > KIDNAPPING_THRESH:
                # Turn off the motors
                await node.set_variables(motors_speed(0,0))

                # Force a new path planning
                start = None
                goal = None

                # Wait for some time (just for visual feedback)
                await client.sleep(SLEEP_TIME_AFTER_KIDNAPPING)

                # Skip the rest of the loop
                continue
            else:
                # Position estimation
                time_sampling = time() - time_last_sample

                # Understand if the camera is on or off and transform the angle in the correct way
                cam_x = cam_y = cam_theta = -1
                camera_state = 'on'
                if(robotPos_measured is None):
                    camera_state = 'off'
                else:
                    cam_x = robotPos_measured_cm[0]
                    cam_y = robotPos_measured_cm[1]
                    cam_theta = conv_2pi_to_pi(robotOrientation_measured)

                # Kalman filter
                [pos_estimated_x, pos_estimated_y, pos_estimated_theta, sp_estimated_lw, sp_estimated_rw] = kalman.update_kalman(d_wl, d_wr, left_speed_measured, right_speed_measured, camera_state, time_sampling, np.array([cam_x, cam_y, cam_theta]))
                robotPos_estimated = np.array([pos_estimated_x, pos_estimated_y])
                robotPos_estimated = map.convertToPx([robotPos_estimated])[0]
                robotOrientation_estimated = pos_estimated_theta
                
                angle_hist.append(robotOrientation_estimated)
                pos_hist.append(robotPos_estimated)
                
                if(len(angle_hist) > 5):
                    angle_hist.pop(0)
                if(len(pos_hist) > 5):
                    pos_hist.pop(0)

                # Control
                    
                if(robotPos_measured is None or robotOrientation_measured is None):
                    robotOrientation_estimated = np.mean(angle_hist)
                    
                robotOrientation_estimated = conv_pi_to_2pi(robotOrientation_estimated)
                angle_goal = glob.compute_angle_traj(robotPos_estimated)
                if local.local_obstacle(prox_horizontal_measured):
                    motorLeft,motorRight = local.local_controller(prox_horizontal_measured, robotOrientation_estimated, angle_goal)
                    glob.local_goal_point_update(robotPos_estimated)
                else:
                    motorLeft = motorRight = 0
                    if not (robotPos_estimated is None or robotOrientation_estimated is None):
                        motorLeft,motorRight = glob.global_controller(robotPos_estimated, robotOrientation_estimated, left_speed_measured, right_speed_measured)
                d_wl = motorLeft - sp_estimated_lw
                d_wr = motorRight - sp_estimated_rw
                
                # Actuation
                await node.set_variables(motors_speed(motorLeft,motorRight))
                # Save the data
                if(camera_state == 'off'):
                    robotPos_measured = robotPos_estimated
                    robotOrientation_measured = robotOrientation_estimated
                df_wheel_speeds_measured.loc[len(df_wheel_speeds_measured)] = [left_speed_measured, right_speed_measured, camera_state]
                df_wheel_speeds_predicted.loc[len(df_wheel_speeds_predicted)] = [sp_estimated_lw, sp_estimated_rw, camera_state]
                df_wheel_speeds_commanded.loc[len(df_wheel_speeds_commanded)] = [motorLeft, motorRight, camera_state]
                df_pos_measured.loc[len(df_pos_measured)] = [robotPos_measured[0], robotPos_measured[1], camera_state]
                df_pos_estimated.loc[len(df_pos_estimated)] = [robotPos_estimated[0], robotPos_estimated[1], camera_state]
                df_orientation_measured.loc[len(df_orientation_measured)] = [robotOrientation_measured, time(), camera_state]
                df_orientation_estimated.loc[len(df_orientation_estimated)] = [robotOrientation_estimated, time(), camera_state]
                    
                time_last_sample = time()

            # Update the camera attributes
            cam.robotMeasuredPosition = robotPos_measured
            cam.robotMeasuredOrientation = robotOrientation_measured
            cam.robotEstimatedPosition = robotPos_estimated
            cam.robotEstimatedOrientation = robotOrientation_estimated
            
            # Display the real time data
            if(cam.display()):
                break
            await client.sleep(0.1)
    except Exception as e:
        traceback.print_exc()
    finally:
        # Stop the robot
        await node.set_variables(motors_speed(0,0))

        # Unlock the robot
        await node.unlock()

        # Turn off the camera
        cam.release()
        cv2.destroyAllWindows()
            
        df_wheel_speeds_measured.to_csv('wheel_speeds_measured.csv')
        df_wheel_speeds_predicted.to_csv('wheel_speeds_predicted.csv')
        df_wheel_speeds_commanded.to_csv('wheel_speeds_commanded.csv')
        df_pos_measured.to_csv('pos_measured.csv')
        df_pos_estimated.to_csv('pos_estimated.csv')
        df_orientation_measured.to_csv('orientation_measured.csv')
        df_orientation_estimated.to_csv('orientation_estimated.csv')

        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax.plot(df_wheel_speeds_measured['left_speed_measured'], label='left_speed_measured')
        ax2.plot(df_wheel_speeds_measured['right_speed_measured'], label='right_speed_measured')
        ax.plot(df_wheel_speeds_predicted['left_speed_predicted'], label='left_speed_predicted')
        ax2.plot(df_wheel_speeds_predicted['right_speed_predicted'], label='right_speed_predicted')
        ax.plot(df_wheel_speeds_commanded['left_speed_commanded'], label='left_speed_commanded')
        ax2.plot(df_wheel_speeds_commanded['right_speed_commanded'], label='right_speed_commanded')
        ax.legend()
        ax.set_title('Wheel speeds')
        
        
        fig3, ax3 = plt.subplots()
        ax3.plot(df_pos_measured['x'], df_pos_measured['y'], label='Measured position')
        ax3.plot(df_pos_estimated['x'], df_pos_estimated['y'], label='Estimated position')
        ax3.legend()
        ax3.set_title('Position')
        
        fig4, ax4 = plt.subplots()
        ax4.plot(df_orientation_measured['time'], df_orientation_measured['theta'], label='Measured orientation')
        ax4.plot(df_orientation_estimated['time'], df_orientation_estimated['theta'], label='Estimated orientation')
        ax4.legend()
        ax4.set_title('Orientation')
        
        plt.show()
        