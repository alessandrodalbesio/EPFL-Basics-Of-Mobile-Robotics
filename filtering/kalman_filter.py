import numpy as np


class KalmanFilter:
    def __init__(self, x0, angle):
        self.ds = 0.0
        self.dtheta = 0.0
        self.x = np.zeros((5, 1))
        self.dt = 0.0
        self.wheelbase = 8.9 #Measured on robot
        self.sigma_gpsx = 0.1
        self.sigma_gpsy = 0.1
        self.sigma_vx = 1.5
        self.sigma_vy = 1.5
        self.sigma_gpstheta = 0.9
        self.speed_conv = 0.045
        self.qv1 = 10 # Process noise Measured in Ex8 in cm2/s2
        self.qv2 = 0.0442 # Process noise Measured in Ex8 in cm2/s2
        
        
        self.A = np.diag([1, 1, 1, 1, 1])
        self.B = np.array([
            [1, 0],
            [0, 1],
            [-1.0 / self.wheelbase, 1.0 / self.wheelbase],
            [1, 0],
            [0, 1],
        ])
        self.H = np.array([
            [1, 0, 0, 0, 0],  # Camera x
            [0, 1, 0, 0, 0],  # Camera y
            [0, 0, 1, 0, 0],  # Camera theta
            [0, 0, 0, 1, 0],  # Camera dx
            [0, 0, 0, 0, 1],  # Camera dy
        ])
        self.Q = np.diag([self.qv1, self.qv1, self.qv1, self.qv2, self.qv2]) # Process noise covariance
        self.R = np.diag([self.sigma_gpsx**2, self.sigma_gpsy**2, self.sigma_gpstheta**2, self.sigma_vx**2, self.sigma_vy**2]) # Measurement noise covariance
        self.P = np.diag([1, 1, 1, 1, 1]) 
        self.x = [x0[0], x0[1], angle, 0, 0]
    
    def robot_speed(self, speed_left, speed_right):
        return ((speed_left + speed_right) / 2.0)

    def update_kalman(self, wsl_in, wsr_in, wheel_speed_left_meas, wheel_speed_right_meas, camera_state, dt, camera_data):
        if(dt == None):
            return self.x[0], self.x[1], self.x[2]
        self.dt = dt
        if(((camera_data[0] == -1) and (camera_data[1] == -1) and (camera_data[2] == -1))):
            camera_state = 'off'
        prev_x = self.x
        self.A = np.array([
            [1, 0, 0, 0.5 * self.dt * np.cos(self.x[2])*self.speed_conv, 0.5 * self.dt * np.cos(self.x[2])*self.speed_conv],
            [0, 1, 0, 0.5 * self.dt * np.sin(self.x[2])*self.speed_conv, 0.5 * self.dt * np.sin(self.x[2])*self.speed_conv],
            [0, 0, 1, -self.dt*self.speed_conv/self.wheelbase, self.dt*self.speed_conv/self.wheelbase],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])
        self.B = np.array([
            [0.5 * self.dt * np.cos(self.x[2])*self.speed_conv, 0.5 * self.dt * np.cos(self.x[2])*self.speed_conv],
            [0.5 * self.dt * np.sin(self.x[2])*self.speed_conv, 0.5 * self.dt * np.sin(self.x[2])*self.speed_conv],
            [-self.dt*self.speed_conv/self.wheelbase, self.dt*self.speed_conv/self.wheelbase],
            [1, 0],
            [0, 1]             
        ])
        # State vector: [x, y, theta]
        u = [wsl_in, wsr_in]  # Actual wheel velocity inputs
        prev_x = self.x
        self.x = self.A.dot(self.x) + np.dot(self.B,u)
        pred_x = self.x
        if(self.x[2] > np.pi):
            self.x[2] = self.x[2] - 2*np.pi
        elif(self.x[2] < -np.pi):
            self.x[2] = self.x[2] + 2*np.pi

        if(camera_state == 'on'):
            # Prediction covariance
            self.R = np.diag([self.sigma_gpsx**2, self.sigma_gpsy**2, self.sigma_gpstheta**2, self.sigma_vx**2, self.sigma_vy**2])
            self.P = (self.A.dot(self.P).dot(self.A.T) + self.Q)

            # Update
                #Intermediate Matrix
            self.S = self.H.dot(self.P).dot(self.H.T) + self.R
                #Update Kalman Gain
            self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.S))

            self.z = [camera_data[0], camera_data[1], camera_data[2], wheel_speed_left_meas, wheel_speed_right_meas]  # Actual sensor measurements (X GPS, Y GPS, Theta GPS)
            self.x = pred_x + self.K.dot(self.z - pred_x)

            # Update covariance
            self.P = (np.eye(5) - self.K.dot(self.H)).dot(self.P)
        else:
            # Prediction covariance
            self.R = np.diag([10000000, 10000000, 10000000, (self.sigma_vx*2)**2, (self.sigma_vy*2)**2]) #If camera is off, the Camera measurements are not used
            self.P = (self.A.dot(self.P).dot(self.A.T) + self.Q)
            # Update
            self.S = self.H.dot(self.P).dot(self.H.T) + self.R
            self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.S))
            self.z = [pred_x[0], pred_x[1], pred_x[2], wheel_speed_left_meas, wheel_speed_right_meas]  # Actual sensor measurements, with camera sending 0, 0, 0)
            self.x = pred_x + self.K.dot(self.z - pred_x)

        if(self.x[2] > np.pi):
            self.x[2] = self.x[2] - 2*np.pi
        elif(self.x[2] < -np.pi):
            self.x[2] = self.x[2] + 2*np.pi
        expected_change = pred_x - prev_x
        actual_change = self.z - pred_x
        # Extract final estimates
        estimated_position = self.x[:2]
        estimated_orientation = self.x[2]
        return estimated_position[0], estimated_position[1], estimated_orientation, self.x[3], self.x[4]
        
