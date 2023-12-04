import numpy as np



class KalmanFilter:
    def __init__(self, x0, angle):
        self.ds = 0.0
        self.dtheta = 0.0
        self.x = np.zeros((3, 1))
        self.dt = 0.0
        self.wheelbase = 8.9 #Measured on robot
        self.sigma_gpsx = 0.9
        self.sigma_gpsy = 0.9
        self.sigma_gpstheta = 0.9
        #self.turning_factor = 0.009
        
        #Tuned Factors: #These variables were tuned to minimize the error in the position estimation
        self.speed_conv = 0.15
        self.turning_factor = 0.075
        self.speed_correct = 0.3
        
        
        self.A = np.diag([1, 1, 1])
        self.B = np.array([
            [1, 0],
            [0, 1],
            [1.0 / self.wheelbase, -1.0 / self.wheelbase],
        ])
        self.H = np.array([
            [1, 0, 0],  # GPS x
            [0, 1, 0],  # GPS y
            [0, 0, 1],  # GPS theta
        ])
        self.qv = 0.0442 # Process noise Measured in Ex8 in cm2/s2
        self.Q = np.diag([self.qv, self.qv, self.qv])
        self.R = np.diag([self.sigma_gpsx**2, self.sigma_gpsy**2, self.sigma_gpstheta**2])
        self.P = np.diag([1, 1, 1])
        self.x = [x0[0], x0[1], angle]
    
    def robot_speed(self, speed_left, speed_right):
        return ((speed_left + speed_right) / 2.0)

    def update_kalman(self, wheel_speed_left, wheel_speed_right, camera_state, dt, camera_data, sc, ac, speed):
        if(sc!=0):
            self.speed_conv = sc
        if(ac!=0):
            self.turning_factor = ac
        if(speed!=0):
            self.speed_correct = speed
        if(dt == None):
            return self.x[0], self.x[1], self.x[2]
        self.dt = dt
        angle = self.x[2]
        if(((camera_data[0] == -1) and (camera_data[1] == -1) and (camera_data[2] == -1))):
            camera_state = 'off'
        v = self.robot_speed(wheel_speed_left*self.speed_conv, wheel_speed_right*self.speed_conv)  # Robot speed
        self.dtheta = self.dt*(wheel_speed_right - wheel_speed_left) / self.wheelbase  # Robot angular speed
        self.ds = v * self.dt  # Robot displacement
        prev_x = self.x
        self.A = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        self.B = np.array([
            [0.5 * self.dt * np.cos(angle)*self.speed_conv, 0.5 * self.dt * np.cos(angle)*self.speed_conv],
            [0.5 * self.dt * np.sin(angle)*self.speed_conv, 0.5 * self.dt * np.sin(angle)*self.speed_conv],
            [(-1.0*self.turning_factor)/ self.wheelbase, (1.0*self.turning_factor)/ self.wheelbase],
        ])
        # State vector: [x, y, theta]
        u = [wheel_speed_left*self.speed_correct, wheel_speed_right*self.speed_correct]  # Actual wheel velocity inputs
        self.x = self.A.dot(self.x) + np.dot(self.B,u)
        pred_x = self.x

        if(camera_state == 'on'):
            # Prediction covariance
            self.P = (self.A.dot(self.P).dot(self.A.T) + self.Q)

            # Update
            self.S = self.H.dot(self.P).dot(self.H.T) + self.R
            self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.S))
            self.z = camera_data  # Actual sensor measurements (X GPS, Y GPS, Theta GPS)
            self.x = pred_x + self.K.dot(self.z - pred_x)
            actual_x = self.z

            # Update covariance
            self.P = (np.eye(3) - self.K.dot(self.H)).dot(self.P)
        else:
            self.x = self.x
        # Extract final estimates
        estimated_position = self.x[:2]
        estimated_orientation = self.x[2]
        return estimated_position[0], estimated_position[1], estimated_orientation
        
