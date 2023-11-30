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
        self.speed_conv = 0.035 #16/500 #Measured in Ex8, converts thymio speed to cm/s
    
    def robot_speed(self, speed_left, speed_right):
        return ((speed_left + speed_right) / 2.0)

    def update_kalman(self, wheel_speed_left, wheel_speed_right, camera_state, dt, camera_data):
        if(dt == None):
            return self.x[0], self.x[1], self.x[2]
        self.dt = dt
        #angle = camera_data[2]
        angle = self.x[2]
        if(((camera_data[0] == -1) and (camera_data[1] == -1) and (camera_data[2] == -1))):
            camera_state = 'off'
        v = self.robot_speed(wheel_speed_left*self.speed_conv, wheel_speed_right*self.speed_conv)  # Robot speed
        #print("Speed in cm/s : ",v)
        self.dtheta = self.dt*(wheel_speed_right - wheel_speed_left) / self.wheelbase  # Robot angular speed
        #print("dt : ", self.dt)
        self.ds = v * self.dt  # Robot displacement
        prev_x = self.x
        self.A = np.array([
            [1, 0, 0],#self.ds * np.cos(self.x[2]+self.dtheta/2)],
            [0, 1, 0],#self.ds * np.sin(self.x[2]+self.dtheta/2)],
            [0, 0, 1],
        ])

        self.B = np.array([
            [0.5 * self.dt * np.cos(angle), 0.5 * self.dt * np.cos(angle)],#self.x[2]), 0.5 * self.dt * np.cos(self.x[2])],
            [0.5 * self.dt * np.sin(angle), 0.5 * self.dt * np.sin(angle)],#self.x[2]), 0.5 * self.dt * np.sin(self.x[2]
            [-1.0 / self.wheelbase, 1.0 / self.wheelbase],
        ])
        # State vector: [x, y, theta]
        u = [wheel_speed_left*self.speed_conv, wheel_speed_right*self.speed_conv]  # Replace with actual wheel velocity inputs
        self.x = self.A.dot(self.x) + np.dot(self.B,u)
        #print("Deg change : ", np.dot(self.B,u))
        pred_x = self.x
        #print("Speed   : ", v)
        #print("A1 = ", self.A)
        #print("Predicted advance : ", np.dot(self.B,u))
        #print("Predicted position: ", self.x)
        if(camera_state == 'on'):
            # Prediction covariance
            #self.P = self.B.dot(self.Q).dot(self.B.T) + self.P
            self.P = (self.A.dot(self.P).dot(self.A.T) + self.Q)
            #print("A = ", self.A)
            #print("P = ", self.P)

            # Update
            self.S = self.H.dot(self.P).dot(self.H.T) + self.R
            self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.S))
            #print("K = ", self.K)
            self.z = camera_data  # Replace with actual sensor measurements (X GPS, Y GPS, Theta GPS)
            self.x = pred_x + self.K.dot(self.z - pred_x)
            actual_x = self.z
            #print("Prev position : ", prev_x, " , Predicted position: ", pred_x, " , Actual position: ", actual_x)
            #print("Dif = ", self.z - self.H.dot(pred_x))
            # Update covariance
            self.P = (np.eye(3) - self.K.dot(self.H)).dot(self.P)
        else:
            self.x = self.x
        # Extract final estimates
        estimated_position = self.x[:2]
        estimated_orientation = self.x[2]
        return estimated_position[0], estimated_position[1], estimated_orientation
        
