class Local:
    
    def local_obstacle(self, prox_horizontal):
        local_dist_thresh = 100
        return max(prox_horizontal) > local_dist_thresh
    
    def local_controller(self, prox_horizontal):
        # Defining nominal speed and speed gains
        nominal_speed     = 80
        obstSpeedGain   = [6, 4, -2, -6, -8] #divided by gain_scale
        gain_scale      = 100
        
        spLeft  = nominal_speed
        spRight = nominal_speed
        
        for i in range(5):
            spLeft += prox_horizontal[i] * obstSpeedGain[i] // gain_scale
            spRight += prox_horizontal[i] * obstSpeedGain[4 - i] // gain_scale
            
        if (spLeft == 0) and (spRight == 0):
            spLeft = -2
            spRight = -2
        
        return spLeft, spRight