
import numpy as np
from physics_sim import PhysicsSim

class Task_quadricopter():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, Vel_max=150):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
            Vel_max: maximum speed allowed to the quadricopter
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        #expands the state vector to include linear velocity and angular velocity information
        self.state_size = self.action_repeat * 12
        self.action_low = 0
        self.action_high = 1000
        self.action_size = 4
        self.Velocity_max=np.array(3*[Vel_max])

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        
        MAX_speed=0
        #causes the agent to fly as fast as possible without exceeding the maximum speed limit
        if self.sim.v is not None:
            MAX_speed=np.minimum(self.sim.v, self.Velocity_max).sum()
            MAX_speed=10*(MAX_speed) #Reschedule at maximum speed 
            
        stable=0
        """
        stabilizes the flight of the agent causing it to be punished if it rotates around any of the axes (Euler angles), making the sum of         the angular velocities in each of the axes is as small as possible.
        """
        if self.sim.angular_v is not None:
            stable=0.0002*(np.power(self.sim.angular_v[0],2)+np.power(self.sim.angular_v[1],2)+np.power(self.sim.angular_v[2],2))
        """
        The function of the term target is to get the agent closer and closer to the target, and the proportion between the current                 distance and the target distance of the agent 
        """
        target = 0.0003*(abs(self.sim.pose[:3] - self.target_pos)).sum()
                              
        """Uses current pose of sim to return reward."""
        reward = 5.+ MAX_speed - target - stable
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            #expands the state vector to include linear velocity and angular velocity information
            state_environment=[]
            state_environment.append(self.sim.pose.reshape((1,-1))[0])
            state_environment.append(self.sim.v.reshape((1,-1))[0])
            state_environment.append(self.sim.angular_v.reshape((1,-1))[0])
            state_environment = np.concatenate(state_environment)
            state = np.concatenate([state_environment] * self.action_repeat) 
            pose_all.append(state_environment)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #expands the state vector to include linear velocity and angular velocity information
        state_environment=[]
        state_environment.append(self.sim.pose.reshape((1,-1))[0])
        state_environment.append(self.sim.v.reshape((1,-1))[0])
        state_environment.append(self.sim.angular_v.reshape((1,-1))[0])
        state_environment = np.concatenate(state_environment)
        state = np.concatenate([state_environment] * self.action_repeat) 
        return state
