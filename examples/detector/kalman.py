import numpy as np
from pykalman import KalmanFilter

# Initial state
initial_state_mean = [0,0,0]
initial_state_covariance = [[1,0,0],[0,1,0],[0,0,1]]

# Define Kalman filter parameters
transition_matrices = [[1,0,0],[0,1,0],[0,0,1]] #[[1]]#
observation_matrices = [[1,0,0],[0,1,0],[0,0,1]]
transition_covariance = [[0.1,0,0],[0,0.1,0],[0,0,0.1]]
observation_covariance = [[1,0,0],[0,1,0],[0,0,1]]
transition_offsets = [0,0,0]
observation_offsets = [0,0,0]

# Create the Kalman filter
kf = KalmanFilter(
    transition_matrices=transition_matrices, 
    observation_matrices=observation_matrices, 
    transition_covariance=transition_covariance, 
    observation_covariance=observation_covariance,
    transition_offsets=transition_offsets,
    observation_offsets=observation_offsets,
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance
)

# Initial state and covariance are the "previous" ones for the first iteration
previous_state_mean = initial_state_mean
previous_state_covariance = initial_state_covariance

# Assume you have 10 measurements
measurements = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Loop over your measurements
for i in range(3):
    # Predict the next state
    predicted_state_mean, predicted_state_covariance = kf.filter_update(
        previous_state_mean, 
        previous_state_covariance
    )
    print("measurement: ", measurements[i])
    print("predicted_state_mean: ", predicted_state_mean)
    print("predicted_state_covariance: ", predicted_state_covariance)
    # Update the state estimate with the new measurement
    updated_state_mean, updated_state_covariance = kf.filter_update(
        predicted_state_mean, 
        predicted_state_covariance,
        observation=measurements[i]
    )
    print("updated_state_mean: ", updated_state_mean)
    print("updated_state_covariance: ", updated_state_covariance)
    # The updated state becomes the "previous" state for the next iteration
    previous_state_mean, previous_state_covariance = updated_state_mean, updated_state_covariance

    # Print the updated state mean
    print(updated_state_mean)

