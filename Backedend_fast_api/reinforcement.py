import tensorflow as tf
from test import ModelManager

# Reward function
def get_reward(feedback):
    if feedback == "thumps-up":
        return 1.0  # Positive reward for thumbs up
    else:
        return -1.0  # Negative reward for thumbs down

# Model update function
def update_model_parameters(image, feedback):

    global accumulated_updates, feedback_count, save_threshold, model
    print(save_threshold)

    model = ModelManager.get_model()

    # Define loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)    

    with tf.GradientTape() as tape:

        # Forward pass through model
        outputs = model(image)
        
        # Check the structure of outputs
        print(outputs)

        # Adjust code to access logits based on the structure of outputs
        logits = outputs[0]  
        # Compute loss
        loss_value = loss(tf.constant([0]), logits)  
        # Apply reward
        reward = get_reward(feedback)
        loss_value *= -reward
        print(loss_value)

    # Backward pass and update model parameters
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Accumulate updates
    accumulated_updates.append(model.get_weights())  
    
    # Increment feedback count
    feedback_count += 1
    
    # Check if threshold for saving is reached
    if feedback_count >= save_threshold:
        # Apply accumulated updates to the model
        averaged_updates = average_updates(accumulated_updates)
        model.set_weights(averaged_updates)  
        
        # Save the model weights only
        model.save_weights('model.h5')
        
        # Reset accumulated updates and feedback count
        accumulated_updates = []
        feedback_count = 0

# Helper function to average accumulated updates
def average_updates(accumulated_updates):
    # Calculate average of accumulated updates
    averaged_updates = [tf.math.reduce_mean(update, axis=0) for update in zip(*accumulated_updates)]
    return averaged_updates

# Method to parse feedback and image and perform RL
def parse_feedback_and_image_prediction(image, feedback):
    # Update model parameters based on feedback and image
    update_model_parameters(image, feedback)

# Example usage
accumulated_updates = []  # Initialize accumulated updates list
feedback_count = 0  # Initialize feedback count
save_threshold = 1  # Set threshold for saving the model