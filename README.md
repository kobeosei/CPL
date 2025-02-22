# **CPL Training for Trajectory Comparison Model**

This repository implements the **CPL (Comparison-based Preference Learning)** algorithm for training a model to compare trajectories using different types of feedback (comparisons, demonstrations, and e-stop). The model is designed for reinforcement learning tasks where the agent's trajectory and feedback are used to learn a preference-based ranking function.

## **Features**

- **Comparison Feedback**: The model is trained based on pairwise comparisons of trajectories.
- **Demonstration Feedback**: Generates preference labels for trajectories based on the reward.
- **E-stop Feedback**: The model is trained to avoid highly undesirable states (e.g., reaching a "very bad" constraint).
- **Logging and Evaluation**: Training progress is logged using TensorBoard, with evaluation metrics such as accuracy and reward.

## **Prerequisites**

To run this project, you'll need the following packages:

- Python 3.8
- PyTorch
- Numpy
- Matplotlib
- Pandas
- TensorBoard

You can install the required dependencies using `pip`:

```bash
pip install torch numpy matplotlib pandas tensorboard
```

## **Usage**

### Running the training

To train the model, use the following command. This command allows you to specify the feedback type (comparisons, demonstrations, or e-stop), and optionally enable rendering of the agent's trajectory during training.

```bash
python train_cpl.py --feedback <feedback_type> [--render]
```

- `<feedback_type>`: Choose from `'comparisons'`, `'demonstrations'`, or `'e_stop'`.
- `--render`: Optional. Set this flag if you want to visualize the agent's trajectory during training.

### Example commands:

1. **Using Comparisons feedback**:

```bash
python train_cpl.py --feedback comparisons
```

2. **Using Demonstrations feedback**:

```bash
python train_cpl.py --feedback demonstrations
```

3. **Using E-stop feedback**:

```bash
python train_cpl.py --feedback e_stop
```

If you want to enable rendering to visualize the agent's trajectory in the environment, add the `--render` flag:

```bash
python train_cpl.py --feedback comparisons --render
```

### Parameters

- **Feedback Types**:
  - `comparisons`: Trains the model using pairwise comparison of trajectories.
  - `demonstrations`: Generates preference labels based on a reward comparison between trajectories.
  - `e_stop`: Model is trained to avoid "very bad" constraints using e-stop feedback.

- **Learning Rate** (`lr`): Adjust the optimizer learning rate.
- **Epochs** (`epochs`): Set the number of epochs for training.
- **Batch Size** (`batch_size`): Set the batch size for training.
- **Gamma** (`gamma`): Hyperparameter for discounting rewards in training.
- **Alpha** (`alpha`): Regularization parameter for the training.

### Output

The training process will output training metrics (e.g., loss, accuracy, precision) and visualizations of the model's progress during training.

- The results of the training process, including evaluation metrics (accuracy, reward, and precision), will be logged in `./logs`.
- Graphs will be saved as `CPL_<feedback_type>.png` showing the training progress for loss, accuracy, and precision.

## **Evaluation**

After training, you can evaluate the model's performance based on accuracy, precision, and reward metrics.

To visualize evaluation results, TensorBoard logs can be examined as follows:

```bash
tensorboard --logdir=./logs

tensorboard --logdir=./runs
```

Then, open your browser and go to `http://localhost:6006/` to visualize the results.

## **File Structure**

```
src/
├── cpl_corrected.py           # Main script for training the CPL model
├── traj_with_rewards.csv  # Input data file with trajectories, X, Y positions, and rewards
logs                   # TensorBoard logs directory
```

## **Model Architecture**

The model is based on a simple feedforward neural network with three fully connected layers:

1. **Input Layer**: Takes the state dimensions (x, y) as input.
2. **Hidden Layers**: Two hidden layers with ReLU activation and dropout for regularization.
3. **Output Layer**: Produces a preference score for each trajectory.

## **Additional Notes**

- **Reward Functions**: The reward functions are computed using a custom method that incorporates both positive and negative rewards. This is based on a policy learned from the rewards generated during training.
- **Data Preprocessing**: The data file `traj_with_rewards.csv` should contain the trajectories, rewards, and corresponding X and Y coordinates.
  
## **References**
