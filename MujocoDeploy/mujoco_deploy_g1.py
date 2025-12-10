import sys
import time
import collections
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer
from legged_gym import LEGGED_GYM_ROOT_DIR
import onnxruntime as ort

from joystick import JoystickController

controller = JoystickController(max_velocity=1.0)

TRACK_CAMERA = True
MOVE_ARM_JOINTS = False
# ARM SMOOTH RANDOM STATE
ARM_UPDATE_STEPS = 1000   # jak často se generuje nové random gesto
SMOOTH_FACTOR = 0.0025      # jak rychle se blíží k cíli (0.01–0.05 je hezké)
VELOCITY_OFFSET = np.array([0.1, 0.0, 0.0])

def load_onnx_policy(path):
    model = ort.InferenceSession(path)
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")
    return run_inference

def load_config(config_path):
    """Load and process the YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process paths with LEGGED_GYM_ROOT_DIR
    for path_key in ['policy_path', 'xml_path']:
        config[path_key] = config[path_key].format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    
    # Convert lists to numpy arrays where needed
    array_keys = ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init']
    for key in array_keys:
        config[key] = np.array(config[key], dtype=np.float32)
    
    return config

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    
    q_conj = np.array([w, -x, -y, -z])
    
    return np.array([
        v[0] * (q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
        
        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
        
        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])

def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)

def compute_observation(d, config, action, cmd, height_cmd, n_joints):
    """Compute the observation vector from current state"""
    # Get state from MuJoCo
    qj = d.qpos[7:7+n_joints].copy()
    dqj = d.qvel[6:6+n_joints].copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()
    
    # Handle default angles padding
    if len(config['default_angles']) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[:len(config['default_angles'])] = config['default_angles']
    else:
        padded_defaults = config['default_angles'][:n_joints]
    
    # Scale the values
    qj_scaled = (qj - padded_defaults) * config['dof_pos_scale']
    dqj_scaled = dqj * config['dof_vel_scale']
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config['ang_vel_scale']
    
    # Calculate single observation dimension
    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + 12
    
    # Create single observation
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd[:3] * config['cmd_scale']
    single_obs[3:4] = np.array([height_cmd])
    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation
    single_obs[10:10+n_joints] = qj_scaled
    single_obs[10+n_joints:10+2*n_joints] = dqj_scaled
    single_obs[10+2*n_joints:10+2*n_joints+12] = action
    
    return single_obs, single_obs_dim

def main():
    # Load configuration
    config = load_config("g1.yaml")
    
    # Load robot model
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    
    # Check number of joints
    n_joints = d.qpos.shape[0] - 7
   # print(f"Robot has {n_joints} joints in MuJoCo model")
    
    # Initialize variables
    action = np.zeros(config['num_actions'], dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    cmd = config['cmd_init'].copy()
    height_cmd = config['height_cmd']
    
    # Initialize observation history
    single_obs, single_obs_dim = compute_observation(d, config, action, cmd, height_cmd, n_joints)
    obs_history = collections.deque(maxlen=config['obs_history_len'])
    for _ in range(config['obs_history_len']):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    
    # Prepare full observation vector
    obs = np.zeros(config['num_obs'], dtype=np.float32)
    
    # Load policy
    #policy = torch.jit.load(config['policy_path'])
    policy = load_onnx_policy(config['policy_path'])
    
    counter = 0

    robot_body_id = m.body('pelvis').id
    fallen_start_time = None
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20  

        current_arm_target = None

        while viewer.is_running():
            # --- FALL DETECTION ---
            pelvis_height = d.xpos[robot_body_id][2]  # Z = výška nad zemí
            FALL_THRESHOLD = 0.2                      # když je pelvis níž, robot spadl

            if pelvis_height < FALL_THRESHOLD:
                if fallen_start_time is None:
                    fallen_start_time = time.time()
                else:
                    if time.time() - fallen_start_time > 1.0:
                        print("Robot fallen! Resetting simulation...")
                        
                        mujoco.mj_resetData(m, d)
                        mujoco.mj_forward(m, d)

                        print("Simulation reset complete.")
                        
                        fallen_start_time = None
                        #start = time.time()   # reset časovače simulace
                        continue
            else:
                # robot stojí → reset odpočítávání
                fallen_start_time = None
            # ------------------------

            # Control leg joints with policy
            leg_tau = pd_control(
                target_dof_pos,
                d.qpos[7:7+config['num_actions']],
                config['kps'],
                np.zeros_like(config['kps']),
                d.qvel[6:6+config['num_actions']],
                config['kds']
            )
            
            d.ctrl[:config['num_actions']] = leg_tau
            
            # Keep other joints at zero positions if they exist
            if n_joints > config['num_actions']:
                arm_kp = 100.0
                arm_kd = 0.5

                hand_dofs = n_joints - config['num_actions']

                # Inicializace targetů při startu
                if current_arm_target is None:
                    current_arm_target = np.zeros(hand_dofs, dtype=np.float32)
                    next_arm_target = np.zeros(hand_dofs, dtype=np.float32)

                # GENERATE NEW RANDOM TARGET EVERY X STEPS
                if MOVE_ARM_JOINTS and counter % ARM_UPDATE_STEPS == 0:
                    noise_scale = 0.7
                    next_arm_target = np.random.uniform(-noise_scale, noise_scale, hand_dofs)

                # SMOOTH INTERPOLATION (LERP)
                current_arm_target = current_arm_target + SMOOTH_FACTOR * (next_arm_target - current_arm_target)

                arm_tau = pd_control(
                    current_arm_target,
                    d.qpos[7+config['num_actions']:7+n_joints],
                    np.ones(n_joints-config['num_actions']) * arm_kp,
                    np.zeros(n_joints-config['num_actions']),
                    d.qvel[6+config['num_actions']:6+n_joints],
                    np.ones(n_joints-config['num_actions']) * arm_kd
                )
                
                if d.ctrl.shape[0] > config['num_actions']:
                    d.ctrl[config['num_actions']:] = arm_tau
            
            # Step physics
            mujoco.mj_step(m, d)
            
            counter += 1
            if counter % config['control_decimation'] == 0:

                vel, h = controller.get_command()
                cmd = vel + VELOCITY_OFFSET
                height_cmd = h

                print("Velocity cmd:", cmd, "Height cmd:", height_cmd)
                # Update observation
                single_obs, _ = compute_observation(d, config, action, cmd, height_cmd, n_joints)
                obs_history.append(single_obs)
                
                # Construct full observation with history
                for i, hist_obs in enumerate(obs_history):
                    start_idx = i * single_obs_dim
                    end_idx = start_idx + single_obs_dim
                    obs[start_idx:end_idx] = hist_obs
                
                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().cpu().numpy().squeeze()
                
                # Transform action to target_dof_pos
                target_dof_pos = action * config['action_scale'] + config['default_angles']
            
                # BEFORE viewer.sync()
                # get robot position
                if TRACK_CAMERA:
                    robot_pos = d.xpos[robot_body_id].copy()

                    # follow robot
                    viewer.cam.lookat[:] = robot_pos
                    #viewer.cam.distance = 3.5     # jak daleko má kamera být
                    #viewer.cam.azimuth = 180      # natočení vlevo/vpravo
                    #viewer.cam.elevation = -20    # úhel shora


                # Sync viewer
                viewer.sync()
            
            # Time keeping
            expected_time = start + (counter * config['simulation_dt'])
            current_time = time.time()
            if current_time < expected_time:
                time.sleep(expected_time - current_time)

if __name__ == "__main__":
    main()