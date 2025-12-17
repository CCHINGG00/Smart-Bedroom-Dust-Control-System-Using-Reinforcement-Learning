import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==========================================
# 1. ENVIRONMENT CLASS (Unchanged)
# ==========================================
class SmartBedroomDustEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, dt_minutes: float = 5.0, episode_hours: float = 24.0, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.dt_minutes = dt_minutes
        self.dt_hours = dt_minutes / 60.0
        self.steps_per_episode = int(episode_hours * 60.0 / dt_minutes)
        self.action_space = spaces.Discrete(4)
        
        low = np.array([5.0, 5.0, 0.0, 20.0, 15.0, 0.0], dtype=np.float32)
        high = np.array([140.0, 80.0, 1.0, 80.0, 30.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.room_area = 20.0
        self.room_height = 2.8
        self.room_volume = self.room_area * self.room_height 
        self.A_over_V = 0.303
        self.vd = 0.44
        self.k_nature = self.A_over_V * self.vd 
        self.cadr_values = np.array([0.0, 24.0, 40.0, 96.0], dtype=np.float32)
        self.power_values = np.array([0.0, 10.0, 16.0, 50.0], dtype=np.float32)
        self.outdoor_noise_std = 3.0
        self.source_prob_occ = 0.1
        self.source_prob_empty = 0.02
        self.pm_ref = 35.0
        self.w_pm = 5
        self.w_energy = 0.1
        self.state = None
        self.current_step = 0
        self.total_energy_Wh = 0.0
        self.cumulative_pm = 0.0
        self.window_levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        self.aer_levels = np.array([0.253, 1.497, 1.926, 2.161, 2.715], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_energy_Wh = 0.0
        self.cumulative_pm = 0.0
        self.state = np.array([
            np.random.uniform(15.0, 50.0), np.random.uniform(5.0, 80.0), 1.0, 
            np.random.uniform(40.0, 60.0), np.random.uniform(15.0, 30.0), 0.0
        ], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        Cin, Cout, occ, hum, temp, win = self.state
        total_hours_elapsed = (self.current_step * self.dt_minutes) / 60.0
        hour_of_day = total_hours_elapsed % 24.0
        
        Cout_next = self._update_outdoor_pm(hour_of_day)
        occ_next, win_next = self._update_occupancy_window(int(round(occ)), win, hour_of_day)
        
        idx = int(np.clip(np.round(win_next * 4), 0, 4))
        alpha = float(self.aer_levels[idx])
        P = 0.8 + 0.2 * win_next
        S_rate = self._sample_source_emission(occ_next, win_next)
        CADR = self.cadr_values[action]
        k_purifier = self.k_nature + CADR / self.room_volume
        
        dCin_dt = alpha * P * Cout_next - (alpha + k_purifier) * Cin + S_rate
        Cin_next = float(np.clip(Cin + self.dt_hours * dCin_dt, 5.0, 140.0))
        hum_next = float(np.clip(hum + np.random.normal(0.0, 0.3), 20.0, 80.0))
        temp_next = float(np.clip(temp + np.random.normal(0.0, 0.2), 15.0, 30.0))

        self.state = np.array([Cin_next, Cout_next, float(occ_next), hum_next, temp_next, float(win_next)], dtype=np.float32)
        reward = self._compute_reward(Cin_next, action)
        
        energy_Wh = self.power_values[action] * (self.dt_minutes / 60.0)
        self.total_energy_Wh += energy_Wh
        self.cumulative_pm += Cin_next
        self.current_step += 1
        truncated = self.current_step >= self.steps_per_episode

        info = {
            "avg_indoor_pm2_5": Cin_next, 
            "outdoor_pm2_5": Cout_next, 
            "total_energy_Wh": self.total_energy_Wh,
            "step_energy_Wh": energy_Wh, 
            "action_name": ["Off", "Low", "Moderate", "High"][int(action)], 
            "occupancy": int(round(occ_next)),
            "humidity": hum_next, 
            "temperature": temp_next, 
            "window_status": win_next,
            "day": int(total_hours_elapsed // 24) + 1, 
            "hour": hour_of_day
        }
        return self.state.copy(), float(reward), False, truncated, info

    def _update_outdoor_pm(self, hour):
        base = 55.0 + 15.0 * np.sin(2 * np.pi * (hour / 24.0))
        return float(np.clip(base + np.random.normal(0.0, self.outdoor_noise_std), 5.0, 80.0))
    def _update_occupancy_window(self, occ, win, hour):
        if 22 <= hour or hour < 7: base_occ = 0.9
        elif 7 <= hour < 9 or 18 <= hour < 22: base_occ = 0.5
        else: base_occ = 0.2
        if np.random.rand() < 0.1: occ_next = 1 if np.random.rand() < base_occ else 0
        else: occ_next = int(occ)
        if 9 <= hour < 18: 
            change_prob = 0.08; level_probs = np.array([0.25, 0.25, 0.25, 0.15, 0.10])
        else: 
            change_prob = 0.05; level_probs = np.array([0.60, 0.25, 0.10, 0.04, 0.01])
        if np.random.rand() < change_prob: win_next = float(np.random.choice(self.window_levels, p=level_probs))
        else: win_next = float(win)
        return occ_next, win_next
    def _sample_source_emission(self, occ, win):
        if occ == 1: p, mean, std = self.source_prob_occ, 15.0, 5.0
        else: p, mean, std = self.source_prob_empty, 8.0, 4.0
        return float(max(np.random.normal(mean, std), 0.0)) if np.random.rand() < p else 0.0
    def _compute_reward(self, Cin, action):
        normalized_pm = np.clip(Cin / self.pm_ref, 0, 3.0)
        return float(-(self.w_pm * np.exp(normalized_pm ** 2) + self.w_energy * self.power_values[action]))
    def render(self): pass
    def close(self): pass

# ==========================================
# 2. STREAMLIT APP
# ==========================================

st.set_page_config(page_title="Smart Bedroom Dust Control", layout="wide")

MODEL_PATH = "ppo_advanced_myenv/final_model.zip"
STATS_PATH = "ppo_advanced_myenv/vec_normalize.pkl"

st.title("🏡 Smart Bedroom Dust Control System Using Reinforcement Learning (PPO)")

# --- SESSION STATE INITIALIZATION ---
if 'simulation_running' not in st.session_state: st.session_state.simulation_running = False
if 'paused' not in st.session_state: st.session_state.paused = False
if 'last_metrics' not in st.session_state: st.session_state.last_metrics = {}
if 'num_days' not in st.session_state: st.session_state.num_days = 1

@st.cache_resource
def load_agent(episode_hours):
    env = SmartBedroomDustEnv(dt_minutes=5.0, episode_hours=episode_hours)
    env = DummyVecEnv([lambda: env])
    if os.path.exists(STATS_PATH):
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False; env.norm_reward = False
    else:
        st.error(f"❌ Stats not found: {STATS_PATH}"); st.stop()
    try: model = PPO.load(MODEL_PATH, env=env)
    except Exception as e: st.error(f"❌ Error loading model: {e}"); st.stop()
    return model, env

# --- SIDEBAR ---
st.sidebar.header("Configuration")
sim_speed = st.sidebar.slider("Simulation Speed", 0.01, 0.5, 0.05)

if st.session_state.simulation_running:
    min_day_val = st.session_state.num_days
else:
    min_day_val = 1

num_days = st.sidebar.slider(
    "Simulation Duration (Days)", 
    min_value=min_day_val, 
    max_value=30, 
    key='num_days', 
    help="You can extend the duration while paused, but you cannot shorten it."
)

total_hours = 24.0 * num_days

# --- CALLBACK FUNCTION FOR RESET ---
def reset_simulation_callback():
    st.session_state.simulation_running = False
    st.session_state.paused = False
    st.session_state.num_days = 1

# --- BUTTONS ---
if st.sidebar.button("▶️ Start Simulation"):
    st.session_state.simulation_running = True
    st.session_state.paused = False
    # === UPDATED DATAFRAME COLUMNS (Changed Window label) ===
    st.session_state.history_df = pd.DataFrame(columns=[
        "Step", "Indoor PM", "Outdoor PM", "Indoor Temperature", "Indoor Humidity", "Window Status", "Occupied", "Action", "Step Energy", "Total Energy"
    ])
    st.session_state.current_step = 0
    st.session_state.current_obs = None
    st.session_state.last_metrics = {}
    st.session_state.env_vars = {'state': None, 'total_energy': 0.0, 'cum_pm': 0.0, 'step': 0}
    st.rerun()

if st.session_state.simulation_running:
    if st.session_state.paused:
        if st.sidebar.button("▶️ Resume Simulation"):
            st.session_state.paused = False
            st.rerun()
    else:
        if st.sidebar.button("⏸ Pause Simulation"):
            st.session_state.paused = True
            st.rerun()

st.sidebar.button("⏹ Cancel Simulation", on_click=reset_simulation_callback)

# --- MAIN LOGIC ---
if st.session_state.simulation_running:
    model, env = load_agent(total_hours)
    real_env = env.venv.envs[0]
    
    if st.session_state.current_step == 0:
        obs = env.reset(); st.session_state.current_obs = obs; st.session_state.env_vars['state'] = real_env.state.copy()
    else:
        obs = env.reset()
        real_env.current_step = st.session_state.env_vars['step']
        real_env.total_energy_Wh = st.session_state.env_vars['total_energy']
        real_env.cumulative_pm = st.session_state.env_vars['cum_pm']
        real_env.state = st.session_state.env_vars['state']
        obs = st.session_state.current_obs

    status_text = st.empty()
    st.subheader("🔴 Action & Air Quality")
    col1, col2, col3, col4 = st.columns(4)
    act_metric = col1.empty(); pm_metric = col2.empty(); out_metric = col3.empty(); energy_metric = col4.empty()
    
    st.subheader("🔵 Bedroom State")
    col5, col6, col7, col8 = st.columns(4)
    temp_metric = col5.empty(); hum_metric = col6.empty(); occ_metric = col7.empty(); win_metric = col8.empty()

    # --- GRAPHS SECTION ---
    st.subheader("📈 Live History")
    
    # Top Row: PM2.5 and Total Accumulated Energy
    row1_col1, row1_col2 = st.columns(2)
    pm_chart = row1_col1.empty()
    total_energy_chart = row1_col2.empty()
    
    # Bottom Row: Energy per Step (Full Width)
    st.markdown("---") 
    step_energy_chart = st.empty()

    # --- NEW: DATA TABLE SECTION ---
    st.subheader("📋 Simulation Data Log")
    data_table_placeholder = st.empty()
    
    progress_bar = st.progress(0)
    total_steps_needed = 288 * num_days

    # Restore Metrics if Paused
    if st.session_state.last_metrics:
        m = st.session_state.last_metrics
        act_metric.metric("Action", m['act'])
        pm_metric.metric("Indoor PM", f"{m['indoor']:.1f}")
        out_metric.metric("Outdoor PM", f"{m['outdoor']:.1f}")
        energy_metric.metric("Total Energy", f"{m['energy']:.0f} Wh")
        temp_metric.metric("Indoor Temperature", f"{m['temp']:.1f}°C")
        hum_metric.metric("Indoor Humidity", f"{m['hum']:.1f}%")
        occ_metric.metric("Occupied", m['occ'])
        # === UPDATED LABEL HERE ===
        win_metric.metric("Window Status", m['win'])

    # Re-draw content from history if it exists
    if not st.session_state.history_df.empty:
        with pm_chart.container():
            st.caption("Indoor vs Outdoor PM2.5")
            st.line_chart(st.session_state.history_df.set_index("Step")[["Indoor PM", "Outdoor PM"]])
        with step_energy_chart.container():
            st.caption("Energy Used per Step (Instantaneous)")
            st.line_chart(st.session_state.history_df.set_index("Step")["Step Energy"], color="#FF5733")
        with total_energy_chart.container():
            st.caption("Total Accumulated Energy")
            st.area_chart(st.session_state.history_df.set_index("Step")["Total Energy"], color="#2ecc71")
        # Draw Table
        with data_table_placeholder.container():
            st.dataframe(st.session_state.history_df.sort_values(by="Step", ascending=True), height=300)

    # --- SIMULATION LOOP ---
    if st.session_state.paused:
        status_text.warning("⚠️ Simulation Paused. You can extend Duration, but not shorten it.")
        progress_bar.progress(min(st.session_state.current_step / total_steps_needed, 1.0))
    else:
        status_text.info(f"Simulation Running...")
        done = False
        obs = st.session_state.current_obs
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            real_info = info[0]
            
            indoor = real_info["avg_indoor_pm2_5"]; outdoor = real_info["outdoor_pm2_5"]
            total_energy = real_info["total_energy_Wh"]
            step_energy = real_info["step_energy_Wh"]
            act_name = real_info["action_name"]
            temp = real_info["temperature"]; hum = real_info["humidity"]
            occ = "Yes" if real_info["occupancy"] == 1 else "No"
            win_str = "Fully Open" if real_info["window_status"] == 1.0 else ("Closed" if real_info["window_status"] == 0.0 else f"{int(real_info['window_status']*100)}%")

            act_metric.metric("Action", act_name)
            pm_metric.metric("Indoor PM", f"{indoor:.1f}")
            out_metric.metric("Outdoor PM", f"{outdoor:.1f}")
            energy_metric.metric("Total Energy", f"{total_energy:.0f} Wh")
            temp_metric.metric("Indoor Temperature", f"{temp:.1f}°C")
            hum_metric.metric("Indoor Humidity", f"{hum:.1f}%")
            occ_metric.metric("Occupied", occ)
            
            # === UPDATED LABEL HERE ===
            win_metric.metric("Window Status", win_str)

            st.session_state.last_metrics = {
                'act': act_name, 'indoor': indoor, 'outdoor': outdoor, 'energy': total_energy,
                'temp': temp, 'hum': hum, 'occ': occ, 'win': win_str
            }

            # Add ALL metrics to the new row
            # === UPDATED DICT KEYS TO MATCH NEW DATAFRAME COLUMNS ===
            new_row = pd.DataFrame({
                "Step": [st.session_state.current_step], 
                "Indoor PM": [indoor], 
                "Outdoor PM": [outdoor], 
                "Indoor Temperature": [temp], 
                "Indoor Humidity": [hum],
                "Window Status": [win_str],  # Key changed to match column
                "Occupied": [occ],
                "Action": [act_name],
                "Step Energy": [step_energy],
                "Total Energy": [total_energy]
            })
            st.session_state.history_df = pd.concat([st.session_state.history_df, new_row], ignore_index=True)
            st.session_state.current_step += 1
            st.session_state.current_obs = obs
            
            st.session_state.env_vars['state'] = real_env.state.copy()
            st.session_state.env_vars['total_energy'] = real_env.total_energy_Wh
            st.session_state.env_vars['cum_pm'] = real_env.cumulative_pm
            st.session_state.env_vars['step'] = real_env.current_step

            # Update Charts
            with pm_chart.container():
                st.caption("Indoor vs Outdoor PM2.5")
                st.line_chart(st.session_state.history_df.set_index("Step")[["Indoor PM", "Outdoor PM"]])
            with step_energy_chart.container():
                st.caption("Energy Used per Step (Instantaneous)")
                st.line_chart(st.session_state.history_df.set_index("Step")["Step Energy"], color="#FF5733")
            with total_energy_chart.container():
                st.caption("Total Accumulated Energy")
                st.area_chart(st.session_state.history_df.set_index("Step")["Total Energy"], color="#2ecc71")
            
            # Update Table (Sorted so newest step is at top)
            with data_table_placeholder.container():
                st.dataframe(st.session_state.history_df.sort_values(by="Step", ascending=True), height=300)
                
            progress_bar.progress(min(st.session_state.current_step / total_steps_needed, 1.0))
            
            time.sleep(sim_speed)
        
        if done:
            status_text.success(f"Simulation Complete! ({num_days} Days Processed)")
            st.session_state.simulation_running = False
else:
    st.info("👈 Select settings and click 'Start Simulation' to begin.")