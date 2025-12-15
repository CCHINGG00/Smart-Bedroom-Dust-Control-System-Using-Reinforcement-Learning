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
# 1. YOUR SPECIFIC ENVIRONMENT CLASS
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
        
        # State: [Indoor, Outdoor, Occ, Hum, Temp, Win]
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
            np.random.uniform(15.0, 50.0), # Indoor
            np.random.uniform(5.0, 80.0),  # Outdoor
            1.0,                           # Occupancy
            np.random.uniform(40.0, 60.0), # Humidity
            np.random.uniform(15.0, 30.0), # Temp
            0.0                            # Window
        ], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        Cin, Cout, occ, hum, temp, win = self.state
        hour = (self.current_step * self.dt_minutes) / 60.0
        Cout_next = self._update_outdoor_pm(hour)
        occ_next, win_next = self._update_occupancy_window(int(round(occ)), win, hour)
        
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

        # --- INFO DICTIONARY ---
        info = {
            "avg_indoor_pm2_5": Cin_next,
            "outdoor_pm2_5": Cout_next,
            "total_energy_Wh": self.total_energy_Wh,
            "action_name": ["Off", "Low", "Moderate", "High"][int(action)],
            "occupancy": int(round(occ_next)),
            "humidity": hum_next,
            "temperature": temp_next,
            "window_status": win_next
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
            change_prob = 0.08
            level_probs = np.array([0.25, 0.25, 0.25, 0.15, 0.10])
        else: 
            change_prob = 0.05
            level_probs = np.array([0.60, 0.25, 0.10, 0.04, 0.01])

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
# 2. STREAMLIT APP LOGIC
# ==========================================

st.set_page_config(page_title="Smart Bedroom Dust Control System Using Reinforcement Learning(PPO)", layout="wide")

st.title("🏡 Smart Bedroom Agent - Performance Dashboard")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model Path", "ppo_advanced_myenv/final_model.zip")
stats_path = st.sidebar.text_input("Stats Path", "ppo_advanced_myenv/vec_normalize.pkl")
sim_speed = st.sidebar.slider("Speed", 0.01, 0.5, 0.05)

# --- STOP BUTTON ---
if st.sidebar.button("⛔ Stop Simulation"):
    st.stop()

# --- LOAD AGENT ---
@st.cache_resource
def load_agent(model_path, stats_path):
    env = SmartBedroomDustEnv(dt_minutes=5.0, episode_hours=24.0)
    env = DummyVecEnv([lambda: env])
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    else:
        st.error(f"❌ Stats not found: {stats_path}")
        st.stop()
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    return model, env

# --- START BUTTON ---
if st.button("▶️ Start Simulation"):
    model, env = load_agent(model_path, stats_path)
    
    # 1. Status Message
    status_text = st.empty()
    status_text.info("Simulation Running... (Click 'Stop' in sidebar to cancel)")

    # --- LAYOUT: REVERTED TO TWO ROWS ---
    
    # Row 1: Primary Metrics
    st.subheader("🔴 Action & Air Quality")
    col1, col2, col3, col4 = st.columns(4)
    with col1: act_metric = st.empty()
    with col2: pm_metric = st.empty()
    with col3: out_metric = st.empty()
    with col4: energy_metric = st.empty()
    
    # Row 2: Secondary States (Temp, Hum, Occupancy, Window)
    st.subheader("🔵 Bedroom State")
    col5, col6, col7, col8 = st.columns(4)
    with col5: temp_metric = st.empty()
    with col6: hum_metric = st.empty()
    with col7: occ_metric = st.empty()
    with col8: win_metric = st.empty()

    # Live Charts
    st.subheader("📈 Live History")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1: pm_chart = st.empty()
    with chart_col2: energy_chart = st.empty()

    # --- SIMULATION LOOP ---
    obs = env.reset()
    done = False
    history_df = pd.DataFrame(columns=["Step", "Indoor PM", "Outdoor PM", "Energy", "Temp", "Hum"])
    step = 0
    progress_bar = st.progress(0)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        real_info = info[0]
        
        # Extract Values
        indoor = real_info["avg_indoor_pm2_5"]
        outdoor = real_info["outdoor_pm2_5"]
        energy = real_info["total_energy_Wh"]
        act_name = real_info["action_name"]
        temp = real_info["temperature"]
        hum = real_info["humidity"]
        occ = "Yes" if real_info["occupancy"] == 1 else "No"
        
        win_val = real_info["window_status"]
        if win_val == 0.0: win_str = "Closed"
        elif win_val == 1.0: win_str = "Open"
        else: win_str = f"{int(win_val*100)}%"

        # --- UPDATE UI ---
        
        # Row 1
        act_metric.metric("Action", act_name)
        pm_metric.metric("Indoor PM", f"{indoor:.1f}", delta=f"{15-indoor:.1f}")
        out_metric.metric("Outdoor PM", f"{outdoor:.1f}")
        energy_metric.metric("Energy", f"{energy:.0f} Wh")

        # Row 2
        temp_metric.metric("Temp", f"{temp:.1f}°C")
        hum_metric.metric("Humidity", f"{hum:.1f}%")
        occ_metric.metric("Occupied", occ)
        win_metric.metric("Window", win_str)

        # Charts
        new_row = pd.DataFrame({
            "Step": [step], "Indoor PM": [indoor], "Outdoor PM": [outdoor], 
            "Energy": [energy], "Temp": [temp], "Hum": [hum]
        })
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        
        with pm_chart.container():
            st.line_chart(history_df.set_index("Step")[["Indoor PM", "Outdoor PM"]])
        with energy_chart.container():
            st.area_chart(history_df.set_index("Step")["Energy"], color="#2ecc71")

        step += 1
        progress_bar.progress(min(step / 288, 1.0))
        time.sleep(sim_speed)

    # --- DONE ---
    status_text.success("Day Complete! Simulation Finished.")