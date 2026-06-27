import pandas as pd
import numpy as np
import datetime

class TelemetrySimulator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.assets = {
            "Asset-01": {"name": "CNC Milling Machine", "stress": "Soft", "base_rul": 150.0},
            "Asset-02": {"name": "Gas Turbine", "stress": "Medium", "base_rul": 300.0},
            "Asset-03": {"name": "Hydraulic Pump", "stress": "Hard", "base_rul": 500.0},
            "Asset-04": {"name": "Centrifugal Compressor", "stress": "Wet", "base_rul": 250.0}
        }
        # soft = high wear, medium = normal wear, hard = low wear, wet = variable/humidity wear
        self.stress_multipliers = {
            "Soft": 2.5,
            "Medium": 1.0,
            "Hard": 0.4,
            "Wet": 1.5
        }
        # Sensor baselines
        self.baselines = {
            "Vibration": {"mean": 2.5, "std": 0.3, "unit": "mm/s", "fail_thresh": 8.0},
            "Temperature": {"mean": 65.0, "std": 3.0, "unit": "°C", "fail_thresh": 95.0},
            "Pressure": {"mean": 45.0, "std": 2.5, "unit": "bar", "fail_thresh": 20.0}, # failure is drop in pressure
            "Speed": {"mean": 1500.0, "std": 50.0, "unit": "RPM", "fail_thresh": 1800.0},
            "OilFlow": {"mean": 12.0, "std": 0.8, "unit": "L/min", "fail_thresh": 5.0} # failure is drop in flow
        }
        
    def generate_history(self, hours=600, interval_mins=30):
        """Generates historical sensor telemetry with embedded degradations and failures."""
        records = []
        n_points = int(hours * (60 / interval_mins))
        start_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        
        for asset_id, info in self.assets.items():
            stress = info["stress"]
            base_rul = info["base_rul"]
            mult = self.stress_multipliers[stress]
            
            # We will simulate multiple cycles of degradation and repair
            time_accumulated = 0.0
            current_cycle_life = base_rul * np.random.uniform(0.8, 1.2)
            
            for i in range(n_points):
                ts = start_time + datetime.timedelta(minutes=i*interval_mins)
                dt_hours = interval_mins / 60.0
                time_accumulated += dt_hours * mult
                
                # RUL decreases
                current_rul = max(0.0, current_cycle_life - time_accumulated)
                
                # If RUL reaches 0, we have a failure, and then we "repair" (reset cycle)
                if current_rul <= 0:
                    # Maintenance / Repair event!
                    # Reset cycle
                    current_cycle_life = base_rul * np.random.uniform(0.8, 1.2)
                    time_accumulated = 0.0
                    current_rul = current_cycle_life
                    
                # Degradation factor (0 at start of cycle, 1 at failure)
                deg_factor = (current_cycle_life - current_rul) / current_cycle_life
                
                # Standard noise
                vib = np.random.normal(self.baselines["Vibration"]["mean"], self.baselines["Vibration"]["std"])
                temp = np.random.normal(self.baselines["Temperature"]["mean"], self.baselines["Temperature"]["std"])
                press = np.random.normal(self.baselines["Pressure"]["mean"], self.baselines["Pressure"]["std"])
                speed = np.random.normal(self.baselines["Speed"]["mean"], self.baselines["Speed"]["std"])
                flow = np.random.normal(self.baselines["OilFlow"]["mean"], self.baselines["OilFlow"]["std"])
                
                # Apply degradation patterns as failure approaches (deg_factor > 0.7)
                failure_prob = 0.0
                if deg_factor > 0.7:
                    # Sigmoid-like probability curve
                    failure_prob = 1.0 / (1.0 + np.exp(-12 * (deg_factor - 0.85)))
                    
                    # Choose a primary failure type for this asset
                    if asset_id == "Asset-01": # Bearing wear
                        vib += 4.5 * ((deg_factor - 0.7) / 0.3) ** 2
                        temp += 15.0 * ((deg_factor - 0.7) / 0.3)
                        flow -= 2.0 * ((deg_factor - 0.7) / 0.3)
                    elif asset_id == "Asset-02": # Overheating
                        temp += 25.0 * ((deg_factor - 0.7) / 0.3) ** 2
                        press += 8.0 * ((deg_factor - 0.7) / 0.3)
                        speed += 100.0 * ((deg_factor - 0.7) / 0.3)
                    elif asset_id == "Asset-03": # Hydraulic Leak
                        press -= 20.0 * ((deg_factor - 0.7) / 0.3) ** 2
                        flow -= 5.0 * ((deg_factor - 0.7) / 0.3)
                        vib += 1.5 * ((deg_factor - 0.7) / 0.3)
                    else: # Compressor Rotor Instability
                        vib += 3.5 * ((deg_factor - 0.7) / 0.3) ** 2
                        speed += 200.0 * ((deg_factor - 0.7) / 0.3)
                        temp += 10.0 * ((deg_factor - 0.7) / 0.3)
                
                # Check for actual failure threshold
                is_failed = 0
                if (vib > self.baselines["Vibration"]["fail_thresh"] or 
                    temp > self.baselines["Temperature"]["fail_thresh"] or 
                    press < self.baselines["Pressure"]["fail_thresh"] or 
                    flow < self.baselines["OilFlow"]["fail_thresh"]):
                    is_failed = 1
                    failure_prob = 1.0
                    
                records.append({
                    "Timestamp": ts,
                    "AssetID": asset_id,
                    "Vibration": max(0.1, vib),
                    "Temperature": max(10.0, temp),
                    "Pressure": max(0.1, press),
                    "Speed": max(10.0, speed),
                    "OilFlow": max(0.0, flow),
                    "StressProfile": stress,
                    "RUL": current_rul / mult, # Actual remaining useful time in hours
                    "FailureState": is_failed,
                    "FailureProb_GT": failure_prob # Ground truth failure probability
                })
                
        df = pd.DataFrame(records)
        df.sort_values(by=["AssetID", "Timestamp"], inplace=True)
        return df

    def get_realtime_stream(self, last_state_dict, elapsed_hours=0.5, override_stress=None):
        """Simulates real-time update given the last state of assets."""
        new_states = {}
        for asset_id, last_state in last_state_dict.items():
            info = self.assets[asset_id]
            stress = override_stress if override_stress else last_state.get("StressProfile", info["stress"])
            mult = self.stress_multipliers[stress]
            base_rul = info["base_rul"]
            
            # Calculate new RUL
            current_rul = last_state["RUL"] * last_state.get("multiplier", 1.0) # retrieve raw RUL
            current_rul -= elapsed_hours * mult
            
            # Check if repaired
            repaired = False
            if current_rul <= 0 or last_state.get("Action") == "BOX":
                # Simulated maintenance
                current_rul = base_rul * np.random.uniform(0.9, 1.1)
                repaired = True
                
            # Total life (for degradation calculation)
            total_life = base_rul
            deg_factor = max(0.0, min(1.0, (total_life - current_rul) / total_life))
            
            # Base sensors
            vib = np.random.normal(self.baselines["Vibration"]["mean"], self.baselines["Vibration"]["std"])
            temp = np.random.normal(self.baselines["Temperature"]["mean"], self.baselines["Temperature"]["std"])
            press = np.random.normal(self.baselines["Pressure"]["mean"], self.baselines["Pressure"]["std"])
            speed = np.random.normal(self.baselines["Speed"]["mean"], self.baselines["Speed"]["std"])
            flow = np.random.normal(self.baselines["OilFlow"]["mean"], self.baselines["OilFlow"]["std"])
            
            # Apply degradation
            if deg_factor > 0.7:
                if asset_id == "Asset-01":
                    vib += 4.5 * ((deg_factor - 0.7) / 0.3) ** 2
                    temp += 15.0 * ((deg_factor - 0.7) / 0.3)
                    flow -= 2.0 * ((deg_factor - 0.7) / 0.3)
                elif asset_id == "Asset-02":
                    temp += 25.0 * ((deg_factor - 0.7) / 0.3) ** 2
                    press += 8.0 * ((deg_factor - 0.7) / 0.3)
                    speed += 100.0 * ((deg_factor - 0.7) / 0.3)
                elif asset_id == "Asset-03":
                    press -= 20.0 * ((deg_factor - 0.7) / 0.3) ** 2
                    flow -= 5.0 * ((deg_factor - 0.7) / 0.3)
                    vib += 1.5 * ((deg_factor - 0.7) / 0.3)
                else:
                    vib += 3.5 * ((deg_factor - 0.7) / 0.3) ** 2
                    speed += 200.0 * ((deg_factor - 0.7) / 0.3)
                    temp += 10.0 * ((deg_factor - 0.7) / 0.3)
            
            # If overridden or manual anomaly injected (for demo purposes)
            if last_state.get("InjectAnomaly") and not repaired:
                vib += 5.0
                temp += 20.0
                press -= 15.0
                flow -= 4.0
                
            is_failed = 0
            if (vib > self.baselines["Vibration"]["fail_thresh"] or 
                temp > self.baselines["Temperature"]["fail_thresh"] or 
                press < self.baselines["Pressure"]["fail_thresh"] or 
                flow < self.baselines["OilFlow"]["fail_thresh"]):
                is_failed = 1
                
            new_states[asset_id] = {
                "Timestamp": datetime.datetime.now(),
                "AssetID": asset_id,
                "Vibration": max(0.1, vib),
                "Temperature": max(10.0, temp),
                "Pressure": max(0.1, press),
                "Speed": max(10.0, speed),
                "OilFlow": max(0.0, flow),
                "StressProfile": stress,
                "RUL": max(0.0, current_rul / mult),
                "FailureState": is_failed,
                "multiplier": mult,
                "repaired": repaired
            }
        return new_states

if __name__ == "__main__":
    sim = TelemetrySimulator()
    df = sim.generate_history(hours=24)
    print(df.head())
    print("History generation test passed!")
