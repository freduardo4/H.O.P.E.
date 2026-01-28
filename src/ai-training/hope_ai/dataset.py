import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

def generate_synthetic_data(
    n_vehicles: int = 100,
    n_samples_per_vehicle: int = 3600,  # 1 hour of data per vehicle
    anomaly_rate: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic OBD2 data for training.

    In production, this would be replaced with real vehicle data.
    """
    logger.info(f"Generating synthetic data for {n_vehicles} vehicles...")

    all_data = []
    all_labels = []

    for vehicle_id in range(n_vehicles):
        # Base parameters for this vehicle (slight variations between vehicles)
        base_rpm = np.random.uniform(800, 1000)
        base_speed = 0
        base_load = np.random.uniform(15, 25)
        base_coolant = np.random.uniform(85, 95)

        vehicle_data = []
        vehicle_labels = []

        for t in range(n_samples_per_vehicle):
            # Simulate driving patterns
            driving_phase = (t % 600) / 600  # 10-minute cycles

            # Normal values with realistic variations
            if driving_phase < 0.2:  # Idle
                rpm = base_rpm + np.random.normal(0, 50)
                speed = 0
                load = base_load + np.random.normal(0, 5)
            elif driving_phase < 0.5:  # Acceleration
                rpm = base_rpm + 2000 * (driving_phase - 0.2) / 0.3 + np.random.normal(0, 100)
                speed = 100 * (driving_phase - 0.2) / 0.3 + np.random.normal(0, 5)
                load = 50 + 30 * (driving_phase - 0.2) / 0.3 + np.random.normal(0, 10)
            elif driving_phase < 0.8:  # Cruising
                rpm = 2500 + np.random.normal(0, 100)
                speed = 100 + np.random.normal(0, 5)
                load = 40 + np.random.normal(0, 5)
            else:  # Deceleration
                rpm = 2500 - 1500 * (driving_phase - 0.8) / 0.2 + np.random.normal(0, 100)
                speed = 100 - 100 * (driving_phase - 0.8) / 0.2 + np.random.normal(0, 5)
                load = 40 - 25 * (driving_phase - 0.8) / 0.2 + np.random.normal(0, 5)

            # Other parameters
            coolant_temp = base_coolant + min(10, t / 360) + np.random.normal(0, 2)
            intake_temp = 25 + speed * 0.1 + np.random.normal(0, 3)
            maf_flow = load * 0.5 + np.random.normal(0, 2)
            throttle = load * 0.8 + np.random.normal(0, 5)
            fuel_pressure = 350 + np.random.normal(0, 10)
            stft = np.random.normal(0, 3)
            ltft = np.random.normal(0, 2)

            # Simulated Ignition Timing (BTDC)
            # Higher load -> less advance
            # Higher RPM -> more advance
            ignition_timing = 15 - (load * 0.1) + (rpm * 0.002) + np.random.normal(0, 1)

            # Simple EGT Physical Model (Simplified)
            # Base temp + load factor + RPM factor + Ignition effect
            # Retarded ignition (lower advance) -> higher EGT
            egt = intake_temp + (load * 6) + (rpm * 0.05) - (ignition_timing * 5) + 300
            
            # Anomaly injection can affect EGT
            is_anomaly = False

            # Inject anomalies
            if np.random.random() < anomaly_rate:
                is_anomaly = True
                anomaly_type = np.random.choice([
                    'misfire', 'overheating', 'fuel_issue', 'sensor_fault', 'egt_spike'
                ])

                if anomaly_type == 'misfire':
                    rpm += np.random.uniform(-500, 500)
                    load += np.random.uniform(-20, 20)
                    egt -= np.random.uniform(50, 150) # Misfire usually drops EGT in that cylinder, but here it's aggregate
                elif anomaly_type == 'overheating':
                    coolant_temp += np.random.uniform(10, 30)
                    egt += np.random.uniform(20, 50)
                elif anomaly_type == 'fuel_issue':
                    stft += np.random.uniform(-15, 15)
                    ltft += np.random.uniform(-10, 10)
                    # Lean condition increases EGT
                    if stft < 0: egt += np.random.uniform(50, 100)
                elif anomaly_type == 'egt_spike':
                    egt += np.random.uniform(150, 300)
                elif anomaly_type == 'sensor_fault':
                    pass 

            # Clamp values to realistic ranges
            rpm = np.clip(rpm, 0, 8000)
            speed = np.clip(speed, 0, 250)
            load = np.clip(load, 0, 100)
            coolant_temp = np.clip(coolant_temp, -40, 150)
            intake_temp = np.clip(intake_temp, -40, 80)
            maf_flow = np.clip(maf_flow, 0, 200)
            throttle = np.clip(throttle, 0, 100)
            fuel_pressure = np.clip(fuel_pressure, 0, 800)
            stft = np.clip(stft, -25, 25)
            ltft = np.clip(ltft, -25, 25)
            ignition_timing = np.clip(ignition_timing, -10, 50)
            egt = np.clip(egt, 200, 1100)

            sample = [
                rpm, speed, load, coolant_temp, intake_temp,
                maf_flow, throttle, fuel_pressure, stft, ltft,
                ignition_timing, egt
            ]

            vehicle_data.append(sample)
            vehicle_labels.append(1 if is_anomaly else 0)

        all_data.append(np.array(vehicle_data))
        all_labels.append(np.array(vehicle_labels))

    # Stack all vehicle data
    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    logger.info(f"Generated {len(data)} samples, {labels.sum()} anomalies ({labels.mean()*100:.1f}%)")

    return data, labels
