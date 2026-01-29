import argparse
import sys
import os
import json
import torch
from hope_ai import (
    train_anomaly_detector,
    train_pinn,
    train_rul,
    XAIExplainer,
    AnomalyDetector
)
from hope_ai.tuning import GeneticOptimizer

def main():
    parser = argparse.ArgumentParser(description="H.O.P.E. AI Unified CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train Anomaly Detector
    train_anomaly_parser = subparsers.add_parser("train-anomaly", help="Train LSTM Anomaly Detector")
    train_anomaly_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_anomaly_parser.add_argument("--output", type=str, default="models", help="Output directory")

    # Train PINN
    train_pinn_parser = subparsers.add_parser("train-pinn", help="Train PINN Virtual Sensor")
    train_pinn_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_pinn_parser.add_argument("--output", type=str, default="models", help="Output directory")

    # Train RUL
    train_rul_parser = subparsers.add_parser("train-rul", help="Train RUL Forecaster")
    train_rul_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_rul_parser.add_argument("--output", type=str, default="models", help="Output directory")

    # Explain
    explain_parser = subparsers.add_parser("explain", help="Explain an anomaly")
    explain_parser.add_argument("--input", type=str, required=True, help="Path to input JSON data")
    explain_parser.add_argument("--model", type=str, help="Path to model file (directory for LSTM)")
    explain_parser.add_argument("--method", type=str, default="SHAP", choices=["SHAP", "LIME"], help="Explanation method")

    # Optimize
    optimize_parser = subparsers.add_parser("optimize", help="Optimize ECU calibration")
    optimize_parser.add_argument("--target-afr", type=float, default=14.7, help="Target AFR")
    optimize_parser.add_argument("--output", type=str, default="optimized_map.json", help="Output JSON file")

    args = parser.parse_args()

    if args.command == "train-anomaly":
        print("Starting Anomaly Detector training...")
        train_anomaly_detector(epochs=args.epochs, save_dir=args.output)
    
    elif args.command == "train-pinn":
        print("Starting PINN training...")
        train_pinn(epochs=args.epochs, save_path=args.output)
        
    elif args.command == "train-rul":
        print("Starting RUL Forecaster training...")
        train_rul(epochs=args.epochs, save_path=args.output)

    elif args.command == "explain":
        print(f"Explaining anomaly for {args.input} using {args.method}...")
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        # Load detector
        detector = AnomalyDetector(model_path=args.model)
        explainer = XAIExplainer(detector)
        report = explainer.generate_narrative(data, method=args.method)
        print(report)

    elif args.command == "optimize":
        print(f"Optimizing for target AFR: {args.target_afr}...")
        optimizer = GeneticOptimizer(target_afr=args.target_afr)
        optimized_map = optimizer.optimize()
        optimizer.save_map(optimized_map, args.output)
        print(f"Optimized map saved to {args.output}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
