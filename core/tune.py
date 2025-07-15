def tune_from_prompt(prompt: str):
    print(f"[TUNE] Received prompt: {prompt}")
    # TODO: Same as train, but add Optuna/Ray for tuning