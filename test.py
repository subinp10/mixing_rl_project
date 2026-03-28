from stable_baselines3 import PPO
from env import MixingEnv

env = MixingEnv()
model = PPO.load("ppo_mixing_final")

obs = env.reset()

print("Target Ratio:", env.target)

for step in range(15):
    action, _ = model.predict(obs)

    obs, reward, done, _ = env.step(action)

    print(f"\nStep {step+1}")
    print("Action (mix):", action)
    print("New Fluid:", env.pool[-1])
    print("Error:", env.compute_error(env.pool[-1]))

    if done:
        print("\nFinished!")
        break

print("\n--- FINAL RESULTS ---")
print("Mixing Sequence:", env.history)
print("Final K (Loads):", env.K)
print("Final B (Bends):", env.B)
print("Final L (Path):", env.L)
