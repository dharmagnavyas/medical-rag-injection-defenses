import pandas as pd, os
os.makedirs("data/raw", exist_ok=True)
url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
df = pd.read_csv(url)
df.to_csv("data/raw/advbench_harmful_behaviors.csv", index=False)
print(len(df), list(df.columns))
for i in range(0, len(df), 40):
    print(f"{i:3d}: {df.loc[i, 'goal']}")