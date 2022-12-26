import matplotlib.pyplot as plt
import pandas as pd

from src.configs import PathVariables

start = 0
interval = 10
step = 10

scores_df = pd.read_csv(PathVariables.SCORES_FILE)
mean_scores = pd.DataFrame(columns=['score'])

actions_df = pd.read_csv(PathVariables.ACTION_FILE)
max_scores = pd.DataFrame(columns=['max_score'])
q_max_scores = pd.DataFrame(columns=['q_max'])

while interval <= len(scores_df):
    mean_scores.loc[len(mean_scores)] = (scores_df.loc[start:interval].mean()['scores'])
    max_scores.loc[len(max_scores)] = (scores_df.loc[start:interval].max()['scores'])
    start = interval
    interval = interval + step

q_max_df = pd.read_csv(PathVariables.Q_VALUE_FILE)

start = 0
interval = 1000
step = 1000
while interval <= len(q_max_df):
    q_max_scores.loc[len(q_max_scores)] = (q_max_df.loc[start:interval].mean()['actions'])
    start = interval
    interval = interval + step

mean_scores.plot()
plt.show()
max_scores.plot()
plt.show()
q_max_scores.plot()
plt.show()

print("len(mean_scores)", len(mean_scores))
print("len(max_scores)", len(max_scores))
print("len(q_max_scores)", len(q_max_scores))
