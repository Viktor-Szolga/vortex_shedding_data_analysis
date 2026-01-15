from src.utils import get_time_series, parse_openfoam_log_to_df
import matplotlib.pyplot as plt

get_time_series("data/velocity/laminar_reynolds_100.csv", "p9", "uy")

with open (r"data\all_logs\velocity_sweep_laminar_reynolds_100_log.pimpleFoam", "r") as f:
    log_text = f.read()

df = parse_openfoam_log_to_df(log_text)
df.reset_index(inplace=True)
print(df)

plt.plot(df["Time"][100:], df["Courant_Max"][100:])
plt.title("Max Corrant number")
plt.plot()
plt.ylabel("Max Co")
plt.xlabel("Time")
plt.yscale("log")
plt.show()

plt.plot(df["Time"][100:], df["Courant_Mean"][100:])
plt.title("Mean Corrant number")
plt.plot()
plt.ylabel("Mean Co")
plt.xlabel("Time")
plt.yscale("log")
plt.show()

plt.plot(df["Time"], df["Cont_Cumulative"])
plt.title("Cumulative Error")
plt.plot()
plt.ylabel("Cumulative Error")
plt.xlabel("Time")
plt.show()

plt.plot(df["Time"], df["p_InitialRes"], label="Initial residual")
plt.plot(df["Time"], df["p_FinalRes"], label="Final residual")
plt.title("Start vs End Residuals")
plt.plot()
plt.ylabel("Residual Error")
plt.xlabel("Time")
plt.yscale("log")
plt.legend()
plt.show()


plt.plot(df["Time"], df["Cont_Local"], label="Local continuity error")
plt.plot(df["Time"], df["Cont_Global"], label="Global continuity error")
plt.title("Global vs Local continuity error")
plt.plot()
plt.ylabel("Continuity Error")
plt.xlabel("Time")
plt.legend()
plt.show()