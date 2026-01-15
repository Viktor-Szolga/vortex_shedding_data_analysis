from src.utils import get_time_series, parse_openfoam_log_to_df


get_time_series("data/velocity/laminar_reynolds_100.csv", "p9", "uy")

with open (r"data\all_logs\velocity_sweep_laminar_reynolds_100_log.pimpleFoam", "r") as f:
    log_text = f.read()

df = parse_openfoam_log_to_df(log_text)
print(df)