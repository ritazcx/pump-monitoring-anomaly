import pandas as pd


def build_condition_episodes(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    anomaly_flag_col: str = "anomaly_flag",
    anomaly_score_col: str = "anomaly_score",
    issue_pattern_col: str = "issue_pattern",
    gap_minutes: int = 5,
    latest_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Group anomaly points into condition episodes.

    Rules:
    - Only rows where anomaly_flag == True are considered.
    - If the gap between consecutive anomaly timestamps is <= gap_minutes,
      they belong to the same episode.
    - Otherwise, a new episode starts.

    Returns a dataframe with one row per episode.
    """

    required_cols = {
        timestamp_col,
        anomaly_flag_col,
        anomaly_score_col,
        issue_pattern_col,
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        return _empty_episode_df()

    working_df = df.copy()
    working_df[timestamp_col] = pd.to_datetime(working_df[timestamp_col])

    # Keep only anomaly rows
    anomaly_df = (
        working_df.loc[working_df[anomaly_flag_col] == True]
        .sort_values(timestamp_col)
        .reset_index(drop=True)
    )

    if anomaly_df.empty:
        return _empty_episode_df()

    # Determine reference latest timestamp
    if latest_timestamp is None:
        latest_timestamp = working_df[timestamp_col].max()
    else:
        latest_timestamp = pd.to_datetime(latest_timestamp)

    # Compute time gap between consecutive anomaly points
    anomaly_df["prev_timestamp"] = anomaly_df[timestamp_col].shift(1)
    anomaly_df["gap_min"] = (
        (anomaly_df[timestamp_col] - anomaly_df["prev_timestamp"])
        .dt.total_seconds()
        .div(60)
    )

    # Start a new episode if:
    # - first row
    # - or gap > threshold
    anomaly_df["new_episode"] = (
        anomaly_df["gap_min"].isna() | (anomaly_df["gap_min"] > gap_minutes)
    )

    anomaly_df["episode_seq"] = anomaly_df["new_episode"].cumsum()

    episode_rows = []

    for episode_seq, group in anomaly_df.groupby("episode_seq", sort=True):
        group = group.sort_values(timestamp_col).reset_index(drop=True)

        start_time = group[timestamp_col].min()
        end_time = group[timestamp_col].max()
        duration_min = max(
            1,
            int((end_time - start_time).total_seconds() / 60) + 1
        )

        point_count = len(group)
        max_score = float(group[anomaly_score_col].max())

        # Count issue patterns inside the episode
        pattern_counts = group[issue_pattern_col].fillna("unknown").value_counts()

        # Prefer known patterns over unknown if any exists
        known_patterns = pattern_counts.drop(labels=["unknown"], errors="ignore")
        if not known_patterns.empty:
            dominant_pattern = known_patterns.index[0]
        else:
            dominant_pattern = "unknown"

        # Active if the latest anomaly point is close enough to the overall latest timestamp
        minutes_since_end = (latest_timestamp - end_time).total_seconds() / 60
        status = "active" if minutes_since_end <= gap_minutes else "cleared"

        episode_rows.append(
            {
                "episode_id": f"EP-{int(episode_seq):03d}",
                "start_time": start_time,
                "end_time": end_time,
                "duration_min": duration_min,
                "point_count": point_count,
                "dominant_pattern": dominant_pattern,
                "max_anomaly_score": round(max_score, 6),
                "status": status,
            }
        )

    episodes_df = pd.DataFrame(episode_rows)

    if episodes_df.empty:
        return _empty_episode_df()

    return episodes_df.sort_values("start_time", ascending=False).reset_index(drop=True)


def _empty_episode_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "episode_id",
            "start_time",
            "end_time",
            "duration_min",
            "point_count",
            "dominant_pattern",
            "max_anomaly_score",
            "status",
        ]
    )