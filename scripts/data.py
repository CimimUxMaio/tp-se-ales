import pandas as pd
import numpy as np


def load_temperature_data() -> pd.DataFrame:
    # Read data from file
    raw = pd.read_excel(
        "./data/temperatura.xlsx",
        sheet_name="MA_AX01",
        header=[1, 2, 3],
        index_col=0,
        nrows=12,
    )

    # Rename columns
    renames = {
        "Media": "Mean",
        "Máxima\nmedia": "Max",
        "Mínima\nmedia": "Min",
        "Absoluta": "Absolute",
        **{
            # Renamed "Unnamed columns to Mean"
            name: "Mean"
            for name in raw.columns.levels[2]
            if name.startswith("Unnamed")
        },
    }
    df = raw.rename(columns=renames)

    # Add missing "Absolute" column to first years
    collected_years = df.columns.get_level_values(0).unique()
    for year in collected_years:
        if "Absolute" not in df[year]["Min"].columns:
            df[year, "Min", "Absolute"] = pd.Series(dtype=np.float64)

        if "Absolute" not in df[year]["Max"].columns:
            df[year, "Max", "Absolute"] = pd.Series(dtype=np.float64)

    # Sort together with new columns
    df = df.reindex(sorted(df.columns), axis=1)

    # Replace "..." values with NaN
    df.replace({"…": np.nan}, inplace=True)
    return df


def load_humidity_data() -> pd.DataFrame:
    # Read data from file
    raw = pd.read_excel(
        "./data/humedad.xlsx",
        sheet_name="MA_AX04",
        header=[1, 2],
        index_col=0,
        nrows=12,
    )

    # Rename columns
    renames = {
        "Media": "Mean",
        "Máxima": "Max",
        "Mínima": "Min",
    }
    df = raw.rename(columns=renames)

    # Add missing "Mean" column to first years
    collected_years = df.columns.get_level_values(0).unique()
    for year in collected_years:
        if "Mean" not in df[year].columns:
            df[year, "Mean"] = pd.Series(dtype=np.float64)

    # Sort together with new columns
    df = df.reindex(sorted(df.columns), axis=1)

    # Replace "..." values with NaN
    df.replace({"…": np.nan}, inplace=True)
    return df


def transposed_subcol(df: pd.DataFrame, subcol: str, value_name: str) -> pd.DataFrame:
    target = df.xs(subcol, level=1, axis=1)
    target.columns.rename("Year", inplace=True)

    transposed = target.transpose()

    melted = transposed.melt(
        var_name="Month", value_name=value_name, ignore_index=False
    )

    return melted.reset_index().set_index(["Year", "Month"])


def analysis_temperature() -> pd.DataFrame:
    raw_temperature = load_temperature_data()
    raw_temperature.columns = raw_temperature.columns.droplevel(-1)
    return transposed_subcol(raw_temperature, "Mean", "Temp")


def analysis_humidity() -> pd.DataFrame:
    raw_humidity = load_humidity_data()

    humidity_min = transposed_subcol(raw_humidity, "Min", "HumMin")
    humidity_max = transposed_subcol(raw_humidity, "Max", "HumMax")
    humidity_mean = transposed_subcol(raw_humidity, "Mean", "HumMean")

    return humidity_min.join(humidity_max, on=["Year", "Month"], how="inner").join(
        humidity_mean, on=["Year", "Month"], how="inner"
    )


def load_analysis_data() -> pd.DataFrame:
    temperature = analysis_temperature()
    humidity = analysis_humidity()
    return temperature.join(humidity, on=["Year", "Month"], how="inner")
