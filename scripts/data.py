import functools
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


def load_precipitation_data() -> pd.DataFrame:
    # Read data from file
    raw = pd.read_excel(
        "./data/precipitaciones.xlsx",
        sheet_name="MA_AX07",
        header=[1, 2],
        index_col=0,
        nrows=13,  # Include "Total" row
    )

    # Rename columns
    renames = {"Días": "Days"}
    df = raw.rename(columns=renames)

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


def merge_analysis(*dfs: pd.DataFrame) -> pd.DataFrame:
    def merge(df1, df2):
        return df1.join(df2, on=["Year", "Month"], how="inner")

    return functools.reduce(merge, dfs)


def analysis_temperature() -> pd.DataFrame:
    raw = load_temperature_data()

    def merge_levels(header):
        year, h1, h2 = header
        return (year, h1 + h2, "")

    raw.columns = raw.columns.map(merge_levels).droplevel(-1)

    temperature_min_abs = transposed_subcol(raw, "MinAbsolute", "TempMinAbs")
    temperature_max_abs = transposed_subcol(raw, "MaxAbsolute", "TempMaxAbs")
    temperature_min_mean = transposed_subcol(raw, "MinMean", "TempMinMean")
    temperature_max_mean = transposed_subcol(raw, "MaxMean", "TempMaxMean")
    temperature_mean = transposed_subcol(raw, "MeanMean", "TempMean")

    return merge_analysis(
        temperature_min_abs,
        temperature_max_abs,
        temperature_min_mean,
        temperature_max_mean,
        temperature_mean,
    )


def analysis_humidity() -> pd.DataFrame:
    raw = load_humidity_data()

    humidity_min = transposed_subcol(raw, "Min", "HumMin")
    humidity_max = transposed_subcol(raw, "Max", "HumMax")
    humidity_mean = transposed_subcol(raw, "Mean", "HumMean")

    return merge_analysis(humidity_min, humidity_max, humidity_mean)


def analysis_precipitation() -> pd.DataFrame:
    raw = load_precipitation_data().drop(index="Total")

    precip_days = transposed_subcol(raw, "Days", "PrecipDays")
    precip_mm = transposed_subcol(raw, "mm", "PrecipMm")

    return merge_analysis(precip_days, precip_mm)


def load_analysis_data() -> pd.DataFrame:
    temperature = analysis_temperature()
    humidity = analysis_humidity()
    precipitations = analysis_precipitation()
    return merge_analysis(temperature, humidity, precipitations)
