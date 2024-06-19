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


def merge_recorridos_temperaturas(recorridos_raw, temp_data):
    def group_data(data):
        return pd.DataFrame(
            data.groupby(["ANIO", "MES_NUM", "MES"], as_index=False).size()
        ).rename(columns={"size": "RECORRIDOS"})

    recorridos = group_data(recorridos_raw)

    def imputed_recorridos(data):
        months = [
            "Enero",
            "Febrero",
            "Marzo",
            "Abril",
            "Mayo",
            "Junio",
            "Julio",
            "Agosto",
            "Septiembre",
            "Octubre",
            "Noviembre",
            "Diciembre",
        ]
        month_nums = range(1, 13)
        years = range(np.min(data["ANIO"]), 2025)
        mean_recorridos = data["RECORRIDOS"].mean()

        expected_groups = [
            (y, m)
            for y in years
            for m in month_nums
            if y != 2024 or y == 2024 and m < 5
        ]

        missing_groups = []
        for year, month_num in expected_groups:
            in_data = data[data["ANIO"] == year][["MES_NUM"]].values
            if month_num not in in_data:
                missing_groups.append((year, month_num))

        # print("Missing groups:", missing_groups)

        def prev_group(year, month_num):
            if month_num == 1:
                return (year - 1, 12)
            return (year, month_num - 1)

        def next_group(year, month_num):
            if month_num == 12:
                return (year + 1, 1)
            return (year, month_num + 1)

        def find_prev_group(year, month_num):
            prev = prev_group(year, month_num)
            if prev in missing_groups:
                return find_prev_group(*prev)
            return prev

        def find_next_group(year, month_num):
            next = next_group(year, month_num)
            if next in missing_groups:
                return find_next_group(*next)
            return next

        def lookup_recorridos(year, month_num):
            recorridos = data[(data["ANIO"] == year) & (data["MES_NUM"] == month_num)][
                "RECORRIDOS"
            ].values

            if len(recorridos) == 0:
                return mean_recorridos
            return recorridos[0]

        def group_distance(group1, group2):
            return (group2[0] - group1[0]) * 12 + group2[1] - group1[1]

        def interpolate_recorridos(group, prev, next):
            prev_recorridos = lookup_recorridos(*prev)
            next_recorridos = lookup_recorridos(*next)
            distance = group_distance(prev, next)

            return (
                prev_recorridos
                + group_distance(prev, group)
                * (next_recorridos - prev_recorridos)
                / distance
            )

        missing_data = {"ANIO": [], "MES_NUM": [], "MES": [], "RECORRIDOS": []}
        for group in missing_groups:
            year, month_num = group
            prev = find_prev_group(*group)
            next = find_next_group(*group)

            missing_data["ANIO"].append(year)
            missing_data["MES_NUM"].append(month_num)
            missing_data["MES"].append(months[month_num - 1])

            imputed_recorridos = round(interpolate_recorridos(group, prev, next))
            missing_data["RECORRIDOS"].append(imputed_recorridos)

        return pd.DataFrame(missing_data)

    missing_df = imputed_recorridos(recorridos)

    imputed_data = pd.concat([recorridos, missing_df], ignore_index=True).sort_values(
        by=["ANIO", "MES_NUM"]
    )

    analysis_data = pd.DataFrame(
        imputed_data.merge(
            temp_data, left_on=["ANIO", "MES"], right_on=["Year", "Month"], how="inner"
        )[["ANIO", "MES_NUM", "MES", "RECORRIDOS", "TempMean"]]
    ).rename(columns={"TempMean": "TEMP"})

    return pd.DataFrame(analysis_data)
