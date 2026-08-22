import pickle

import folium
import matplotlib.pyplot as plt
import pandas as pd
from folium import TileLayer, Icon, Rectangle
import seaborn as sns


DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def clean_code_lines(lines):
    """
    Removes comments, blank lines, and docstring content from a list
    containing the source code of a function.

    :param lines: List of strings containing the function source code.
    :return: List of strings excluding comments, blank lines, and docstrings.
    """
    result = []
    in_docstring = False

    for line in lines:
        stripped_line = line.strip()

        # Detect the beginning or end of a docstring
        if stripped_line.startswith('"""'):
            if not in_docstring:
                in_docstring = True
            else:
                in_docstring = False
            continue

        if in_docstring:
            continue

        if stripped_line.startswith("#") or not stripped_line:
            continue

        result.append(line)

    return result


def visualize_trips(
    outliers_pd: pd.DataFrame,
    inliers_pd: pd.DataFrame,
    boundaries: dict
) -> folium.Map:
    """
    Visualizes pickup locations for two trip samples: outliers and inliers.

    NYC geographic boundaries are displayed as a rectangle, while outlier
    pickup locations are shown with red markers and inlier pickup locations
    with standard markers.

    :param outliers_pd: Pandas DataFrame containing outlier trips with
                        'pickup_latitude' and 'pickup_longitude'.
    :param inliers_pd: Pandas DataFrame containing inlier trips with
                       'pickup_latitude' and 'pickup_longitude'.
    :param boundaries: Dictionary containing NYC geographic boundaries with
                       'min_longitude', 'max_longitude',
                       'min_latitude', and 'max_latitude'.
    :return: Folium map containing trip locations and NYC boundaries.
    """

    map_osm = folium.Map(
        location=[40.734695, -73.990372],
        tiles="Stamen Toner",
        attr="bla",
        zoom_start=12
    )

    boundary_points = [
        (boundaries["min_latitude"], boundaries["min_longitude"]),
        (boundaries["min_latitude"], boundaries["max_longitude"]),
        (boundaries["max_latitude"], boundaries["max_longitude"]),
        (boundaries["max_latitude"], boundaries["min_longitude"])
    ]

    folium.PolyLine(
        locations=[boundary_points[0], boundary_points[1]],
        color="yellow",
        weight=4,
        opacity=0.8
    ).add_to(map_osm)

    folium.PolyLine(
        locations=[boundary_points[1], boundary_points[2]],
        color="yellow",
        weight=4,
        opacity=0.8
    ).add_to(map_osm)

    folium.PolyLine(
        locations=[boundary_points[2], boundary_points[3]],
        color="yellow",
        weight=4,
        opacity=0.8
    ).add_to(map_osm)

    folium.PolyLine(
        locations=[boundary_points[3], boundary_points[0]],
        color="yellow",
        weight=4,
        opacity=0.8
    ).add_to(map_osm)

    TileLayer(
        "CartoDB positron",
        attr="© OpenStreetMap contributors"
    ).add_to(map_osm)

    # Plot sampled outlier pickup locations
    for _, trip in outliers_pd.iterrows():
        if int(trip["pickup_latitude"]) != 0:
            folium.Marker(
                [trip["pickup_latitude"], trip["pickup_longitude"]],
                icon=Icon(color="red", icon="map-marker")
            ).add_to(map_osm)

    # Plot sampled inlier pickup locations
    for _, trip in inliers_pd.iterrows():
        if int(trip["pickup_latitude"]) != 0:
            folium.Marker(
                [trip["pickup_latitude"], trip["pickup_longitude"]]
            ).add_to(map_osm)

    return map_osm


def add_district_layers(input_map: folium.Map, gdf):
    """
    Adds NYC Community District polygons to an existing Folium map.

    Polygon colors represent the poverty rate, while district attributes
    are displayed through interactive tooltips.

    :param input_map: Existing Folium map.
    :param gdf: GeoPandas GeoDataFrame containing district geometries and
                a 'poverty_rate' column.
    """

    import json

    geojson_dict = json.loads(gdf.to_json())
    features = geojson_dict.get("features", [])
    first_props = features[0].get("properties", {})
    tooltip_fields = list(first_props.keys())

    vmin = float(gdf["poverty_rate"].min())
    vmax = float(gdf["poverty_rate"].max())

    colormap = folium.LinearColormap(
        colors=["green", "yellow", "red"],
        vmin=vmin,
        vmax=vmax,
        caption="Poverty Rate"
    )

    def style_function(feature):
        value = feature.get("properties", {}).get("poverty_rate")

        try:
            color = colormap(float(value))
        except Exception:
            color = "#cccccc"

        return {
            "fillColor": color,
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7
        }

    folium.GeoJson(
        data=geojson_dict,
        name="NYC Community Districts",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            labels=True
        )
    ).add_to(input_map)

    folium.LayerControl().add_to(input_map)


def visualize_community_districts(
    districts_gdf,
    trips_pd: pd.DataFrame
) -> folium.Map:
    """
    Visualizes NYC Community District boundaries together with sampled
    taxi pickup locations.

    :param districts_gdf: GeoPandas GeoDataFrame containing NYC Community
                          District geometries and attributes.
    :param trips_pd: Pandas DataFrame containing sampled taxi trips with
                     pickup latitude and longitude.
    :return: Interactive Folium map.
    """

    import json

    geojson_dict = json.loads(districts_gdf.to_json())
    features = geojson_dict.get("features", [])
    first_props = features[0].get("properties", {})
    tooltip_fields = list(first_props.keys())

    map_osm = folium.Map(
        location=[40.734695, -73.990372],
        tiles="Stamen Toner",
        attr="bla",
        zoom_start=12
    )

    TileLayer(
        "CartoDB positron",
        attr="© OpenStreetMap contributors"
    ).add_to(map_osm)

    for _, trip in trips_pd.iterrows():
        if int(trip["pickup_latitude"]) != 0:
            folium.Marker(
                [trip["pickup_latitude"], trip["pickup_longitude"]]
            ).add_to(map_osm)

    folium.GeoJson(
        data=geojson_dict,
        name="NYC Community Districts",
        style_function=lambda feature: {
            "fillColor": "gray",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.4
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            labels=True
        )
    ).add_to(map_osm)

    folium.LayerControl().add_to(map_osm)

    return map_osm


def plot_boxplot_facets(stats_df, variable: str) -> None:
    """
    Generates a faceted boxplot from the pivoted DataFrame returned by
    compute_boxplot_stats().

    Expected input:
        - One row per pickup hour (0-23).
        - Five statistics for each day of the week:
          q1, median, q3, whisker_low, and whisker_high.

    Layout:
        - Rows: hours of the day.
        - Columns: days of the week.

    Each subplot contains one boxplot generated from the precomputed
    statistics.
    """

    # Convert Spark DataFrame to Pandas when necessary
    if hasattr(stats_df, "toPandas"):
        pdf: pd.DataFrame = stats_df.toPandas()
    else:
        pdf = stats_df.copy()

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    hours = sorted(pdf["pickup_hour"].unique())

    n_rows = len(hours)
    n_cols = len(days)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(n_cols * 1.6, n_rows * 1.2),
        sharex=False,
        sharey=True,
        squeeze=False
    )

    for row_idx, hour in enumerate(hours):

        hour_row = pdf[pdf["pickup_hour"] == hour]

        for col_idx, day in enumerate(days):

            ax = axes[row_idx][col_idx]

            if hour_row.empty:
                ax.set_visible(False)
                continue

            row = hour_row.iloc[0]

            q1_val = row.get(f"{day}_q1", None)
            median_val = row.get(f"{day}_median", None)
            q3_val = row.get(f"{day}_q3", None)
            whisker_low = row.get(f"{day}_whisker_low", None)
            whisker_high = row.get(f"{day}_whisker_high", None)

            if any(
                value is None or
                (isinstance(value, float) and pd.isna(value))
                for value in [q1_val, median_val, q3_val]
            ):
                ax.set_visible(False)
                continue

            boxplot_stats = [{
                "med": float(median_val),
                "q1": float(q1_val),
                "q3": float(q3_val),
                "whislo": (
                    float(whisker_low)
                    if whisker_low is not None and not pd.isna(whisker_low)
                    else float(q1_val)
                ),
                "whishi": (
                    float(whisker_high)
                    if whisker_high is not None and not pd.isna(whisker_high)
                    else float(q3_val)
                ),
                "fliers": []
            }]

            ax.bxp(
                boxplot_stats,
                showfliers=False,
                widths=0.5,
                boxprops=dict(color="steelblue"),
                medianprops=dict(color="tomato", linewidth=1.5),
                whiskerprops=dict(color="steelblue"),
                capprops=dict(color="steelblue")
            )

            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=5)

            if col_idx == 0:
                ax.set_ylabel(
                    f"{hour:02d} h",
                    fontsize=10,
                    rotation=0,
                    labelpad=18,
                    va="center"
                )

            if row_idx == 0:
                ax.set_title(day, fontsize=10, pad=3)

    fig.suptitle(
        f"{variable.replace('_', ' ').title()} Distribution by Hour and Day",
        fontsize=12,
        y=1.002
    )

    plt.tight_layout()
    plt.show()


def plot_week_hour_heatmap(
    df_pd: pd.DataFrame,
    title="Heatmap by Day of Week and Hour of Day",
    cbar_label="Value",
    annotate=True,
    fmt=".1f"
):
    """
    Generates a heatmap with days of the week as rows and hours of the day
    as columns.

    :param df_pd: Pandas DataFrame containing seven rows (days) and
                  24 hourly columns.
    :param title: Heatmap title.
    :param cbar_label: Color bar label.
    :param annotate: Whether values should be displayed inside each cell.
    :param fmt: Number formatting used for annotations.
    """

    heatmap_data = df_pd.copy()

    heatmap_data.set_index(
        "day_of_week",
        drop=True,
        inplace=True
    )

    heatmap_data.columns = [
        int(column)
        for column in heatmap_data.columns
    ]

    # Ensure chronological weekday order
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    heatmap_data = heatmap_data.reindex(day_order)

    fig, ax = plt.subplots(figsize=(14, 4))

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="RdBu_r",
        linewidths=0.5,
        linecolor="white",
        annot=annotate,
        fmt=fmt,
        annot_kws={"size": 7},
        cbar_kws={"label": cbar_label}
    )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Day of Week", fontsize=11)

    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.show()