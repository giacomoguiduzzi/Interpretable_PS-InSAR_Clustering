import geopandas as gpd


def read_shapefile(file_path, spatial_ref, multi_geom_type):
    shape_data = gpd.read_file(file_path)

    if shape_data.crs != spatial_ref:
        shape_data = (
            shape_data.set_crs(spatial_ref)
            if shape_data.crs is None
            else shape_data.to_crs(spatial_ref)
        )

        if any(shape_data.geometry.type.isin(["Point", "MultiPoint"])):
            shape_data["LON"] = shape_data.geometry.x
            shape_data["LAT"] = shape_data.geometry.y

    if any(shape_data.geometry.type == multi_geom_type):
        shape_data = shape_data.explode(index_parts=True).reset_index(drop=True)

    return shape_data


def extract_ts(df):
    ts_columns = [
        col
        for col in df.columns
        if col.startswith("D20") or (col.startswith("20") and "/" in col)
    ]
    ts_df = df[ts_columns]
    return ts_df
